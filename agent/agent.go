package agent

import (
	"context"
	"sync"

	"pi-go/ai"
)

// AgentOptions configure an Agent.
type AgentOptions struct {
	InitialState *State

	ConvertToLLM     func([]AgentMessage) []AgentMessage
	TransformContext func([]AgentMessage, context.Context) []AgentMessage
	StreamFn         StreamFn
	GetAPIKey        func(provider string) string

	BeforeToolCall      func(*BeforeToolCallContext, context.Context) *BeforeToolCallResult
	AfterToolCall       func(*AfterToolCallContext, context.Context) *AfterToolCallResult
	ShouldStopAfterTurn func(*ShouldStopAfterTurnContext, context.Context) bool
	PrepareNextTurn     func(*PrepareNextTurnContext, context.Context) *AgentLoopTurnUpdate

	SteeringMode    QueueMode
	FollowUpMode    QueueMode
	SessionID       string
	ThinkingBudgets ai.ThinkingBudgets
	Transport       ai.Transport
	MaxRetryDelayMs int
	ToolExecution   ToolExecutionMode
}

// msgQueue is a message queue with an inject-all / inject-one mode.
type msgQueue struct {
	messages []AgentMessage
	mode     QueueMode
}

func newMsgQueue(mode QueueMode) *msgQueue {
	if mode == "" {
		mode = QueueOneAtATime
	}
	return &msgQueue{mode: mode}
}

func (q *msgQueue) enqueue(m AgentMessage) { q.messages = append(q.messages, m) }
func (q *msgQueue) hasItems() bool         { return len(q.messages) > 0 }

func (q *msgQueue) drain() []AgentMessage {
	if q.mode == QueueAll {
		out := q.messages
		q.messages = nil
		return out
	}
	if len(q.messages) == 0 {
		return nil
	}
	first := q.messages[0]
	q.messages = q.messages[1:]
	return []AgentMessage{first}
}

func (q *msgQueue) clear() { q.messages = nil }

type listenerEntry struct {
	fn      func(Event, context.Context)
	removed bool
}

// Agent is a stateful wrapper around the low-level loop. It owns the
// transcript, emits lifecycle events, executes tools, and exposes steering and
// follow-up queues.
type Agent struct {
	mu        sync.Mutex
	state     State
	listeners []*listenerEntry
	steering  *msgQueue
	followUp  *msgQueue

	convertToLLM        func([]AgentMessage) []AgentMessage
	transformContext    func([]AgentMessage, context.Context) []AgentMessage
	streamFunction      StreamFn
	getAPIKey           func(provider string) string
	beforeToolCall      func(*BeforeToolCallContext, context.Context) *BeforeToolCallResult
	afterToolCall       func(*AfterToolCallContext, context.Context) *AfterToolCallResult
	shouldStopAfterTurn func(*ShouldStopAfterTurnContext, context.Context) bool
	prepareNextTurn     func(*PrepareNextTurnContext, context.Context) *AgentLoopTurnUpdate

	sessionID       string
	thinkingBudgets ai.ThinkingBudgets
	transport       ai.Transport
	maxRetryDelayMs int
	toolExecution   ToolExecutionMode

	activeCancel context.CancelFunc
	activeCtx    context.Context
	activeDone   chan struct{}
}

// NewAgent constructs an Agent.
func NewAgent(options AgentOptions) *Agent {
	a := &Agent{}
	if options.InitialState != nil {
		a.state = *options.InitialState
	}
	if a.state.Model == nil {
		a.state.Model = unknownModel()
	}
	if a.state.PendingToolCalls == nil {
		a.state.PendingToolCalls = map[string]struct{}{}
	}
	a.convertToLLM = options.ConvertToLLM
	a.transformContext = options.TransformContext
	a.streamFunction = options.StreamFn
	a.getAPIKey = options.GetAPIKey
	a.beforeToolCall = options.BeforeToolCall
	a.afterToolCall = options.AfterToolCall
	a.shouldStopAfterTurn = options.ShouldStopAfterTurn
	a.prepareNextTurn = options.PrepareNextTurn
	a.steering = newMsgQueue(options.SteeringMode)
	a.followUp = newMsgQueue(options.FollowUpMode)
	a.sessionID = options.SessionID
	a.thinkingBudgets = options.ThinkingBudgets
	if options.Transport == "" {
		a.transport = ai.TransportAuto
	} else {
		a.transport = options.Transport
	}
	a.maxRetryDelayMs = options.MaxRetryDelayMs
	if options.ToolExecution == "" {
		a.toolExecution = ToolExecutionParallel
	} else {
		a.toolExecution = options.ToolExecution
	}
	return a
}

func unknownModel() *ai.Model {
	return &ai.Model{ID: "unknown", Name: "unknown", API: "unknown", Provider: "unknown", Input: []string{}}
}

// Subscribe registers a lifecycle listener. Returns an unsubscribe func.
func (a *Agent) Subscribe(listener func(Event, context.Context)) func() {
	a.mu.Lock()
	entry := &listenerEntry{fn: listener}
	a.listeners = append(a.listeners, entry)
	a.mu.Unlock()
	return func() {
		a.mu.Lock()
		entry.removed = true
		a.mu.Unlock()
	}
}

// State returns the live agent state pointer.
func (a *Agent) State() *State {
	return &a.state
}

// SteeringMode returns the steering queue mode.
func (a *Agent) SteeringMode() QueueMode { return a.steering.mode }

// SetSteeringMode sets the steering queue mode.
func (a *Agent) SetSteeringMode(m QueueMode) { a.steering.mode = m }

// FollowUpMode returns the follow-up queue mode.
func (a *Agent) FollowUpMode() QueueMode { return a.followUp.mode }

// SetFollowUpMode sets the follow-up queue mode.
func (a *Agent) SetFollowUpMode(m QueueMode) { a.followUp.mode = m }

// Steer queues a message to inject after the current turn.
func (a *Agent) Steer(m AgentMessage) { a.steering.enqueue(m) }

// FollowUp queues a message to run once the agent would otherwise stop.
func (a *Agent) FollowUp(m AgentMessage) { a.followUp.enqueue(m) }

// ClearSteeringQueue drops queued steering messages.
func (a *Agent) ClearSteeringQueue() { a.steering.clear() }

// ClearFollowUpQueue drops queued follow-up messages.
func (a *Agent) ClearFollowUpQueue() { a.followUp.clear() }

// ClearAllQueues drops all queued messages.
func (a *Agent) ClearAllQueues() {
	a.ClearSteeringQueue()
	a.ClearFollowUpQueue()
}

// HasQueuedMessages reports whether either queue still has pending messages.
func (a *Agent) HasQueuedMessages() bool {
	return a.steering.hasItems() || a.followUp.hasItems()
}

// Abort cancels the current run, if any.
func (a *Agent) Abort() {
	if a.activeCancel != nil {
		a.activeCancel()
	}
}

// WaitForIdle blocks until the current run (and its listeners) finishes.
func (a *Agent) WaitForIdle() {
	if a.activeDone != nil {
		<-a.activeDone
	}
}

// Reset clears transcript and runtime state. It errors if a run is active.
func (a *Agent) Reset() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.activeDone != nil {
		select {
		case <-a.activeDone:
		default:
			return cannotContinueError{msg: "agent is already processing; cannot reset"}
		}
	}
	a.state.Messages = nil
	a.state.IsStreaming = false
	a.state.StreamingMessage = nil
	a.state.PendingToolCalls = map[string]struct{}{}
	a.state.ErrorMessage = ""
	a.ClearAllQueues()
	return nil
}

// Prompt starts a new run from text, a message, or a batch of messages.
func (a *Agent) Prompt(input any, images ...ai.ContentBlock) error {
	if err := a.ensureIdle(); err != nil {
		return err
	}
	messages, err := normalizePromptInput(input, images)
	if err != nil {
		return err
	}
	return a.runPromptMessages(messages, false)
}

// Continue resumes from the current transcript. The last message must be a
// user or tool-result message.
func (a *Agent) Continue() error {
	if err := a.ensureIdle(); err != nil {
		return err
	}
	msgs := a.state.Messages
	if len(msgs) == 0 {
		return errCannotContinue("no messages to continue from")
	}
	if messageRole(msgs[len(msgs)-1]) == "assistant" {
		if queued := a.steering.drain(); len(queued) > 0 {
			return a.runPromptMessages(queued, true)
		}
		if queued := a.followUp.drain(); len(queued) > 0 {
			return a.runPromptMessages(queued, false)
		}
		return errCannotContinue("cannot continue from message role: assistant")
	}
	return a.runContinuation()
}

func (a *Agent) ensureIdle() error {
	if a.activeDone != nil {
		select {
		case <-a.activeDone:
		default:
			return cannotContinueError{msg: "agent is already processing a prompt"}
		}
	}
	return nil
}

func normalizePromptInput(input any, images []ai.ContentBlock) ([]AgentMessage, error) {
	switch v := input.(type) {
	case []AgentMessage:
		return v, nil
	case AgentMessage:
		return []AgentMessage{v}, nil
	case string:
		content := []ai.ContentBlock{ai.TextBlock(v)}
		content = append(content, images...)
		return []AgentMessage{&ai.UserMessage{Role: "user", Content: content, Timestamp: nowMs()}}, nil
	default:
		return nil, cannotContinueError{msg: "unsupported prompt input type"}
	}
}

func (a *Agent) runPromptMessages(messages []AgentMessage, skipInitialSteeringPoll bool) error {
	return a.runWithLifecycle(func(ctx context.Context) error {
		_, err := runAgentLoop(ctx, messages, a.snapshot(), a.loopConfig(skipInitialSteeringPoll), a.processEvents, a.streamFunction)
		return err
	})
}

func (a *Agent) runContinuation() error {
	return a.runWithLifecycle(func(ctx context.Context) error {
		_, err := runAgentLoopContinue(ctx, a.snapshot(), a.loopConfig(false), a.processEvents, a.streamFunction)
		return err
	})
}

func (a *Agent) snapshot() AgentContext {
	a.mu.Lock()
	defer a.mu.Unlock()
	return AgentContext{
		SystemPrompt: a.state.SystemPrompt,
		Messages:     append([]AgentMessage{}, a.state.Messages...),
		Tools:        append([]*AgentTool{}, a.state.Tools...),
	}
}

func (a *Agent) loopConfig(skipInitialSteeringPoll bool) AgentLoopConfig {
	skip := skipInitialSteeringPoll
	cfg := AgentLoopConfig{
		Model:            a.state.Model,
		ToolExecution:    a.toolExecution,
		ConvertToLLM:     a.convertToLLM,
		TransformContext: a.transformContext,
		GetAPIKey:        a.getAPIKey,
		BeforeToolCall:   a.beforeToolCall,
		AfterToolCall:    a.afterToolCall,
	}
	cfg.SimpleStreamOptions.Reasoning = reasoningFor(a.state.ThinkingLevel)
	cfg.SimpleStreamOptions.SessionID = a.sessionID
	cfg.SimpleStreamOptions.ThinkingBudgets = a.thinkingBudgets
	cfg.SimpleStreamOptions.Transport = a.transport
	cfg.SimpleStreamOptions.MaxRetryDelayMs = a.maxRetryDelayMs

	if a.shouldStopAfterTurn != nil {
		cfg.ShouldStopAfterTurn = func(ctx *ShouldStopAfterTurnContext) bool {
			return a.shouldStopAfterTurn(ctx, a.activeCtx)
		}
	}
	if a.prepareNextTurn != nil {
		cfg.PrepareNextTurn = func(ctx *PrepareNextTurnContext) *AgentLoopTurnUpdate {
			return a.prepareNextTurn(ctx, a.activeCtx)
		}
	}
	cfg.GetSteeringMessages = func() []AgentMessage {
		if skip {
			skip = false
			return nil
		}
		return a.steering.drain()
	}
	cfg.GetFollowUpMessages = func() []AgentMessage {
		return a.followUp.drain()
	}
	return cfg
}

func reasoningFor(level ai.ThinkingLevel) ai.ThinkingLevel {
	if level == "" || level == ai.ThinkingOff {
		return ""
	}
	return level
}

func (a *Agent) runWithLifecycle(executor func(ctx context.Context) error) error {
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	a.mu.Lock()
	a.activeCtx = ctx
	a.activeCancel = cancel
	a.activeDone = done
	a.state.IsStreaming = true
	a.state.StreamingMessage = nil
	a.state.ErrorMessage = ""
	a.mu.Unlock()

	err := executor(ctx)
	if err != nil && ctx.Err() == nil {
		a.handleRunFailure(err)
	}

	a.mu.Lock()
	a.state.IsStreaming = false
	a.state.StreamingMessage = nil
	a.state.PendingToolCalls = map[string]struct{}{}
	close(done)
	a.activeDone = nil
	a.activeCancel = nil
	a.activeCtx = nil
	a.mu.Unlock()
	return err
}

func (a *Agent) handleRunFailure(err error) {
	failure := &ai.AssistantMessage{
		Role:         "assistant",
		Content:      []ai.ContentBlock{},
		API:          a.state.Model.API,
		Provider:     a.state.Model.Provider,
		Model:        a.state.Model.ID,
		Usage:        ai.Usage{},
		StopReason:   ai.StopError,
		ErrorMessage: err.Error(),
		Timestamp:    nowMs(),
	}
	_ = a.processEvents(Event{Type: EventMessageStart, Message: failure})
	_ = a.processEvents(Event{Type: EventMessageEnd, Message: failure})
	_ = a.processEvents(Event{Type: EventTurnEnd, Message: failure, ToolResults: nil})
	_ = a.processEvents(Event{Type: EventAgentEnd, Messages: []AgentMessage{failure}})
}

func (a *Agent) processEvents(ev Event) error {
	a.mu.Lock()
	switch ev.Type {
	case EventMessageStart:
		a.state.StreamingMessage = ev.Message
	case EventMessageUpdate:
		a.state.StreamingMessage = ev.Message
	case EventMessageEnd:
		a.state.StreamingMessage = nil
		a.state.Messages = append(a.state.Messages, ev.Message)
	case EventToolExecStart:
		if a.state.PendingToolCalls == nil {
			a.state.PendingToolCalls = map[string]struct{}{}
		}
		a.state.PendingToolCalls[ev.ToolCallID] = struct{}{}
	case EventToolExecEnd:
		delete(a.state.PendingToolCalls, ev.ToolCallID)
	case EventTurnEnd:
		if ev.Message != nil {
			if am, ok := ev.Message.(*ai.AssistantMessage); ok && am.ErrorMessage != "" {
				a.state.ErrorMessage = am.ErrorMessage
			}
		}
	case EventAgentEnd:
		a.state.StreamingMessage = nil
	}
	listeners := append([]*listenerEntry{}, a.listeners...)
	a.mu.Unlock()

	for _, l := range listeners {
		if l.removed {
			continue
		}
		l.fn(ev, a.activeCtx)
	}
	return nil
}
