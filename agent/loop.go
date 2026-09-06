package agent

import (
	"context"
	"sync"
	"time"

	"pi-go/ai"
)

// AgentEventSink is the synchronous event consumer used by the low-level loop.
type AgentEventSink = func(Event) error

const defaultToolExecution = ToolExecutionParallel

// AgentLoop starts a new agent loop with prompt messages and returns an
// observable stream whose Result resolves to the produced messages.
func AgentLoop(ctx context.Context, prompts []AgentMessage, context AgentContext, config AgentLoopConfig, streamFn StreamFn) *EventStream {
	stream := newEventStream()
	go func() {
		messages, _ := runAgentLoop(ctx, prompts, context, config, func(e Event) error {
			stream.Push(e)
			return nil
		}, streamFn)
		stream.End(messages)
	}()
	return stream
}

// AgentLoopContinue resumes from an existing context without a new message.
func AgentLoopContinue(ctx context.Context, context AgentContext, config AgentLoopConfig, streamFn StreamFn) *EventStream {
	stream := newEventStream()
	go func() {
		messages, _ := runAgentLoopContinue(ctx, context, config, func(e Event) error {
			stream.Push(e)
			return nil
		}, streamFn)
		stream.End(messages)
	}()
	return stream
}

func runAgentLoop(ctx context.Context, prompts []AgentMessage, context AgentContext, config AgentLoopConfig, emit AgentEventSink, streamFn StreamFn) ([]AgentMessage, error) {
	newMessages := append([]AgentMessage{}, prompts...)
	current := AgentContext{
		SystemPrompt: context.SystemPrompt,
		Messages:     append(append([]AgentMessage{}, context.Messages...), prompts...),
		Tools:        context.Tools,
	}

	if err := emit(Event{Type: EventAgentStart}); err != nil {
		return newMessages, err
	}
	if err := emit(Event{Type: EventTurnStart}); err != nil {
		return newMessages, err
	}
	for _, p := range prompts {
		if err := emit(Event{Type: EventMessageStart, Message: p}); err != nil {
			return newMessages, err
		}
		if err := emit(Event{Type: EventMessageEnd, Message: p}); err != nil {
			return newMessages, err
		}
	}

	if err := runLoop(ctx, &current, &newMessages, config, emit, streamFn); err != nil {
		return newMessages, err
	}
	return newMessages, nil
}

func runAgentLoopContinue(ctx context.Context, context AgentContext, config AgentLoopConfig, emit AgentEventSink, streamFn StreamFn) ([]AgentMessage, error) {
	if len(context.Messages) == 0 {
		return nil, errCannotContinue("no messages in context")
	}
	if messageRole(context.Messages[len(context.Messages)-1]) == "assistant" {
		return nil, errCannotContinue("last message is assistant")
	}

	newMessages := []AgentMessage{}
	current := context
	if err := emit(Event{Type: EventAgentStart}); err != nil {
		return newMessages, err
	}
	if err := emit(Event{Type: EventTurnStart}); err != nil {
		return newMessages, err
	}
	if err := runLoop(ctx, &current, &newMessages, config, emit, streamFn); err != nil {
		return newMessages, err
	}
	return newMessages, nil
}

type cannotContinueError struct{ msg string }

func (e cannotContinueError) Error() string { return e.msg }
func errCannotContinue(msg string) error    { return cannotContinueError{msg: msg} }

func runLoop(ctx context.Context, current *AgentContext, newMessages *[]AgentMessage, config AgentLoopConfig, emit AgentEventSink, streamFn StreamFn) error {
	var lastCompletedTurn *ShouldStopAfterTurnContext

	pending := []AgentMessage{}
	if config.GetSteeringMessages != nil {
		pending = append(pending, config.GetSteeringMessages()...)
	}

	for {
		hasMoreToolCalls := true

		for hasMoreToolCalls || len(pending) > 0 {
			if lastCompletedTurn != nil {
				if config.PrepareNextTurn != nil {
					if upd := config.PrepareNextTurn(lastCompletedTurn); upd != nil {
						if upd.Context != nil {
							*current = *upd.Context
						}
						if upd.Model != nil {
							config.Model = upd.Model
						}
						if upd.HasThinkingLevel {
							config.Reasoning = upd.ThinkingLevel
						}
					}
				}
				if len(pending) == 0 && config.GetSteeringMessages != nil {
					pending = append(pending, config.GetSteeringMessages()...)
				}
				if err := emit(Event{Type: EventTurnStart}); err != nil {
					return err
				}
			}

			if len(pending) > 0 {
				for _, m := range pending {
					if err := emit(Event{Type: EventMessageStart, Message: m}); err != nil {
						return err
					}
					if err := emit(Event{Type: EventMessageEnd, Message: m}); err != nil {
						return err
					}
					current.Messages = append(current.Messages, m)
					*newMessages = append(*newMessages, m)
				}
				pending = nil
			}

			message, err := streamAssistantResponse(current, config, ctx, emit, streamFn)
			if err != nil {
				return err
			}
			*newMessages = append(*newMessages, message)

			if message.StopReason == ai.StopError || message.StopReason == ai.StopAborted {
				if err := emit(Event{Type: EventTurnEnd, Message: message, ToolResults: nil}); err != nil {
					return err
				}
				return emit(Event{Type: EventAgentEnd, Messages: *newMessages})
			}

			toolCalls := filterToolCalls(message)
			var toolResults []*ai.ToolResultMessage
			hasMoreToolCalls = false
			if len(toolCalls) > 0 {
				var batch *executedToolBatch
				var err error
				if message.StopReason == ai.StopLength {
					batch, err = failToolCallsFromTruncatedMessage(toolCalls, emit)
				} else {
					batch, err = executeToolCalls(current, message, config, ctx, emit)
				}
				if err != nil {
					return err
				}
				toolResults = batch.messages
				hasMoreToolCalls = !batch.terminate

				for _, tr := range toolResults {
					current.Messages = append(current.Messages, tr)
					*newMessages = append(*newMessages, tr)
				}
			}

			if err := emit(Event{Type: EventTurnEnd, Message: message, ToolResults: toolResults}); err != nil {
				return err
			}

			lastCompletedTurn = &ShouldStopAfterTurnContext{
				Message:     message,
				ToolResults: toolResults,
				Context:     current,
				NewMessages: *newMessages,
			}

			if config.ShouldStopAfterTurn != nil && config.ShouldStopAfterTurn(lastCompletedTurn) {
				return emit(Event{Type: EventAgentEnd, Messages: *newMessages})
			}

			pending = nil
			if config.GetSteeringMessages != nil {
				pending = append(pending, config.GetSteeringMessages()...)
			}
		}

		var followUps []AgentMessage
		if config.GetFollowUpMessages != nil {
			followUps = config.GetFollowUpMessages()
		}
		if len(followUps) > 0 {
			pending = followUps
			continue
		}
		break
	}

	return emit(Event{Type: EventAgentEnd, Messages: *newMessages})
}

func filterToolCalls(message *ai.AssistantMessage) []*ai.ContentBlock {
	var out []*ai.ContentBlock
	for i := range message.Content {
		if message.Content[i].Type == ai.ContentTypeToolCall {
			out = append(out, &message.Content[i])
		}
	}
	return out
}

func messageRole(m AgentMessage) string {
	switch m.(type) {
	case *ai.UserMessage, ai.UserMessage:
		return "user"
	case *ai.AssistantMessage, ai.AssistantMessage:
		return "assistant"
	case *ai.ToolResultMessage, ai.ToolResultMessage:
		return "toolResult"
	default:
		return "custom"
	}
}

func defaultConvertToLLM(messages []AgentMessage) []AgentMessage {
	out := make([]AgentMessage, 0, len(messages))
	for _, m := range messages {
		switch messageRole(m) {
		case "user", "assistant", "toolResult":
			out = append(out, m)
		}
	}
	return out
}

func streamAssistantResponse(context *AgentContext, config AgentLoopConfig, ctx context.Context, emit AgentEventSink, streamFn StreamFn) (*ai.AssistantMessage, error) {
	messages := context.Messages
	if config.TransformContext != nil {
		messages = config.TransformContext(messages, ctx)
	}
	convert := config.ConvertToLLM
	if convert == nil {
		convert = defaultConvertToLLM
	}
	llmMessages := convert(messages)

	llmContext := &ai.Context{
		SystemPrompt: context.SystemPrompt,
		Messages:     llmMessages,
		Tools:        agentToolsToAITools(context.Tools),
	}

	apiKey := config.APIKey
	if config.GetAPIKey != nil && config.Model != nil {
		if k := config.GetAPIKey(config.Model.Provider); k != "" {
			apiKey = k
		}
	}

	opts := config.SimpleStreamOptions
	opts.Ctx = ctx
	opts.APIKey = apiKey

	response := streamFn(config.Model, llmContext, &opts)
	if response == nil {
		return nil, cannotContinueError{msg: "stream function returned nil"}
	}

	var partialMessage *ai.AssistantMessage
	addedPartial := false

	for e := range response.Events() {
		switch e.Type {
		case ai.EventStart:
			partialMessage = e.Partial
			context.Messages = append(context.Messages, partialMessage)
			addedPartial = true
			if err := emit(Event{Type: EventMessageStart, Message: cloneAssistant(partialMessage)}); err != nil {
				return nil, err
			}
		case ai.EventTextStart, ai.EventTextDelta, ai.EventTextEnd,
			ai.EventThinkingStart, ai.EventThinkingDelta, ai.EventThinkingEnd,
			ai.EventToolCallStart, ai.EventToolCallDelta, ai.EventToolCallEnd:
			if partialMessage != nil {
				partialMessage = e.Partial
				context.Messages[len(context.Messages)-1] = partialMessage
				ev := e
				if err := emit(Event{Type: EventMessageUpdate, Message: cloneAssistant(partialMessage), AssistantMessageEvent: &ev}); err != nil {
					return nil, err
				}
			}
		case ai.EventDone, ai.EventError:
			finalMessage := response.Result()
			if addedPartial {
				context.Messages[len(context.Messages)-1] = finalMessage
			} else {
				context.Messages = append(context.Messages, finalMessage)
			}
			if !addedPartial {
				if err := emit(Event{Type: EventMessageStart, Message: cloneAssistant(finalMessage)}); err != nil {
					return nil, err
				}
			}
			if err := emit(Event{Type: EventMessageEnd, Message: finalMessage}); err != nil {
				return nil, err
			}
			return finalMessage, nil
		}
	}

	finalMessage := response.Result()
	if addedPartial {
		context.Messages[len(context.Messages)-1] = finalMessage
	} else {
		context.Messages = append(context.Messages, finalMessage)
		if err := emit(Event{Type: EventMessageStart, Message: cloneAssistant(finalMessage)}); err != nil {
			return nil, err
		}
	}
	if err := emit(Event{Type: EventMessageEnd, Message: finalMessage}); err != nil {
		return nil, err
	}
	return finalMessage, nil
}

func cloneAssistant(m *ai.AssistantMessage) *ai.AssistantMessage {
	if m == nil {
		return nil
	}
	cp := *m
	cp.Content = append([]ai.ContentBlock{}, m.Content...)
	return &cp
}

func agentToolsToAITools(tools []*AgentTool) []ai.Tool {
	out := make([]ai.Tool, 0, len(tools))
	for _, t := range tools {
		out = append(out, t.Tool)
	}
	return out
}

// executedToolBatch is the outcome of a tool-call batch.
type executedToolBatch struct {
	messages  []*ai.ToolResultMessage
	terminate bool
}

func failToolCallsFromTruncatedMessage(toolCalls []*ai.ContentBlock, emit AgentEventSink) (*executedToolBatch, error) {
	messages := make([]*ai.ToolResultMessage, 0, len(toolCalls))
	for _, tc := range toolCalls {
		if err := emit(Event{Type: EventToolExecStart, ToolCallID: tc.ID, ToolName: tc.Name, Args: tc.Arguments}); err != nil {
			return nil, err
		}
		finalized := &finalizedToolOutcome{
			toolCall: tc,
			result:   createErrorToolResult(`Tool call "` + tc.Name + `" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.`),
			isError:  true,
		}
		if err := emitToolExecutionEnd(finalized, emit); err != nil {
			return nil, err
		}
		msg := createToolResultMessage(finalized)
		if err := emitTemplateResultMessage(msg, emit); err != nil {
			return nil, err
		}
		messages = append(messages, msg)
	}
	return &executedToolBatch{messages: messages, terminate: false}, nil
}

func executeToolCalls(current *AgentContext, assistantMessage *ai.AssistantMessage, config AgentLoopConfig, ctx context.Context, emit AgentEventSink) (*executedToolBatch, error) {
	toolCalls := filterToolCalls(assistantMessage)
	hasSequential := false
	for _, tc := range toolCalls {
		if toolByType(current, tc.Name).ExecutionMode == ToolExecutionSequential {
			hasSequential = true
			break
		}
	}
	mode := config.ToolExecution
	if mode == "" {
		mode = defaultToolExecution
	}
	if mode == ToolExecutionSequential || hasSequential {
		return executeToolCallsSequential(current, assistantMessage, toolCalls, config, ctx, emit)
	}
	return executeToolCallsParallel(current, assistantMessage, toolCalls, config, ctx, emit)
}

func toolByType(current *AgentContext, name string) *AgentTool {
	for _, t := range current.Tools {
		if t.Name == name {
			return t
		}
	}
	return &AgentTool{}
}

type finalizedToolOutcome struct {
	toolCall *ai.ContentBlock
	result   *AgentToolResult
	isError  bool
}

func executeToolCallsSequential(current *AgentContext, assistantMessage *ai.AssistantMessage, toolCalls []*ai.ContentBlock, config AgentLoopConfig, ctx context.Context, emit AgentEventSink) (*executedToolBatch, error) {
	var finals []*finalizedToolOutcome
	messages := make([]*ai.ToolResultMessage, 0, len(toolCalls))

	for _, tc := range toolCalls {
		if err := emit(Event{Type: EventToolExecStart, ToolCallID: tc.ID, ToolName: tc.Name, Args: tc.Arguments}); err != nil {
			return nil, err
		}
		prep := prepareToolCall(current, assistantMessage, tc, config, ctx)
		var fin *finalizedToolOutcome
		if prep.prepared == nil {
			fin = &finalizedToolOutcome{toolCall: tc, result: prep.result, isError: prep.isError}
		} else {
			exec := executePreparedToolCall(prep.prepared, ctx, emit)
			fin = finalizeExecutedToolCall(current, assistantMessage, prep.prepared, exec, config, ctx)
		}
		if err := emitToolExecutionEnd(fin, emit); err != nil {
			return nil, err
		}
		msg := createToolResultMessage(fin)
		if err := emitTemplateResultMessage(msg, emit); err != nil {
			return nil, err
		}
		finals = append(finals, fin)
		messages = append(messages, msg)

		if ctx.Err() != nil {
			break
		}
	}

	return &executedToolBatch{messages: messages, terminate: shouldTerminateToolBatch(finals)}, nil
}

func executeToolCallsParallel(current *AgentContext, assistantMessage *ai.AssistantMessage, toolCalls []*ai.ContentBlock, config AgentLoopConfig, ctx context.Context, emit AgentEventSink) (*executedToolBatch, error) {
	n := len(toolCalls)
	finals := make([]*finalizedToolOutcome, n)
	emittedEnd := make([]bool, n)

	var emitMu sync.Mutex
	safeEmit := func(e Event) error {
		emitMu.Lock()
		defer emitMu.Unlock()
		return emit(e)
	}

	type toExec struct {
		idx  int
		prep *preparedToolCall
	}
	var execs []toExec

	for i, tc := range toolCalls {
		if err := safeEmit(Event{Type: EventToolExecStart, ToolCallID: tc.ID, ToolName: tc.Name, Args: tc.Arguments}); err != nil {
			return nil, err
		}
		prep := prepareToolCall(current, assistantMessage, tc, config, ctx)
		if prep.prepared == nil {
			fin := &finalizedToolOutcome{toolCall: tc, result: prep.result, isError: prep.isError}
			finals[i] = fin
			emittedEnd[i] = true
			if err := safeEmit(eventForEnd(fin)); err != nil {
				return nil, err
			}
			continue
		}
		execs = append(execs, toExec{idx: i, prep: prep.prepared})
	}

	var wg sync.WaitGroup
	for _, ex := range execs {
		wg.Add(1)
		go func(ex toExec) {
			defer wg.Done()
			if ctx.Err() != nil {
				finals[ex.idx] = &finalizedToolOutcome{toolCall: ex.prep.toolCall, result: createErrorToolResult("Operation aborted"), isError: true}
				return
			}
			exec := executePreparedToolCall(ex.prep, ctx, safeEmit)
			finals[ex.idx] = finalizeExecutedToolCall(current, assistantMessage, ex.prep, exec, config, ctx)
		}(ex)
	}
	wg.Wait()

	messages := make([]*ai.ToolResultMessage, 0, n)
	var finalList []*finalizedToolOutcome
	for i, fin := range finals {
		if fin == nil {
			continue
		}
		if !emittedEnd[i] {
			if err := emit(eventForEnd(fin)); err != nil {
				return nil, err
			}
		}
		msg := createToolResultMessage(fin)
		if err := emitTemplateResultMessage(msg, emit); err != nil {
			return nil, err
		}
		messages = append(messages, msg)
		finalList = append(finalList, fin)
	}

	return &executedToolBatch{messages: messages, terminate: shouldTerminateToolBatch(finalList)}, nil
}

func eventForEnd(fin *finalizedToolOutcome) Event {
	return Event{Type: EventToolExecEnd, ToolCallID: fin.toolCall.ID, ToolName: fin.toolCall.Name, Result: fin.result, IsError: fin.isError}
}

func emitToolExecutionEnd(fin *finalizedToolOutcome, emit AgentEventSink) error {
	return emit(eventForEnd(fin))
}

func emitTemplateResultMessage(msg *ai.ToolResultMessage, emit AgentEventSink) error {
	if err := emit(Event{Type: EventMessageStart, Message: msg}); err != nil {
		return err
	}
	return emit(Event{Type: EventMessageEnd, Message: msg})
}

func shouldTerminateToolBatch(finals []*finalizedToolOutcome) bool {
	if len(finals) == 0 {
		return false
	}
	for _, f := range finals {
		if f.result == nil || !f.result.Terminate {
			return false
		}
	}
	return true
}

type preparedToolCall struct {
	toolCall *ai.ContentBlock
	tool     *AgentTool
	args     map[string]any
}

type preparedResult struct {
	prepared *preparedToolCall
	result   *AgentToolResult
	isError  bool
}

func prepareToolCall(current *AgentContext, assistantMessage *ai.AssistantMessage, toolCall *ai.ContentBlock, config AgentLoopConfig, ctx context.Context) *preparedResult {
	tool := toolByType(current, toolCall.Name)
	if tool.Name == "" {
		return &preparedResult{result: createErrorToolResult("Tool " + toolCall.Name + " not found"), isError: true}
	}

	args := toolCall.Arguments
	if tool.PrepareArguments != nil {
		args = tool.PrepareArguments(args)
	}

	validated, err := ai.ValidateToolArguments(&tool.Tool, args)
	if err != nil {
		return &preparedResult{result: createErrorToolResult(err.Error()), isError: true}
	}

	if config.BeforeToolCall != nil {
		before := config.BeforeToolCall(&BeforeToolCallContext{
			AssistantMessage: assistantMessage,
			ToolCall:         toolCall,
			Args:             validated,
			Context:          current,
		}, ctx)
		if ctx.Err() != nil {
			return &preparedResult{result: createErrorToolResult("Operation aborted"), isError: true}
		}
		if before != nil && before.Block {
			reason := before.Reason
			if reason == "" {
				reason = "Tool execution was blocked"
			}
			res := createErrorToolResult(reason)
			if before.Terminate {
				res.Terminate = true
			}
			return &preparedResult{result: res, isError: true}
		}
	}

	if ctx.Err() != nil {
		return &preparedResult{result: createErrorToolResult("Operation aborted"), isError: true}
	}

	return &preparedResult{
		prepared: &preparedToolCall{toolCall: toolCall, tool: tool, args: validated},
	}
}

func executePreparedToolCall(prepared *preparedToolCall, ctx context.Context, emit AgentEventSink) *AgentToolResult {
	acceptingUpdates := true
	onUpdate := func(partial *AgentToolResult) {
		if !acceptingUpdates {
			return
		}
		_ = emit(Event{Type: EventToolExecUpdate, ToolCallID: prepared.toolCall.ID, ToolName: prepared.toolCall.Name, Args: prepared.toolCall.Arguments, PartialResult: partial})
	}

	result, err := prepared.tool.Execute(prepared.toolCall.ID, prepared.args, ctx, onUpdate)
	acceptingUpdates = false
	if err != nil {
		return createErrorToolResult(err.Error())
	}
	return result
}

func finalizeExecutedToolCall(current *AgentContext, assistantMessage *ai.AssistantMessage, prepared *preparedToolCall, executed *AgentToolResult, config AgentLoopConfig, ctx context.Context) *finalizedToolOutcome {
	result := executed
	isError := false

	if config.AfterToolCall != nil {
		after := config.AfterToolCall(&AfterToolCallContext{
			AssistantMessage: assistantMessage,
			ToolCall:         prepared.toolCall,
			Args:             prepared.args,
			Result:           result,
			IsError:          isError,
			Context:          current,
		}, ctx)
		if after != nil {
			if result == nil {
				result = &AgentToolResult{}
			}
			if after.Content != nil {
				result.Content = *after.Content
			}
			if after.Details != nil {
				result.Details = *after.Details
			}
			if after.Usage != nil {
				result.Usage = after.Usage
			}
			if after.Terminate != nil {
				result.Terminate = *after.Terminate
			}
			if after.IsError != nil {
				isError = *after.IsError
			}
		}
	}

	if result == nil {
		result = &AgentToolResult{}
	}
	return &finalizedToolOutcome{toolCall: prepared.toolCall, result: result, isError: isError}
}

func createErrorToolResult(message string) *AgentToolResult {
	if message == "" {
		message = "Tool execution was blocked"
	}
	return &AgentToolResult{
		Content: []ai.ContentBlock{ai.TextBlock(message)},
		Details: map[string]any{},
	}
}

func createToolResultMessage(fin *finalizedToolOutcome) *ai.ToolResultMessage {
	content := []ai.ContentBlock{}
	if fin.result != nil {
		content = fin.result.Content
	}
	msg := &ai.ToolResultMessage{
		Role:       "toolResult",
		ToolCallID: fin.toolCall.ID,
		ToolName:   fin.toolCall.Name,
		Content:    content,
		IsError:    fin.isError,
		Timestamp:  nowMs(),
	}
	if fin.result != nil {
		msg.Details = fin.result.Details
		msg.Usage = fin.result.Usage
		if len(fin.result.AddedToolNames) > 0 {
			msg.AddedToolNames = fin.result.AddedToolNames
		}
	}
	return msg
}

func nowMs() int64 { return time.Now().UnixMilli() }
