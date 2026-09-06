// Package agent implements a stateful agent runtime with tool execution and
// event streaming, built on the pi-go/ai package. It mirrors the core of
// @earendil-works/pi-agent-core (Agent, agentLoop, and their types).
package agent

import (
	"context"

	"pi-go/ai"
)

// StreamFn produces an assistant-message event stream for a model request.
// models.StreamSimple satisfies this shape.
type StreamFn func(model *ai.Model, ctx *ai.Context, opts *ai.SimpleStreamOptions) *ai.AssistantMessageEventStream

// ToolExecutionMode controls how multiple tool calls in one assistant message
// are executed.
type ToolExecutionMode = string

const (
	ToolExecutionSequential ToolExecutionMode = "sequential"
	ToolExecutionParallel   ToolExecutionMode = "parallel"
)

// QueueMode controls how many queued messages are injected at a drain point.
type QueueMode = string

const (
	QueueAll        QueueMode = "all"
	QueueOneAtATime QueueMode = "one-at-a-time"
)

// AgentMessage is the flexible transcript unit. The MVP supports the standard
// ai messages (user/assistant/toolResult); custom message types can be handled
// in ConvertToLLM.
type AgentMessage = ai.Message

// AgentToolResult is the final or partial result produced by a tool.
type AgentToolResult struct {
	Content        []ai.ContentBlock
	Details        any
	Usage          *ai.Usage
	AddedToolNames []string
	// Terminate hints the agent should stop after the current tool batch.
	Terminate bool
}

// AgentToolUpdateCallback streams partial tool execution updates.
type AgentToolUpdateCallback func(partial *AgentToolResult)

// AgentTool is a tool definition used by the agent runtime.
type AgentTool struct {
	ai.Tool

	Label string

	// PrepareArguments is an optional compatibility shim applied before
	// schema validation.
	PrepareArguments func(args map[string]any) map[string]any

	// Execute runs the tool. Return an error to report a failed call.
	Execute func(toolCallID string, params map[string]any, ctx context.Context, onUpdate AgentToolUpdateCallback) (*AgentToolResult, error)

	// Replay is a recovery policy ("never" or "safe").
	Replay string

	// ExecutionMode overrides the global mode for this tool.
	ExecutionMode ToolExecutionMode
}

// AgentContext is the context snapshot passed into the agent loop.
type AgentContext struct {
	SystemPrompt string
	Messages     []AgentMessage
	Tools        []*AgentTool
}

// State is public agent state. Tools and Messages are live slices.
type State struct {
	SystemPrompt  string
	Model         *ai.Model
	ThinkingLevel ai.ThinkingLevel
	Tools         []*AgentTool
	Messages      []AgentMessage

	IsStreaming      bool
	StreamingMessage AgentMessage
	PendingToolCalls map[string]struct{}
	ErrorMessage     string
}

// BeforeToolCallResult lets a hook block execution.
type BeforeToolCallResult struct {
	Block     bool
	Reason    string
	Terminate bool
}

// BeforeToolCallContext is passed to BeforeToolCall.
type BeforeToolCallContext struct {
	AssistantMessage *ai.AssistantMessage
	ToolCall         *ai.ContentBlock
	Args             map[string]any
	Context          *AgentContext
}

// AfterToolCallContext is passed to AfterToolCall.
type AfterToolCallContext struct {
	AssistantMessage *ai.AssistantMessage
	ToolCall         *ai.ContentBlock
	Args             map[string]any
	Result           *AgentToolResult
	IsError          bool
	Context          *AgentContext
}

// AfterToolCallResult is a partial, field-by-field override of a tool result.
// Nil fields keep the original value.
type AfterToolCallResult struct {
	Content   *[]ai.ContentBlock
	Details   *any
	IsError   *bool
	Usage     *ai.Usage
	Terminate *bool
}

// ShouldStopAfterTurnContext is passed to ShouldStopAfterTurn.
type ShouldStopAfterTurnContext struct {
	Message     *ai.AssistantMessage
	ToolResults []*ai.ToolResultMessage
	Context     *AgentContext
	NewMessages []AgentMessage
}

// PrepareNextTurnContext is passed to PrepareNextTurn.
type PrepareNextTurnContext = ShouldStopAfterTurnContext

// AgentLoopTurnUpdate replaces state before the next provider request.
type AgentLoopTurnUpdate struct {
	Context       *AgentContext
	Model         *ai.Model
	ThinkingLevel ai.ThinkingLevel
	// HasThinkingLevel distinguishes "set to off" from "leave unchanged".
	HasThinkingLevel bool
}

// AgentLoopConfig configures the low-level loop.
type AgentLoopConfig struct {
	ai.SimpleStreamOptions

	Model         *ai.Model
	ToolExecution ToolExecutionMode

	ConvertToLLM     func([]AgentMessage) []AgentMessage
	TransformContext func([]AgentMessage, context.Context) []AgentMessage
	GetAPIKey        func(provider string) string

	BeforeToolCall      func(*BeforeToolCallContext, context.Context) *BeforeToolCallResult
	AfterToolCall       func(*AfterToolCallContext, context.Context) *AfterToolCallResult
	ShouldStopAfterTurn func(*ShouldStopAfterTurnContext) bool
	PrepareNextTurn     func(*PrepareNextTurnContext) *AgentLoopTurnUpdate

	GetSteeringMessages func() []AgentMessage
	GetFollowUpMessages func() []AgentMessage
}

// Agent event type tags.
const (
	EventAgentStart     = "agent_start"
	EventAgentEnd       = "agent_end"
	EventTurnStart      = "turn_start"
	EventTurnEnd        = "turn_end"
	EventMessageStart   = "message_start"
	EventMessageUpdate  = "message_update"
	EventMessageEnd     = "message_end"
	EventToolExecStart  = "tool_execution_start"
	EventToolExecUpdate = "tool_execution_update"
	EventToolExecEnd    = "tool_execution_end"
)

// Event is an agent lifecycle event (tagged union).
type Event struct {
	Type string

	// message_start / message_end / turn_end / message_update
	Message AgentMessage
	// turn_end
	ToolResults []*ai.ToolResultMessage
	// agent_end
	Messages []AgentMessage
	// message_update
	AssistantMessageEvent *ai.Event
	// tool_execution_*
	ToolCallID    string
	ToolName      string
	Args          map[string]any
	PartialResult *AgentToolResult // tool_execution_update
	Result        *AgentToolResult // tool_execution_end
	IsError       bool
}
