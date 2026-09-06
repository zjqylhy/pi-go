package agent

import (
	"context"
	"testing"
	"time"

	"github.com/zjqylhy/pi-go/ai"
)

func testModel() *ai.Model {
	return &ai.Model{ID: "test-model", Name: "Test", API: "anthropic-messages", Provider: "test", Input: []string{"text"}}
}

func textStream(text string) *ai.AssistantMessageEventStream {
	s := ai.NewAssistantMessageEventStream()
	partial := &ai.AssistantMessage{
		Role:       "assistant",
		Content:    []ai.ContentBlock{},
		API:        "anthropic-messages",
		Provider:   "test",
		Model:      "test-model",
		StopReason: ai.StopPending,
		Timestamp:  time.Now().UnixMilli(),
	}
	go func() {
		partial.Content = append(partial.Content, ai.TextBlock(""))
		s.Push(ai.Event{Type: ai.EventStart, Partial: partial})
		s.Push(ai.Event{Type: ai.EventTextStart, ContentIndex: 0, Partial: partial})
		partial.Content[0].Text = text
		s.Push(ai.Event{Type: ai.EventTextDelta, ContentIndex: 0, Delta: text, Partial: partial})
		s.Push(ai.Event{Type: ai.EventTextEnd, ContentIndex: 0, Content: text, Partial: partial})
		partial.StopReason = ai.StopEnd
		s.Push(ai.Event{Type: ai.EventDone, Reason: ai.StopEnd, Message: partial})
	}()
	return s
}

func TestAgentPromptStreamsText(t *testing.T) {
	agent := NewAgent(AgentOptions{
		InitialState: &State{Model: testModel(), SystemPrompt: "You are helpful."},
		StreamFn: func(model *ai.Model, ctx *ai.Context, opts *ai.SimpleStreamOptions) *ai.AssistantMessageEventStream {
			return textStream("hello world")
		},
	})

	var got string
	var sawTurnEnd, sawAgentEnd bool
	agent.Subscribe(func(ev Event, ctx context.Context) {
		if ev.Type == EventMessageUpdate && ev.AssistantMessageEvent != nil {
			got += ev.AssistantMessageEvent.Delta
		}
		if ev.Type == EventTurnEnd {
			sawTurnEnd = true
		}
		if ev.Type == EventAgentEnd {
			sawAgentEnd = true
		}
	})

	if err := agent.Prompt("hi"); err != nil {
		t.Fatalf("prompt: %v", err)
	}

	if got != "hello world" {
		t.Fatalf("streamed text = %q, want %q", got, "hello world")
	}
	if !sawTurnEnd || !sawAgentEnd {
		t.Fatalf("expected turn_end and agent_end events (turnEnd=%v agentEnd=%v)", sawTurnEnd, sawAgentEnd)
	}

	// State should contain user + assistant messages.
	if len(agent.State().Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(agent.State().Messages))
	}
}

func TestAgentExecutesToolCall(t *testing.T) {
	calls := 0
	streamFn := func(model *ai.Model, ctx *ai.Context, opts *ai.SimpleStreamOptions) *ai.AssistantMessageEventStream {
		calls++
		if calls == 1 {
			return toolCallStream()
		}
		return textStream("result is 3")
	}

	var executed string
	agent := NewAgent(AgentOptions{
		InitialState: &State{Model: testModel()},
		StreamFn:     streamFn,
		BeforeToolCall: func(ctx *BeforeToolCallContext, cancel context.Context) *BeforeToolCallResult {
			return nil
		},
	})
	agent.State().Tools = []*AgentTool{{
		Tool:          ai.Tool{Name: "add", Description: "adds", Parameters: map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "integer"}, "b": map[string]any{"type": "integer"}}, "required": []any{"a", "b"}}},
		ExecutionMode: ToolExecutionSequential,
		Execute: func(toolCallID string, params map[string]any, ctx context.Context, onUpdate AgentToolUpdateCallback) (*AgentToolResult, error) {
			a := toInt(params["a"])
			b := toInt(params["b"])
			executed = intToStr(a + b)
			return &AgentToolResult{Content: []ai.ContentBlock{ai.TextBlock(executed)}, Details: map[string]any{}}, nil
		},
	}}

	if err := agent.Prompt("add 1 and 2"); err != nil {
		t.Fatalf("prompt: %v", err)
	}

	if executed != "3" {
		t.Fatalf("tool not executed or wrong result, got %q", executed)
	}
	// user + assistant(tool call) + toolResult + assistant(text)
	if len(agent.State().Messages) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(agent.State().Messages))
	}
}

func toolCallStream() *ai.AssistantMessageEventStream {
	s := ai.NewAssistantMessageEventStream()
	partial := &ai.AssistantMessage{
		Role:       "assistant",
		Content:    []ai.ContentBlock{},
		API:        "anthropic-messages",
		Provider:   "test",
		Model:      "test-model",
		StopReason: ai.StopPending,
		Timestamp:  time.Now().UnixMilli(),
	}
	go func() {
		s.Push(ai.Event{Type: ai.EventStart, Partial: partial})
		tc := ai.ToolCallBlock("call_1", "add", map[string]any{"a": 1, "b": 2})
		partial.Content = append(partial.Content, tc)
		s.Push(ai.Event{Type: ai.EventToolCallStart, ContentIndex: 0, Partial: partial})
		s.Push(ai.Event{Type: ai.EventToolCallEnd, ContentIndex: 0, ToolCall: &partial.Content[0], Partial: partial})
		partial.StopReason = ai.StopToolUse
		s.Push(ai.Event{Type: ai.EventDone, Reason: ai.StopToolUse, Message: partial})
	}()
	return s
}

func intToStr(i int) string {
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	if neg {
		b = append([]byte{'-'}, b...)
	}
	return string(b)
}

func toInt(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	case int64:
		return int(n)
	}
	return 0
}

func TestValidateToolArgumentsCoercesTypes(t *testing.T) {
	tool := &ai.Tool{
		Name:       "f",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{"n": map[string]any{"type": "integer"}}, "required": []any{"n"}},
	}
	out, err := ai.ValidateToolArguments(tool, map[string]any{"n": "42"})
	if err != nil {
		t.Fatalf("validate: %v", err)
	}
	if out["n"] != 42 && out["n"] != int(42) {
		t.Fatalf("expected coerced 42, got %v (%T)", out["n"], out["n"])
	}
}

func TestAgentLoopContinueFromUser(t *testing.T) {
	ctx := context.Background()
	model := testModel()
	msgs := []AgentMessage{&ai.UserMessage{Role: "user", Content: []ai.ContentBlock{ai.TextBlock("hi")}, Timestamp: time.Now().UnixMilli()}}
	config := AgentLoopConfig{Model: model}
	stream := AgentLoopContinue(ctx, AgentContext{SystemPrompt: "s", Messages: msgs}, config, func(model *ai.Model, ctx *ai.Context, opts *ai.SimpleStreamOptions) *ai.AssistantMessageEventStream {
		return textStream("ok")
	})

	events := []string{}
	for e := range stream.Events() {
		events = append(events, e.Type)
	}
	if len(stream.Result()) != 1 {
		t.Fatalf("expected 1 produced message, got %d", len(stream.Result()))
	}
	_ = events
}
