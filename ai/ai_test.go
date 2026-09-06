package ai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestCalculateCost(t *testing.T) {
	model := mkModel("m", "Model", ApiAnthropicMessages, "anthropic", 3, 15, 100000, 1000, false)
	usage := &Usage{Input: 100, Output: 200, CacheRead: 10, CacheWrite: 5}
	cost := calculateCost(model, usage)
	if cost.Total <= 0 {
		t.Fatalf("expected positive cost, got %+v", cost)
	}
	// input cost = 3/1e6*100, output = 15/1e6*200
	wantInput := 3.0 / 1e6 * 100
	if absFloat(cost.Input-wantInput) > 1e-9 {
		t.Errorf("input cost = %v, want %v", cost.Input, wantInput)
	}
}

func absFloat(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}

func TestGetSupportedThinkingLevels(t *testing.T) {
	model := mkModel("m", "Model", ApiAnthropicMessages, "anthropic", 3, 15, 100000, 1000, true)
	levels := GetSupportedThinkingLevels(model)
	if len(levels) == 0 {
		t.Fatal("expected levels")
	}
	got := ClampThinkingLevel(model, "xhigh")
	if got == "off" {
		t.Errorf("clamp of xhigh should find nearest supported, got %q", got)
	}
}

func TestTransformMessagesDropsAbortedTurn(t *testing.T) {
	model := mkModel("m", "Model", ApiOpenAICompletions, "openai", 1, 2, 1000, 100, false)
	messages := []Message{
		&UserMessage{Role: "user", Content: []ContentBlock{TextBlock("hi")}},
		&AssistantMessage{Role: "assistant", StopReason: StopAborted, Content: []ContentBlock{}},
	}
	out := TransformMessages(messages, model)
	if len(out) != 1 {
		t.Fatalf("expected aborted turn dropped, got %d messages", len(out))
	}
}

func TestAnthropicStreamEndToEnd(t *testing.T) {
	body := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude-test","usage":{"input_tokens":10,"output_tokens":0}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(body))
	}))
	defer server.Close()

	model := mkModel("claude-test", "Test", ApiAnthropicMessages, "anthropic", 3, 15, 100000, 1000, false)
	model.BaseURL = server.URL

	p := CreateProvider(CreateProviderOptions{
		ID:     "anthropic",
		Models: []*Model{model},
		Auth:   ProviderAuth{APIKey: NewEnvAPIKeyAuth("Anthropic API key", "ANTHROPIC_API_KEY")},
		API:    anthropicAPI{},
	})

	c := &Context{Messages: []Message{&UserMessage{Role: "user", Content: []ContentBlock{TextBlock("hi")}}}}
	stream := p.Stream(model, c, &StreamOptions{APIKey: "test-key", Ctx: context.Background()})

	// Drain the stream through the channel and grab the final result.
	got := ""
	for e := range stream.Events() {
		if e.Type == EventTextDelta {
			got += e.Delta
		}
	}
	if got != "Hello" {
		t.Fatalf("expected streamed text %q, got %q", "Hello", got)
	}
	result := stream.Result()
	if result.StopReason != StopEnd {
		t.Fatalf("expected stop reason %q, got %q", StopEnd, result.StopReason)
	}
	if result.ResponseID != "msg_1" {
		t.Errorf("expected response id msg_1, got %q", result.ResponseID)
	}
}

func TestBuiltinModelsHasProviders(t *testing.T) {
	m := BuiltinModels()
	if len(m.GetProviders()) == 0 {
		t.Fatal("expected builtin providers")
	}
	model := m.GetModel("anthropic", "claude-sonnet-4-5")
	if model == nil {
		t.Fatal("expected claude-sonnet-4-5 model")
	}
}

func TestModelClampThinking(t *testing.T) {
	model := mkModel("m", "Model", ApiAnthropicMessages, "anthropic", 3, 15, 100000, 1000, true)
	if !containsLevel(GetSupportedThinkingLevels(model), "off") {
		t.Error("expected off to be supported")
	}
	_ = time.Now()
}
