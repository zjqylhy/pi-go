package agent_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/zjqylhy/pi-go/agent"
	"github.com/zjqylhy/pi-go/ai"
	"github.com/zjqylhy/pi-go/harness"
	"github.com/zjqylhy/pi-go/session"
)

// TestAgentToolSessionIntegration drives the full pipeline — the real
// anthropic HTTP/SSE client against a scripted LLM, the agent loop, a harness
// write tool, and session persistence — end to end.
func TestAgentToolSessionIntegration(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test-key")

	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		if calls.Add(1) == 1 {
			_, _ = w.Write([]byte(toolUseBody()))
		} else {
			_, _ = w.Write([]byte(textBody()))
		}
	}))
	defer server.Close()

	models := ai.BuiltinModels()
	model := models.GetModel("anthropic", "claude-sonnet-4-5")
	if model == nil {
		t.Fatal("expected claude-sonnet-4-5 model")
	}
	model.BaseURL = server.URL

	dir := t.TempDir()
	env := harness.NewLocalEnv(dir)

	sess, err := session.NewMemoryRepo().Create(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	a := agent.NewAgent(agent.AgentOptions{
		InitialState: &agent.State{
			SystemPrompt: "You are a coding assistant. Use tools when asked to write files.",
			Model:        model,
			Tools: []*agent.AgentTool{
				harness.CreateReadTool(env, nil),
				harness.CreateWriteTool(env),
				harness.CreateEditTool(env),
				harness.CreateBashTool(env, nil),
			},
		},
		StreamFn: models.StreamSimple,
	})

	var streamed strings.Builder
	a.Subscribe(func(ev agent.Event, cancel context.Context) {
		switch ev.Type {
		case agent.EventMessageUpdate:
			if ev.AssistantMessageEvent != nil && ev.AssistantMessageEvent.Type == ai.EventTextDelta {
				streamed.WriteString(ev.AssistantMessageEvent.Delta)
			}
		case agent.EventMessageEnd:
			if ev.Message != nil {
				_, _ = sess.AppendMessage(ev.Message, context.Background())
			}
		}
	})

	if err := a.Prompt("write a file greeting.txt containing 'hello from integration'"); err != nil {
		t.Fatalf("prompt: %v", err)
	}

	// The write tool should have actually created the file on disk.
	data, err := os.ReadFile(filepath.Join(dir, "greeting.txt"))
	if err != nil {
		t.Fatalf("expected greeting.txt to be written: %v", err)
	}
	if !strings.Contains(string(data), "hello from integration") {
		t.Fatalf("unexpected file content %q", string(data))
	}

	// The final assistant turn should have streamed text.
	if !strings.Contains(streamed.String(), "I wrote the file") {
		t.Fatalf("expected final assistant text, got %q", streamed.String())
	}

	// Two LLM round-trips: first the tool_use, then the final answer.
	if calls.Load() != 2 {
		t.Fatalf("expected 2 LLM calls, got %d", calls.Load())
	}

	// The session should persist the full transcript: prompt, tool-calling
	// assistant turn, tool result, and final assistant turn.
	msgs, err := sess.Messages(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 4 {
		t.Fatalf("expected 4 persisted messages, got %d", len(msgs))
	}
	if _, ok := msgs[2].(*ai.ToolResultMessage); !ok {
		t.Fatalf("expected 3rd message to be a tool result, got %T", msgs[2])
	}
}

func toolUseBody() string {
	return strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-4-5","usage":{"input_tokens":10,"output_tokens":0}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"write","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"greeting.txt\",\"content\":\"hello from integration\"}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
}

func textBody() string {
	return strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_2","model":"claude-sonnet-4-5","usage":{"input_tokens":10,"output_tokens":0}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"I wrote the file for you."}}`,
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
}
