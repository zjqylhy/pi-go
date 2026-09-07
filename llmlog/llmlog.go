// Package llmlog records LLM requests and responses to a local JSONL file.
//
// It registers a StreamSimpleHook via init(), so importing this package for its
// side effect enables logging without modifying any call site:
//
//	import _ "github.com/zjqylhy/pi-go/llmlog"
//
// The hook is built on the ai package's per-request OnPayload/OnResponse
// extension points (mirroring the upstream `onPayload`/`onResponse` callbacks):
// the request body is captured from OnPayload right before it is sent, the HTTP
// status/headers from OnResponse, and the final answer text from the completed
// stream.
package llmlog

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/zjqylhy/pi-go/ai"
)

// LogEntry is a single JSONL record written to the log file.
type LogEntry struct {
	Timestamp int64  `json:"timestamp"`
	Direction string `json:"direction"` // "request" or "response"

	// Request fields
	Model    string `json:"model,omitempty"`
	Provider string `json:"provider,omitempty"`
	Payload  any    `json:"payload,omitempty"`

	// Response fields
	Status       int               `json:"status,omitempty"`
	Headers      map[string]string `json:"headers,omitempty"`
	StopReason   string            `json:"stopReason,omitempty"`
	ErrorMessage string            `json:"errorMessage,omitempty"`
	Usage        *ai.Usage         `json:"usage,omitempty"`
	Content      string            `json:"content,omitempty"`
}

// Logger writes request/response records to a daily JSONL file.
type Logger struct {
	dir string

	mu       sync.Mutex
	file     *os.File
	fileDate string
}

// New returns a Logger writing into dir. An empty dir resolves to the value of
// PI_GO_LLM_LOG_DIR, falling back to ~/.pi-go/logs.
func New(dir string) *Logger {
	if dir == "" {
		dir = os.Getenv("PI_GO_LLM_LOG_DIR")
	}
	if dir == "" {
		if home, err := os.UserHomeDir(); err == nil {
			dir = filepath.Join(home, ".pi-go", "logs")
		}
	}
	return &Logger{dir: dir}
}

var defaultLogger = New("")

func init() {
	ai.RegisterStreamSimpleHook(defaultLogger.hook)
}

// hook is the StreamSimpleHook installed by this package. It attaches OnPayload
// and OnResponse observers (chaining any call-site callbacks) and records the
// final answer once the stream completes.
func (l *Logger) hook(model *ai.Model, ctxt *ai.Context, opts *ai.SimpleStreamOptions, next func() *ai.AssistantMessageEventStream) *ai.AssistantMessageEventStream {
	if opts == nil {
		opts = &ai.SimpleStreamOptions{}
	}

	var (
		respMu      sync.Mutex
		respStatus  int
		respHeaders map[string]string
	)

	// Mirror upstream onPayload: inspect the outbound provider payload.
	prevOnPayload := opts.OnPayload
	opts.OnPayload = func(payload any, m *ai.Model) (any, error) {
		l.write(LogEntry{
			Timestamp: time.Now().UnixMilli(),
			Direction: "request",
			Model:     m.Provider + "/" + m.ID,
			Provider:  m.Provider,
			Payload:   payload,
		})
		if prevOnPayload != nil {
			return prevOnPayload(payload, m)
		}
		return payload, nil
	}

	// Mirror upstream onResponse: observe HTTP status/headers after the
	// response arrives, before its body stream is consumed.
	prevOnResponse := opts.OnResponse
	opts.OnResponse = func(status int, headers map[string]string, m *ai.Model) {
		respMu.Lock()
		respStatus = status
		respHeaders = headers
		respMu.Unlock()
		if prevOnResponse != nil {
			prevOnResponse(status, headers, m)
		}
	}

	stream := next()

	// Record the final answer once the stream completes.
	go func() {
		result := stream.Result()
		respMu.Lock()
		status := respStatus
		headers := respHeaders
		respMu.Unlock()
		l.write(LogEntry{
			Timestamp:    time.Now().UnixMilli(),
			Direction:    "response",
			Model:        model.Provider + "/" + model.ID,
			Provider:     model.Provider,
			Status:       status,
			Headers:      headers,
			StopReason:   result.StopReason,
			ErrorMessage: result.ErrorMessage,
			Usage:        &result.Usage,
			Content:      contentText(result.Content),
		})
	}()

	return stream
}

func contentText(blocks []ai.ContentBlock) string {
	var out string
	for _, b := range blocks {
		switch b.Type {
		case ai.ContentTypeText:
			out += b.Text
		case ai.ContentTypeThinking:
			out += "[thinking] " + b.Thinking
		}
	}
	return out
}

func (l *Logger) write(entry LogEntry) {
	f, err := l.fileFor(time.Now())
	if err != nil {
		return
	}
	data, _ := json.Marshal(entry)
	l.mu.Lock()
	_, _ = f.Write(append(data, '\n'))
	l.mu.Unlock()
}

func (l *Logger) fileFor(now time.Time) (*os.File, error) {
	today := now.Format("2006-01-02")
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.fileDate != today {
		if l.file != nil {
			_ = l.file.Close()
			l.file = nil
		}
		l.fileDate = today
	}
	if l.file == nil {
		if l.dir == "" {
			return nil, os.ErrNotExist
		}
		if err := os.MkdirAll(l.dir, 0700); err != nil {
			return nil, err
		}
		f, err := os.OpenFile(filepath.Join(l.dir, "llm-"+today+".jsonl"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
		if err != nil {
			return nil, err
		}
		l.file = f
	}
	return l.file, nil
}