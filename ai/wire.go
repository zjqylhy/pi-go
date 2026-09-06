package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"
)

var errStreamComplete = errors.New("stream complete")

func errStreamError(data string) error {
	return errors.New("provider stream error" + detailSuffix(data))
}

func detailSuffix(data string) string {
	data = strings.TrimSpace(data)
	if data == "" {
		return ""
	}
	return ": " + data
}

func trimSlash(s string) string { return strings.TrimRight(s, "/") }

func joinURL(base, path string) string { return trimSlash(base) + path }

func ctxFrom(opts *StreamOptions) context.Context {
	if opts != nil && opts.Ctx != nil {
		return opts.Ctx
	}
	return context.Background()
}

// doHTTP serializes body as JSON and performs the request.
func doHTTP(ctx context.Context, opts *StreamOptions, method, url string, headers map[string]string, body any, model *Model) (*http.Response, error) {
	payload := body
	if opts != nil && opts.OnPayload != nil {
		p, err := opts.OnPayload(body, model)
		if err != nil {
			return nil, err
		}
		payload = p
	}
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(payload); err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, method, url, &buf)
	if err != nil {
		return nil, err
	}
	for k, v := range headers {
		if v == "" {
			continue
		}
		req.Header.Set(k, v)
	}
	client := &http.Client{}
	if opts != nil && opts.TimeoutMs > 0 {
		client.Timeout = time.Duration(opts.TimeoutMs) * time.Millisecond
	}
	return client.Do(req)
}

func headerMap(h http.Header) map[string]string {
	out := map[string]string{}
	for k, v := range h {
		out[k] = strings.Join(v, ", ")
	}
	return out
}

// readHTTPError formats an HTTP error response body.
func readHTTPError(resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	text := strings.TrimSpace(string(body))
	if text == "" {
		return errors.New(resp.Status)
	}
	return errors.New(resp.Status + ": " + text)
}

func isCtxAborted(opts *StreamOptions) bool {
	return opts != nil && opts.Ctx != nil && opts.Ctx.Err() != nil
}

// buildErrorAssistantMessage constructs a terminal error/aborted message.
func buildErrorAssistantMessage(model *Model, err error, aborted bool) *AssistantMessage {
	reason := StopError
	msg := ""
	if err != nil {
		msg = err.Error()
	}
	if aborted {
		reason = StopAborted
	}
	return &AssistantMessage{
		Role:         "assistant",
		Content:      []ContentBlock{},
		API:          model.API,
		Provider:     model.Provider,
		Model:        model.ID,
		Usage:        Usage{},
		StopReason:   reason,
		ErrorMessage: msg,
		Timestamp:    nowMs(),
	}
}

// pumpSSE performs an HTTP SSE request, converts events, and drives the stream.
// onEvent returns the public events, whether it is terminal, and an error.
func pumpSSE(model *Model, opts *StreamOptions, method, url string, headers map[string]string, body any, onEvent func(sseEvent) ([]Event, bool, error)) *AssistantMessageEventStream {
	stream := NewAssistantMessageEventStream()
	go func() {
		var terminal bool
		ctx := ctxFrom(opts)
		resp, err := doHTTP(ctx, opts, method, url, headers, body, model)
		if err == nil {
			defer resp.Body.Close()
			if opts != nil && opts.OnResponse != nil {
				opts.OnResponse(resp.StatusCode, headerMap(resp.Header), model)
			}
			if resp.StatusCode < 200 || resp.StatusCode >= 300 {
				err = readHTTPError(resp)
			} else {
				err = scanSSE(resp.Body, func(event, data string) error {
					events, term, e := onEvent(sseEvent{Event: event, Data: data})
					if e != nil {
						return e
					}
					for _, ev := range events {
						stream.Push(ev)
					}
					if term {
						terminal = true
						return errStreamComplete
					}
					return nil
				})
				if err == errStreamComplete {
					err = nil
				} else if err == nil && !terminal {
					err = errors.New("stream ended without a terminal event")
				}
			}
		}
		if err != nil {
			msg := buildErrorAssistantMessage(model, err, isCtxAborted(opts))
			stream.Push(Event{Type: EventError, Reason: msg.StopReason, Error: msg})
			stream.End(msg)
		}
	}()
	return stream
}

// baseHeaders builds the common request headers (auth + accept). The special
// "unused" apiKey convention is applied when an explicit Authorization or
// cf-aig-authorization header is already present.
func baseHeaders(opts *StreamOptions, apiKey string, extra map[string]string) map[string]string {
	h := map[string]string{
		"accept":       "application/json, text/event-stream",
		"content-type": "application/json",
		"user-agent":   "pi-go/0.1",
	}
	if apiKey != "" && apiKey != "unused" {
		h["authorization"] = "Bearer " + apiKey
	} else if apiKey == "unused" {
		h["authorization"] = "Bearer unused"
	}
	applyRequestHeaders(h, opts, extra)
	return h
}

// applyRequestHeaders merges explicit headers, where a non-nil value overrides
// case-insensitively and a nil value suppresses the header.
func applyRequestHeaders(h map[string]string, opts *StreamOptions, extra map[string]string) {
	for k, v := range extra {
		h[strings.ToLower(k)] = v
	}
	if opts == nil {
		return
	}
	for k, v := range opts.Headers {
		lk := strings.ToLower(k)
		if v == nil {
			delete(h, lk)
			continue
		}
		h[lk] = *v
	}
}
