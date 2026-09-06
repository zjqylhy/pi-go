package ai

import (
	"encoding/json"
	"strings"
)

// anthropicAPI implements the Anthropic Messages wire protocol via SSE.
type anthropicAPI struct{}

func (anthropicAPI) Stream(model *Model, context *Context, opts *StreamOptions) *AssistantMessageEventStream {
	return anthropicStream(model, context, opts, nil)
}

func (anthropicAPI) StreamSimple(model *Model, context *Context, opts *SimpleStreamOptions) *AssistantMessageEventStream {
	return anthropicStream(model, context, &opts.StreamOptions, opts)
}

func anthropicStream(model *Model, context *Context, opts *StreamOptions, simple *SimpleStreamOptions) *AssistantMessageEventStream {
	baseURL := model.BaseURL
	if baseURL == "" {
		baseURL = "https://api.anthropic.com"
	}
	url := joinURL(baseURL, "/v1/messages")

	apiKey := ""
	if opts != nil {
		apiKey = opts.APIKey
	}

	// Auth headers: an oauth-token-looking key is sent as a Bearer token.
	headers := map[string]string{
		"accept":            "application/json",
		"content-type":      "application/json",
		"anthropic-version": "2023-06-01",
	}
	if isAnthropicOAuthToken(apiKey) {
		headers["authorization"] = "Bearer " + apiKey
	} else {
		headers["x-api-key"] = apiKey
	}

	if simple != nil && simple.SessionID != "" && hasCompatFlag(model, "sendSessionAffinityHeaders") {
		headers["x-session-affinity"] = simple.SessionID
	}

	messages := anthropicWireMessages(context.Messages)
	body := map[string]any{
		"model":      model.ID,
		"max_tokens": effectiveMaxTokens(model, opts),
		"stream":     true,
		"messages":   messages,
	}
	if context.SystemPrompt != "" {
		body["system"] = []map[string]any{{"type": "text", "text": context.SystemPrompt}}
	}
	if opts != nil && opts.Temperature != nil {
		body["temperature"] = *opts.Temperature
	}
	if len(context.Tools) > 0 {
		body["tools"] = anthropicTools(context.Tools)
	}
	if simple != nil {
		if simple.ToolChoice == ToolChoiceNone {
			body["tool_choice"] = map[string]any{"type": "none"}
		} else if simple.ToolChoice == ToolChoiceAuto {
			body["tool_choice"] = map[string]any{"type": "auto"}
		}
		if simple.Reasoning != "" && simple.Reasoning != ThinkingOff {
			body["thinking"] = map[string]any{"type": "enabled", "budget_tokens": anthropicThinkingBudget(simple.Reasoning)}
		}
	}

	conv := newAnthropicConverter(model)
	return pumpSSE(model, opts, "POST", url, headers, body, anthropicEventAdapter(conv))
}

// anthropicEventAdapter wraps the single-event converter for the multi-emit
// SSE pump.
func anthropicEventAdapter(conv *anthropicConverter) func(sseEvent) ([]Event, bool, error) {
	return func(ev sseEvent) ([]Event, bool, error) {
		e, term, err := conv.handle(ev)
		if err != nil || e.Type == "" {
			return nil, term, err
		}
		return []Event{e}, term, nil
	}
}

func isAnthropicOAuthToken(key string) bool {
	return strings.Contains(key, "sk-ant-oat")
}

func effectiveMaxTokens(model *Model, opts *StreamOptions) int {
	if opts != nil && opts.MaxTokens > 0 {
		return opts.MaxTokens
	}
	if model.MaxTokens > 0 {
		return model.MaxTokens
	}
	return 4096
}

func anthropicThinkingBudget(level ThinkingLevel) int {
	switch level {
	case ThinkingMinimal, ThinkingLow:
		return 16000
	case ThinkingMedium:
		return 32000
	default:
		return 64000
	}
}

func anthropicTools(tools []Tool) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		params := t.Parameters
		if params == nil {
			params = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		out = append(out, map[string]any{
			"name":         t.Name,
			"description":  t.Description,
			"input_schema": params,
		})
	}
	return out
}

func hasCompatFlag(model *Model, flag string) bool {
	if model.Compat == nil {
		return false
	}
	v, ok := model.Compat[flag].(bool)
	return ok && v
}

// anthropicWireMessages converts and filters the transcript for replay.
func anthropicWireMessages(messages []Message) []map[string]any {
	neutral := &Model{Provider: "anthropic", Input: []string{"text", "image"}}
	transformed := TransformMessages(messages, neutral)
	out := make([]map[string]any, 0, len(transformed))
	for _, m := range transformed {
		switch msg := m.(type) {
		case *UserMessage:
			out = append(out, anthropicUserMsg(*msg))
		case UserMessage:
			out = append(out, anthropicUserMsg(msg))
		case *AssistantMessage:
			out = append(out, anthropicAssistantMsg(*msg))
		case AssistantMessage:
			out = append(out, anthropicAssistantMsg(msg))
		case *ToolResultMessage:
			out = append(out, anthropicToolResultMsg(*msg))
		case ToolResultMessage:
			out = append(out, anthropicToolResultMsg(msg))
		}
	}
	return out
}

func anthropicUserMsg(m UserMessage) map[string]any {
	return map[string]any{"role": "user", "content": anthropicContent(m.Content)}
}

func anthropicContent(blocks []ContentBlock) any {
	if len(blocks) == 1 && blocks[0].Type == ContentTypeText {
		return blocks[0].Text
	}
	arr := make([]map[string]any, 0, len(blocks))
	for _, b := range blocks {
		switch b.Type {
		case ContentTypeText:
			arr = append(arr, map[string]any{"type": "text", "text": b.Text})
		case ContentTypeImage:
			arr = append(arr, map[string]any{"type": "image", "source": map[string]any{"type": "base64", "media_type": b.MimeType, "data": b.Data}})
		}
	}
	return arr
}

func anthropicToolResultMsg(m ToolResultMessage) map[string]any {
	blocks := []map[string]any{}
	for _, b := range m.Content {
		switch b.Type {
		case ContentTypeText:
			blocks = append(blocks, map[string]any{"type": "text", "text": b.Text})
		case ContentTypeImage:
			blocks = append(blocks, map[string]any{"type": "image", "source": map[string]any{"type": "base64", "media_type": b.MimeType, "data": b.Data}})
		}
	}
	return map[string]any{
		"role":    "user",
		"content": []map[string]any{{"type": "tool_result", "tool_use_id": m.ToolCallID, "content": blocks, "is_error": m.IsError}},
	}
}

func anthropicAssistantMsg(m AssistantMessage) map[string]any {
	blocks := []map[string]any{}
	for _, b := range m.Content {
		switch b.Type {
		case ContentTypeText:
			blocks = append(blocks, map[string]any{"type": "text", "text": b.Text})
		case ContentTypeThinking:
			block := map[string]any{"type": "thinking", "thinking": b.Thinking}
			if b.ThinkingSignature != "" {
				block["signature"] = b.ThinkingSignature
			}
			blocks = append(blocks, block)
		case ContentTypeToolCall:
			args := b.Arguments
			if args == nil {
				args = map[string]any{}
			}
			blocks = append(blocks, map[string]any{"type": "tool_use", "id": b.ID, "name": b.Name, "input": args})
		}
	}
	return map[string]any{"role": "assistant", "content": blocks}
}

// anthropicConverter accumulates SSE events into a partial AssistantMessage.
type anthropicConverter struct {
	model    *Model
	partial  *AssistantMessage
	started  bool
	terminal bool
	// block index -> content index mapping (1:1 in practice)
	toolJSON map[int]string
}

func newAnthropicConverter(model *Model) *anthropicConverter {
	return &anthropicConverter{
		model: model,
		partial: &AssistantMessage{
			Role:       "assistant",
			Content:    []ContentBlock{},
			API:        model.API,
			Provider:   model.Provider,
			Model:      model.ID,
			Usage:      Usage{},
			StopReason: StopPending,
			Timestamp:  nowMs(),
		},
		toolJSON: map[int]string{},
	}
}

func (c *anthropicConverter) handle(ev sseEvent) (Event, bool, error) {
	if ev.Event == "error" {
		return Event{}, true, errStreamError(ev.Data)
	}
	if ev.Data == "" || ev.Data == "[DONE]" {
		return Event{}, false, nil
	}
	var data map[string]any
	if err := json.Unmarshal([]byte(ev.Data), &data); err != nil {
		return Event{}, false, nil
	}
	typ, _ := data["type"].(string)

	switch typ {
	case "message_start":
		if c.started || c.terminal {
			return Event{}, false, nil
		}
		c.started = true
		if msg, ok := data["message"].(map[string]any); ok {
			c.applyMessageStart(msg)
		}
		return Event{Type: EventStart, Partial: c.partial}, false, nil

	case "content_block_start":
		idx, _ := data["index"].(float64)
		block, _ := data["content_block"].(map[string]any)
		bt, _ := block["type"].(string)
		switch bt {
		case "text":
			c.ensureBlock(int(idx))
			c.partial.Content[int(idx)] = TextBlock(stringVal(block, "text"))
			return c.ev(EventTextStart, int(idx)), false, nil
		case "thinking", "redacted_thinking":
			c.ensureBlock(int(idx))
			thinking, _ := block["thinking"].(string)
			c.partial.Content[int(idx)] = ThinkingBlock(thinking)
			if bt == "redacted_thinking" {
				c.partial.Content[int(idx)].Redacted = true
			}
			return c.ev(EventThinkingStart, int(idx)), false, nil
		case "tool_use":
			c.ensureBlock(int(idx))
			id, _ := block["id"].(string)
			name, _ := block["name"].(string)
			c.partial.Content[int(idx)] = ToolCallBlock(id, name, map[string]any{})
			c.toolJSON[int(idx)] = ""
			return c.ev(EventToolCallStart, int(idx)), false, nil
		}
		return Event{}, false, nil

	case "content_block_delta":
		idx, _ := data["index"].(float64)
		delta, _ := data["delta"].(map[string]any)
		dt, _ := delta["type"].(string)
		switch dt {
		case "text_delta":
			i := int(idx)
			c.ensureBlock(i)
			text, _ := delta["text"].(string)
			block := &c.partial.Content[i]
			block.Text += text
			return c.deltaEvent(EventTextDelta, i, text), false, nil
		case "thinking_delta":
			i := int(idx)
			c.ensureBlock(i)
			text, _ := delta["thinking"].(string)
			block := &c.partial.Content[i]
			block.Thinking += text
			return c.deltaEvent(EventThinkingDelta, i, text), false, nil
		case "input_json_delta":
			i := int(idx)
			c.ensureBlock(i)
			partial, _ := delta["partial_json"].(string)
			c.toolJSON[i] += partial
			c.partial.Content[i].Arguments = parseStreamingJson(c.toolJSON[i])
			return c.deltaEvent(EventToolCallDelta, i, partial), false, nil
		case "signature_delta":
			i := int(idx)
			c.ensureBlock(i)
			sig, _ := delta["signature"].(string)
			c.partial.Content[i].ThinkingSignature += sig
			return Event{}, false, nil
		}
		return Event{}, false, nil

	case "content_block_stop":
		idx, _ := data["index"].(float64)
		i := int(idx)
		c.ensureBlock(i)
		block := c.partial.Content[i]
		switch block.Type {
		case ContentTypeText:
			c.partial.Content[i].Text = block.Text
			ev := c.ev(EventTextEnd, i)
			ev.Content = block.Text
			return ev, false, nil
		case ContentTypeThinking:
			ev := c.ev(EventThinkingEnd, i)
			ev.Content = block.Thinking
			return ev, false, nil
		case ContentTypeToolCall:
			ev := c.ev(EventToolCallEnd, i)
			ev.ToolCall = &c.partial.Content[i]
			return ev, false, nil
		}
		return Event{}, false, nil

	case "message_delta":
		delta, _ := data["delta"].(map[string]any)
		if sr, ok := delta["stop_reason"].(string); ok {
			c.partial.RawStopReason = sr
			c.partial.StopReason = mapAnthropicStopReason(sr)
		}
		if usage, ok := data["usage"].(map[string]any); ok {
			c.applyUsage(usage)
		}
		return Event{}, false, nil

	case "message_stop":
		c.partial.StopReason = mapAnthropicStopReason(c.partial.RawStopReason)
		if c.partial.StopReason == StopPending {
			c.partial.StopReason = StopEnd
		}
		c.computeCost()
		done := c.partial.StopReason
		return Event{Type: EventDone, Reason: done, Message: c.partial}, true, nil

	default:
		return Event{}, false, nil
	}
}

func (c *anthropicConverter) ensureBlock(i int) {
	for len(c.partial.Content) <= i {
		c.partial.Content = append(c.partial.Content, ContentBlock{})
	}
}

func (c *anthropicConverter) ev(typ string, i int) Event {
	return Event{Type: typ, ContentIndex: i, Partial: c.partial}
}

func (c *anthropicConverter) deltaEvent(typ string, i int, delta string) Event {
	return Event{Type: typ, ContentIndex: i, Delta: delta, Partial: c.partial}
}

func (c *anthropicConverter) applyMessageStart(msg map[string]any) {
	if id, ok := msg["id"].(string); ok {
		c.partial.ResponseID = id
	}
	if m, ok := msg["model"].(string); ok {
		c.partial.ResponseModel = m
	}
	if usage, ok := msg["usage"].(map[string]any); ok {
		c.applyUsage(usage)
	}
}

func (c *anthropicConverter) applyUsage(usage map[string]any) {
	c.partial.Usage.Input = intOr(usage, "input_tokens")
	c.partial.Usage.Output = intOr(usage, "output_tokens")
	c.partial.Usage.CacheRead = intOr(usage, "cache_read_input_tokens")
	c.partial.Usage.CacheWrite = intOr(usage, "cache_creation_input_tokens")
	if cc, ok := usage["cache_creation"].(map[string]any); ok {
		c.partial.Usage.CacheWrite1h = intOr(cc, "ephemeral_1h_input_tokens")
	}
	c.partial.Usage.TotalTokens = c.partial.Usage.Input + c.partial.Usage.Output
}

func (c *anthropicConverter) computeCost() {
	c.partial.Usage.Cost = calculateCost(c.model, &c.partial.Usage)
}

func intOr(m map[string]any, key string) int {
	if v, ok := m[key].(float64); ok {
		return int(v)
	}
	if v, ok := m[key].(int); ok {
		return v
	}
	return 0
}

func mapAnthropicStopReason(sr string) StopReason {
	switch sr {
	case "end_turn":
		return StopEnd
	case "max_tokens":
		return StopLength
	case "tool_use":
		return StopToolUse
	case "refusal", "sensitive":
		return StopError
	case "pause_turn", "stop_sequence":
		return StopEnd
	default:
		return StopPending
	}
}

func stringVal(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}
