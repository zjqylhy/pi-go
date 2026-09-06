package ai

import "encoding/json"

// openAICompletionsAPI implements the OpenAI Chat Completions wire protocol,
// reused by OpenAI-compatible providers (deepseek, groq, xai, etc.).
type openAICompletionsAPI struct{}

func (openAICompletionsAPI) Stream(model *Model, context *Context, opts *StreamOptions) *AssistantMessageEventStream {
	return openAICompletionsStream(model, context, opts, nil)
}

func (openAICompletionsAPI) StreamSimple(model *Model, context *Context, opts *SimpleStreamOptions) *AssistantMessageEventStream {
	return openAICompletionsStream(model, context, &opts.StreamOptions, opts)
}

func openAICompletionsStream(model *Model, context *Context, opts *StreamOptions, simple *SimpleStreamOptions) *AssistantMessageEventStream {
	baseURL := model.BaseURL
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}
	url := joinURL(baseURL, "/chat/completions")

	apiKey := ""
	if opts != nil {
		apiKey = opts.APIKey
	}
	headers := baseHeaders(opts, apiKey, nil)

	body := map[string]any{
		"model":          model.ID,
		"messages":       openAIChatMessages(context.Messages, model),
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
	}
	if opts != nil && opts.Temperature != nil {
		body["temperature"] = *opts.Temperature
	}
	if mt := effectiveMaxTokens(model, opts); mt > 0 {
		body["max_tokens"] = mt
	}
	if len(context.Tools) > 0 {
		body["tools"] = openAIChatTools(context.Tools)
	}
	if simple != nil && simple.ToolChoice != "" {
		body["tool_choice"] = simple.ToolChoice
	}

	conv := newOpenAICompletionsConverter(model)
	return pumpSSE(model, opts, "POST", url, headers, body, conv.handle)
}

func openAIChatMessages(messages []Message, model *Model) []map[string]any {
	transformed := TransformMessages(messages, model)
	out := make([]map[string]any, 0, len(transformed))
	for _, m := range transformed {
		switch msg := m.(type) {
		case *UserMessage:
			out = append(out, openAIChatUser(*msg))
		case UserMessage:
			out = append(out, openAIChatUser(msg))
		case *AssistantMessage:
			out = append(out, openAIChatAssistant(*msg))
		case AssistantMessage:
			out = append(out, openAIChatAssistant(msg))
		case *ToolResultMessage:
			out = append(out, openAIChatToolResult(*msg))
		case ToolResultMessage:
			out = append(out, openAIChatToolResult(msg))
		}
	}
	return out
}

func openAIChatUser(m UserMessage) map[string]any {
	return map[string]any{"role": "user", "content": contentText(m.Content)}
}

func openAIChatAssistant(m AssistantMessage) map[string]any {
	msg := map[string]any{"role": "assistant"}
	var text string
	var toolCalls []map[string]any
	for _, b := range m.Content {
		switch b.Type {
		case ContentTypeText:
			text += b.Text
		case ContentTypeToolCall:
			args, _ := json.Marshal(b.Arguments)
			toolCalls = append(toolCalls, map[string]any{
				"id":   b.ID,
				"type": "function",
				"function": map[string]any{
					"name":      b.Name,
					"arguments": string(args),
				},
			})
		}
	}
	if text != "" {
		msg["content"] = text
	}
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
	}
	return msg
}

func openAIChatToolResult(m ToolResultMessage) map[string]any {
	return map[string]any{
		"role":         "tool",
		"tool_call_id": m.ToolCallID,
		"content":      contentText(m.Content),
	}
}

func openAIChatTools(tools []Tool) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		params := t.Parameters
		if params == nil {
			params = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		out = append(out, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        t.Name,
				"description": t.Description,
				"parameters":  params,
			},
		})
	}
	return out
}

// openAICompletionsConverter accumulates chat-completion SSE chunks.
type openAICompletionsConverter struct {
	model       *Model
	partial     *AssistantMessage
	textIdx     int
	thinkingIdx int
	toolIdx     map[int]int
	toolJSON    map[int]string
	finalized   bool
}

func newOpenAICompletionsConverter(model *Model) *openAICompletionsConverter {
	return &openAICompletionsConverter{
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
		textIdx:     -1,
		thinkingIdx: -1,
		toolIdx:     map[int]int{},
		toolJSON:    map[int]string{},
	}
}

func (c *openAICompletionsConverter) handle(ev sseEvent) ([]Event, bool, error) {
	if ev.Event == "error" {
		return nil, true, errStreamError(ev.Data)
	}
	data, ok := parseSSEData(ev.Data)
	if !ok {
		return nil, false, nil
	}
	var chunk map[string]any
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return nil, false, nil
	}

	if id, ok := chunk["id"].(string); ok && c.partial.ResponseID == "" {
		c.partial.ResponseID = id
	}
	if m, ok := chunk["model"].(string); ok {
		c.partial.ResponseModel = m
	}
	if usage, ok := chunk["usage"].(map[string]any); ok {
		c.applyUsage(usage)
	}

	var out []Event
	terminal := false
	choices, _ := chunk["choices"].([]any)
	for _, raw := range choices {
		choice, _ := raw.(map[string]any)
		if usage, ok := choice["usage"].(map[string]any); ok {
			c.applyUsage(usage)
		}
		delta, _ := choice["delta"].(map[string]any)
		out = append(out, c.applyDelta(delta)...)
		if fr, ok := choice["finish_reason"].(string); ok && fr != "" {
			c.partial.RawStopReason = fr
			c.partial.StopReason = mapOpenAICompletionsStopReason(fr)
			terminal = true
		}
	}

	if terminal {
		out = append(out, c.endEvents()...)
		c.computeCost()
		out = append(out, Event{Type: EventDone, Reason: c.partial.StopReason, Message: c.partial})
		return out, true, nil
	}
	return out, false, nil
}

func (c *openAICompletionsConverter) applyDelta(delta map[string]any) []Event {
	if delta == nil {
		return nil
	}
	var out []Event
	for _, key := range []string{"reasoning_content", "reasoning"} {
		if v, ok := delta[key].(string); ok && v != "" {
			out = append(out, c.appendThinking(v)...)
			break
		}
	}
	if content, ok := delta["content"].(string); ok && content != "" {
		out = append(out, c.appendText(content)...)
	}
	if tcs, ok := delta["tool_calls"].([]any); ok {
		for _, raw := range tcs {
			tc, _ := raw.(map[string]any)
			out = append(out, c.applyToolCall(tc)...)
		}
	}
	return out
}

func (c *openAICompletionsConverter) appendText(text string) []Event {
	if c.textIdx < 0 {
		c.textIdx = len(c.partial.Content)
		c.partial.Content = append(c.partial.Content, TextBlock(""))
		start := Event{Type: EventTextStart, ContentIndex: c.textIdx, Partial: c.partial}
		c.partial.Content[c.textIdx].Text += text
		return []Event{start, Event{Type: EventTextDelta, ContentIndex: c.textIdx, Delta: text, Partial: c.partial}}
	}
	c.partial.Content[c.textIdx].Text += text
	return []Event{{Type: EventTextDelta, ContentIndex: c.textIdx, Delta: text, Partial: c.partial}}
}

func (c *openAICompletionsConverter) appendThinking(text string) []Event {
	if c.thinkingIdx < 0 {
		c.thinkingIdx = len(c.partial.Content)
		c.partial.Content = append(c.partial.Content, ThinkingBlock(""))
		start := Event{Type: EventThinkingStart, ContentIndex: c.thinkingIdx, Partial: c.partial}
		c.partial.Content[c.thinkingIdx].Thinking += text
		return []Event{start, Event{Type: EventThinkingDelta, ContentIndex: c.thinkingIdx, Delta: text, Partial: c.partial}}
	}
	c.partial.Content[c.thinkingIdx].Thinking += text
	return []Event{{Type: EventThinkingDelta, ContentIndex: c.thinkingIdx, Delta: text, Partial: c.partial}}
}

func (c *openAICompletionsConverter) applyToolCall(tc map[string]any) []Event {
	rawIdx, _ := tc["index"].(float64)
	idx := int(rawIdx)
	ci, ok := c.toolIdx[idx]
	var out []Event
	if !ok {
		ci = len(c.partial.Content)
		c.toolIdx[idx] = ci
		c.toolJSON[idx] = ""
		c.partial.Content = append(c.partial.Content, ToolCallBlock("", "", map[string]any{}))
		out = append(out, Event{Type: EventToolCallStart, ContentIndex: ci, Partial: c.partial})
	}
	if id := stringVal(tc, "id"); id != "" {
		c.partial.Content[ci].ID = id
	}
	fn, _ := tc["function"].(map[string]any)
	if fn != nil {
		if name := stringVal(fn, "name"); name != "" {
			c.partial.Content[ci].Name = name
		}
		if args := stringVal(fn, "arguments"); args != "" {
			c.toolJSON[idx] += args
			c.partial.Content[ci].Arguments = parseStreamingJson(c.toolJSON[idx])
			out = append(out, Event{Type: EventToolCallDelta, ContentIndex: ci, Delta: args, Partial: c.partial})
		}
	}
	return out
}

func (c *openAICompletionsConverter) endEvents() []Event {
	if c.finalized {
		return nil
	}
	c.finalized = true
	var out []Event
	for i := range c.partial.Content {
		b := c.partial.Content[i]
		switch b.Type {
		case ContentTypeText:
			out = append(out, Event{Type: EventTextEnd, ContentIndex: i, Content: b.Text, Partial: c.partial})
		case ContentTypeThinking:
			out = append(out, Event{Type: EventThinkingEnd, ContentIndex: i, Content: b.Thinking, Partial: c.partial})
		case ContentTypeToolCall:
			out = append(out, Event{Type: EventToolCallEnd, ContentIndex: i, ToolCall: &c.partial.Content[i], Partial: c.partial})
		}
	}
	return out
}

func (c *openAICompletionsConverter) applyUsage(usage map[string]any) {
	in := intOr(usage, "prompt_tokens")
	out := intOr(usage, "completion_tokens")
	total := intOr(usage, "total_tokens")
	cached := 0
	if details, ok := usage["prompt_tokens_details"].(map[string]any); ok {
		cached = intOr(details, "cached_tokens")
		if cached == 0 {
			cached = intOr(usage, "prompt_cache_hit_tokens")
		}
		c.partial.Usage.CacheWrite = intOr(details, "cache_write_tokens")
	}
	c.partial.Usage.Input = in
	c.partial.Usage.CacheRead = cached
	c.partial.Usage.Output = out
	c.partial.Usage.TotalTokens = total
	if total == 0 {
		c.partial.Usage.TotalTokens = in + out
	}
}

func (c *openAICompletionsConverter) computeCost() {
	c.partial.Usage.Cost = calculateCost(c.model, &c.partial.Usage)
}

func mapOpenAICompletionsStopReason(sr string) StopReason {
	switch sr {
	case "stop", "end":
		return StopEnd
	case "length":
		return StopLength
	case "function_call", "tool_calls":
		return StopToolUse
	case "content_filter", "network_error":
		return StopError
	default:
		return StopEnd
	}
}
