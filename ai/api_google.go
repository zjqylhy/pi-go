package ai

import "encoding/json"

// googleGenerativeAIAPI implements the Google Generative Language
// generateContent streaming protocol.
type googleGenerativeAIAPI struct{}

func (googleGenerativeAIAPI) Stream(model *Model, context *Context, opts *StreamOptions) *AssistantMessageEventStream {
	return googleStream(model, context, opts, nil)
}

func (googleGenerativeAIAPI) StreamSimple(model *Model, context *Context, opts *SimpleStreamOptions) *AssistantMessageEventStream {
	return googleStream(model, context, &opts.StreamOptions, opts)
}

func googleStream(model *Model, context *Context, opts *StreamOptions, simple *SimpleStreamOptions) *AssistantMessageEventStream {
	baseURL := model.BaseURL
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com/v1beta"
	}
	url := joinURL(baseURL, "/models/"+model.ID+":streamGenerateContent?alt=sse")

	if opts != nil && opts.APIKey != "" {
		url += "&key=" + opts.APIKey
	}

	headers := map[string]string{
		"content-type": "application/json",
	}

	body := map[string]any{
		"contents": googleMessages(context.Messages, model),
	}
	if context.SystemPrompt != "" {
		body["systemInstruction"] = map[string]any{
			"parts": []map[string]any{{"text": context.SystemPrompt}},
		}
	}
	genConfig := map[string]any{}
	if opts != nil && opts.Temperature != nil {
		genConfig["temperature"] = *opts.Temperature
	}
	if mt := effectiveMaxTokens(model, opts); mt > 0 {
		genConfig["maxOutputTokens"] = mt
	}
	if len(genConfig) > 0 {
		body["generationConfig"] = genConfig
	}
	if len(context.Tools) > 0 {
		body["tools"] = googleTools(context.Tools)
	}

	conv := newGoogleConverter(model)
	return pumpSSE(model, opts, "POST", url, headers, body, conv.handle)
}

func googleMessages(messages []Message, model *Model) []map[string]any {
	transformed := TransformMessages(messages, model)
	out := make([]map[string]any, 0, len(transformed))
	for _, m := range transformed {
		switch msg := m.(type) {
		case *UserMessage:
			out = append(out, googleUserMsg(*msg))
		case UserMessage:
			out = append(out, googleUserMsg(msg))
		case *AssistantMessage:
			out = append(out, googleAssistantMsg(*msg))
		case AssistantMessage:
			out = append(out, googleAssistantMsg(msg))
		case *ToolResultMessage:
			out = append(out, googleToolResultMsg(*msg))
		case ToolResultMessage:
			out = append(out, googleToolResultMsg(msg))
		}
	}
	return out
}

func googleUserMsg(m UserMessage) map[string]any {
	return map[string]any{"role": "user", "parts": []map[string]any{{"text": contentText(m.Content)}}}
}

func googleAssistantMsg(m AssistantMessage) map[string]any {
	parts := []map[string]any{}
	var text string
	for _, b := range m.Content {
		switch b.Type {
		case ContentTypeText:
			text += b.Text
		case ContentTypeThinking:
			text += b.Thinking
		case ContentTypeToolCall:
			args := b.Arguments
			if args == nil {
				args = map[string]any{}
			}
			parts = append(parts, map[string]any{"functionCall": map[string]any{"name": b.Name, "args": args}})
		}
	}
	if text != "" {
		parts = append([]map[string]any{{"text": text}}, parts...)
	}
	return map[string]any{"role": "model", "parts": parts}
}

func googleToolResultMsg(m ToolResultMessage) map[string]any {
	parts := []map[string]any{}
	for _, b := range m.Content {
		if b.Type == ContentTypeText {
			parts = append(parts, map[string]any{
				"functionResponse": map[string]any{
					"name":     m.ToolName,
					"response": map[string]any{"result": b.Text},
				},
			})
		}
	}
	return map[string]any{"role": "user", "parts": parts}
}

func googleTools(tools []Tool) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		params := t.Parameters
		if params == nil {
			params = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		out = append(out, map[string]any{
			"functionDeclarations": []map[string]any{{
				"name":        t.Name,
				"description": t.Description,
				"parameters":  params,
			}},
		})
	}
	return out
}

// googleConverter accumulates generateContent SSE events.
type googleConverter struct {
	model    *Model
	partial  *AssistantMessage
	textIdx  int
	thinkIdx int
	toolIdx  int
	toolJSON string
	finished bool
}

func newGoogleConverter(model *Model) *googleConverter {
	return &googleConverter{
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
		textIdx:  -1,
		thinkIdx: -1,
		toolIdx:  -1,
	}
}

func (c *googleConverter) handle(ev sseEvent) ([]Event, bool, error) {
	if ev.Event == "error" {
		return nil, true, errStreamError(ev.Data)
	}
	data, ok := parseSSEData(ev.Data)
	if !ok {
		return nil, false, nil
	}
	var resp map[string]any
	if err := json.Unmarshal([]byte(data), &resp); err != nil {
		return nil, false, nil
	}

	if usage, ok := resp["usageMetadata"].(map[string]any); ok {
		c.partial.Usage.Input = intOr(usage, "promptTokenCount")
		c.partial.Usage.Output = intOr(usage, "candidatesTokenCount")
		c.partial.Usage.TotalTokens = intOr(usage, "totalTokenCount")
	}

	var out []Event
	terminal := false
	candidates, _ := resp["candidates"].([]any)
	for _, raw := range candidates {
		cand, _ := raw.(map[string]any)
		content, _ := cand["content"].(map[string]any)
		parts, _ := content["parts"].([]any)
		for _, rawPart := range parts {
			part, _ := rawPart.(map[string]any)
			out = append(out, c.applyPart(part)...)
		}
		if fr, ok := cand["finishReason"].(string); ok && fr != "" {
			c.partial.RawStopReason = fr
			c.partial.StopReason = mapGoogleStopReason(fr)
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

func (c *googleConverter) applyPart(part map[string]any) []Event {
	if fc, ok := part["functionCall"].(map[string]any); ok {
		name, _ := fc["name"].(string)
		args, _ := fc["args"].(map[string]any)
		if args == nil {
			args = map[string]any{}
		}
		if c.toolIdx < 0 {
			c.toolIdx = len(c.partial.Content)
			c.partial.Content = append(c.partial.Content, ToolCallBlock("", name, args))
			c.partial.Content[c.toolIdx].ID = googleToolID(c.toolIdx)
			return []Event{{Type: EventToolCallStart, ContentIndex: c.toolIdx, Partial: c.partial},
				{Type: EventToolCallEnd, ContentIndex: c.toolIdx, ToolCall: &c.partial.Content[c.toolIdx], Partial: c.partial}}
		}
		return nil
	}

	thought, _ := part["thought"].(bool)
	if thought {
		text, _ := part["text"].(string)
		return c.appendThinking(text)
	}
	text, _ := part["text"].(string)
	if text != "" {
		return c.appendText(text)
	}
	return nil
}

func googleToolID(i int) string {
	return "tool_" + intToStr(i)
}

func intToStr(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}

func (c *googleConverter) appendText(text string) []Event {
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

func (c *googleConverter) appendThinking(text string) []Event {
	if c.thinkIdx < 0 {
		c.thinkIdx = len(c.partial.Content)
		c.partial.Content = append(c.partial.Content, ThinkingBlock(""))
		start := Event{Type: EventThinkingStart, ContentIndex: c.thinkIdx, Partial: c.partial}
		c.partial.Content[c.thinkIdx].Thinking += text
		return []Event{start, Event{Type: EventThinkingDelta, ContentIndex: c.thinkIdx, Delta: text, Partial: c.partial}}
	}
	c.partial.Content[c.thinkIdx].Thinking += text
	return []Event{{Type: EventThinkingDelta, ContentIndex: c.thinkIdx, Delta: text, Partial: c.partial}}
}

func (c *googleConverter) endEvents() []Event {
	if c.finished {
		return nil
	}
	c.finished = true
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

func (c *googleConverter) computeCost() {
	c.partial.Usage.Cost = calculateCost(c.model, &c.partial.Usage)
}

func mapGoogleStopReason(sr string) StopReason {
	switch sr {
	case "STOP":
		return StopEnd
	case "MAX_TOKENS":
		return StopLength
	case "TOOL_CALLS":
		return StopToolUse
	case "SAFETY", "RECITATION", "IMAGE_SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "MALFORMED_FUNCTION_CALL":
		return StopError
	default:
		return StopEnd
	}
}
