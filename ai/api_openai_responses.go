package ai

import "encoding/json"

// openAIResponsesAPI implements the OpenAI Responses wire protocol, used by
// the first-party `openai` provider and OpenAI-compatible gateways.
type openAIResponsesAPI struct{}

func (openAIResponsesAPI) Stream(model *Model, context *Context, opts *StreamOptions) *AssistantMessageEventStream {
	return openAIResponsesStream(model, context, opts, nil)
}

func (openAIResponsesAPI) StreamSimple(model *Model, context *Context, opts *SimpleStreamOptions) *AssistantMessageEventStream {
	return openAIResponsesStream(model, context, &opts.StreamOptions, opts)
}

func openAIResponsesStream(model *Model, context *Context, opts *StreamOptions, simple *SimpleStreamOptions) *AssistantMessageEventStream {
	baseURL := model.BaseURL
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}
	url := joinURL(baseURL, "/responses")

	apiKey := ""
	if opts != nil {
		apiKey = opts.APIKey
	}
	headers := baseHeaders(opts, apiKey, nil)

	body := map[string]any{
		"model":  model.ID,
		"input":  openAIResponsesInput(context.Messages, model),
		"stream": true,
		"store":  false,
	}
	if context.SystemPrompt != "" {
		body["instructions"] = context.SystemPrompt
	}
	if opts != nil && opts.Temperature != nil {
		body["temperature"] = *opts.Temperature
	}
	if mt := effectiveMaxTokens(model, opts); mt > 0 {
		body["max_output_tokens"] = mt
	}
	if len(context.Tools) > 0 {
		body["tools"] = openAIResponsesTools(context.Tools)
	}
	if simple != nil && simple.ToolChoice != "" {
		body["tool_choice"] = simple.ToolChoice
	}

	conv := newOpenAIResponsesConverter(model)
	return pumpSSE(model, opts, "POST", url, headers, body, conv.handle)
}

func openAIResponsesInput(messages []Message, model *Model) []map[string]any {
	transformed := TransformMessages(messages, model)
	out := make([]map[string]any, 0, len(transformed))
	for _, m := range transformed {
		switch msg := m.(type) {
		case *UserMessage:
			out = append(out, responsesUserMsg(*msg, bodyRoleUser))
		case UserMessage:
			out = append(out, responsesUserMsg(msg, bodyRoleUser))
		case *AssistantMessage:
			out = append(out, responsesAssistantMsg(*msg))
		case AssistantMessage:
			out = append(out, responsesAssistantMsg(msg))
		case *ToolResultMessage:
			out = append(out, responsesToolResultMsg(*msg))
		case ToolResultMessage:
			out = append(out, responsesToolResultMsg(msg))
		}
	}
	return out
}

const (
	bodyRoleUser   = "user"
	bodyRoleSystem = "system"
)

func responsesUserMsg(m UserMessage, role string) map[string]any {
	content := []map[string]any{}
	for _, b := range m.Content {
		switch b.Type {
		case ContentTypeText:
			content = append(content, map[string]any{"type": "input_text", "text": b.Text})
		case ContentTypeImage:
			content = append(content, map[string]any{"type": "input_image", "image_url": "data:" + b.MimeType + ";base64," + b.Data})
		}
	}
	if len(content) == 0 {
		content = append(content, map[string]any{"type": "input_text", "text": contentText(m.Content)})
	}
	return map[string]any{"role": role, "content": content}
}

func responsesAssistantMsg(m AssistantMessage) map[string]any {
	content := []map[string]any{}
	var text string
	for _, b := range m.Content {
		switch b.Type {
		case ContentTypeText:
			text += b.Text
		case ContentTypeThinking:
			// reasoning is not replayed in responses assistant turns
		}
	}
	if text != "" {
		content = append(content, map[string]any{"type": "output_text", "text": text})
	}
	msg := map[string]any{
		"type":    "message",
		"role":    "assistant",
		"content": content,
		"status":  "completed",
	}
	if m.ResponseID != "" {
		msg["id"] = m.ResponseID
	}
	return msg
}

func responsesToolResultMsg(m ToolResultMessage) map[string]any {
	output := contentText(m.Content)
	return map[string]any{
		"type":    "function_call_output",
		"call_id": m.ToolCallID,
		"output":  output,
	}
}

func openAIResponsesTools(tools []Tool) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		params := t.Parameters
		if params == nil {
			params = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		out = append(out, map[string]any{
			"type":        "function",
			"name":        t.Name,
			"description": t.Description,
			"parameters":  params,
		})
	}
	return out
}

type responsesSlot struct {
	kind     string // "text" | "thinking" | "toolcall"
	index    int
	name     string
	id       string
	argsJSON string
}

type openAIResponsesConverter struct {
	model     *Model
	partial   *AssistantMessage
	slots     map[int]*responsesSlot
	finalized bool
}

func newOpenAIResponsesConverter(model *Model) *openAIResponsesConverter {
	return &openAIResponsesConverter{
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
		slots: map[int]*responsesSlot{},
	}
}

func (c *openAIResponsesConverter) handle(ev sseEvent) ([]Event, bool, error) {
	if ev.Event == "error" {
		return nil, true, errStreamError(ev.Data)
	}
	data, ok := parseSSEData(ev.Data)
	if !ok {
		return nil, false, nil
	}
	var obj map[string]any
	if err := json.Unmarshal([]byte(data), &obj); err != nil {
		return nil, false, nil
	}
	typ, _ := obj["type"].(string)

	switch typ {
	case "response.created":
		if resp, ok := obj["response"].(map[string]any); ok {
			c.partial.ResponseID = stringVal(resp, "id")
		}
		return nil, false, nil

	case "response.output_item.added":
		idx := intOf(obj, "output_index")
		item, _ := obj["item"].(map[string]any)
		itype, _ := item["type"].(string)
		return c.addSlot(idx, itype, item), false, nil

	case "response.output_text.delta", "response.refusal.delta":
		idx := intOf(obj, "output_index")
		delta := stringVal(obj, "delta")
		return c.appendTextSlot(idx, delta), false, nil

	case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
		idx := intOf(obj, "output_index")
		delta := stringVal(obj, "delta")
		return c.appendThinkingSlot(idx, delta), false, nil

	case "response.function_call_arguments.delta":
		idx := intOf(obj, "output_index")
		delta := stringVal(obj, "delta")
		return c.appendToolArgs(idx, delta), false, nil

	case "response.output_item.done":
		idx := intOf(obj, "output_index")
		return c.finalizeSlot(idx), false, nil

	case "response.completed":
		if resp, ok := obj["response"].(map[string]any); ok {
			c.applyResponsesUsage(resp)
		}
		return c.finish(StopEnd), true, nil

	case "response.incomplete":
		reason := StopError
		if resp, ok := obj["response"].(map[string]any); ok {
			c.applyResponsesUsage(resp)
			if det, ok := resp["incomplete_details"].(map[string]any); ok {
				if d, _ := det["reason"].(string); d == "max_output_tokens" {
					reason = StopLength
				}
			}
		}
		return c.finish(reason), true, nil

	case "response.failed":
		msg := ""
		if resp, ok := obj["response"].(map[string]any); ok {
			if e, ok := resp["error"].(map[string]any); ok {
				msg = stringVal(e, "code") + ": " + stringVal(e, "message")
			}
		}
		return nil, true, errStreamError(msg)

	default:
		return nil, false, nil
	}
}

func (c *openAIResponsesConverter) addSlot(idx int, itype string, item map[string]any) []Event {
	slot := &responsesSlot{}
	switch itype {
	case "message":
		slot.kind = ContentTypeText
		slot.index = len(c.partial.Content)
		slot.id = stringVal(item, "id")
		c.partial.Content = append(c.partial.Content, TextBlock(""))
		c.slots[idx] = slot
		return []Event{{Type: EventTextStart, ContentIndex: slot.index, Partial: c.partial}}
	case "reasoning":
		slot.kind = ContentTypeThinking
		slot.index = len(c.partial.Content)
		c.partial.Content = append(c.partial.Content, ThinkingBlock(""))
		c.slots[idx] = slot
		return []Event{{Type: EventThinkingStart, ContentIndex: slot.index, Partial: c.partial}}
	case "function_call", "custom_tool_call":
		slot.kind = ContentTypeToolCall
		slot.index = len(c.partial.Content)
		slot.name = stringVal(item, "name")
		slot.id = stringVal(item, "id")
		if slot.id == "" {
			slot.id = stringVal(item, "call_id")
		}
		c.partial.Content = append(c.partial.Content, ToolCallBlock(slot.id, slot.name, map[string]any{}))
		c.slots[idx] = slot
		return []Event{{Type: EventToolCallStart, ContentIndex: slot.index, Partial: c.partial}}
	}
	return nil
}

func (c *openAIResponsesConverter) appendTextSlot(idx int, delta string) []Event {
	slot, ok := c.slots[idx]
	if !ok || slot.kind != ContentTypeText {
		return nil
	}
	c.partial.Content[slot.index].Text += delta
	return []Event{{Type: EventTextDelta, ContentIndex: slot.index, Delta: delta, Partial: c.partial}}
}

func (c *openAIResponsesConverter) appendThinkingSlot(idx int, delta string) []Event {
	slot, ok := c.slots[idx]
	if !ok || slot.kind != ContentTypeThinking {
		return nil
	}
	c.partial.Content[slot.index].Thinking += delta
	return []Event{{Type: EventThinkingDelta, ContentIndex: slot.index, Delta: delta, Partial: c.partial}}
}

func (c *openAIResponsesConverter) appendToolArgs(idx int, delta string) []Event {
	slot, ok := c.slots[idx]
	if !ok || slot.kind != ContentTypeToolCall {
		return nil
	}
	slot.argsJSON += delta
	c.partial.Content[slot.index].Arguments = parseStreamingJson(slot.argsJSON)
	return []Event{{Type: EventToolCallDelta, ContentIndex: slot.index, Delta: delta, Partial: c.partial}}
}

func (c *openAIResponsesConverter) finalizeSlot(idx int) []Event {
	slot, ok := c.slots[idx]
	if !ok {
		return nil
	}
	i := slot.index
	done := false
	defer func() {
		if done {
			delete(c.slots, idx)
		}
	}()
	switch slot.kind {
	case ContentTypeText:
		done = true
		return []Event{{Type: EventTextEnd, ContentIndex: i, Content: c.partial.Content[i].Text, Partial: c.partial}}
	case ContentTypeThinking:
		done = true
		return []Event{{Type: EventThinkingEnd, ContentIndex: i, Content: c.partial.Content[i].Thinking, Partial: c.partial}}
	case ContentTypeToolCall:
		done = true
		c.partial.Content[i].Arguments = parseStreamingJson(slot.argsJSON)
		return []Event{{Type: EventToolCallEnd, ContentIndex: i, ToolCall: &c.partial.Content[i], Partial: c.partial}}
	}
	return nil
}

func (c *openAIResponsesConverter) applyResponsesUsage(resp map[string]any) {
	usage, _ := resp["usage"].(map[string]any)
	if usage == nil {
		return
	}
	in := intOr(usage, "input_tokens")
	out := intOr(usage, "output_tokens")
	total := intOr(usage, "total_tokens")
	cached := 0
	if det, ok := usage["input_tokens_details"].(map[string]any); ok {
		cached = intOr(det, "cached_tokens")
		c.partial.Usage.CacheWrite = intOr(det, "cache_write_tokens")
	}
	c.partial.Usage.Input = in
	c.partial.Usage.CacheRead = cached
	c.partial.Usage.Output = out
	c.partial.Usage.TotalTokens = total
	if total == 0 {
		c.partial.Usage.TotalTokens = in + out
	}
	c.partial.Usage.Cost = calculateCost(c.model, &c.partial.Usage)
}

func (c *openAIResponsesConverter) finish(reason StopReason) []Event {
	if c.finalized {
		return nil
	}
	c.finalized = true
	c.partial.StopReason = reason
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
	out = append(out, Event{Type: EventDone, Reason: reason, Message: c.partial})
	return out
}

func intOf(m map[string]any, key string) int {
	if v, ok := m[key].(float64); ok {
		return int(v)
	}
	return 0
}
