package ai

import "strings"

// TransformMessages rewrites the internal Message transcript into a form
// suitable for a target model: it downgrades unsupported images, converts
// cross-model thinking into text, drops incomplete turns, and inserts synthetic
// tool results for orphaned tool calls.
func TransformMessages(messages []Message, model *Model) []Message {
	supportsImage := false
	for _, i := range model.Input {
		if i == "image" {
			supportsImage = true
			break
		}
	}

	// First pass: per-message content transforms.
	first := make([]Message, 0, len(messages))
	for _, msg := range messages {
		switch m := msg.(type) {
		case UserMessage:
			if !supportsImage {
				m.Content = downgradeImages(m.Content, false, false)
			}
			first = append(first, m)
		case *UserMessage:
			cp := *m
			if !supportsImage {
				cp.Content = downgradeImages(m.Content, false, false)
			}
			first = append(first, &cp)
		case ToolResultMessage:
			if !supportsImage {
				m.Content = downgradeImages(m.Content, true, false)
			}
			first = append(first, m)
		case *ToolResultMessage:
			cp := *m
			if !supportsImage {
				cp.Content = downgradeImages(m.Content, true, false)
			}
			first = append(first, &cp)
		case AssistantMessage:
			am := transformAssistant(m, model)
			if am != nil {
				first = append(first, am)
			}
		case *AssistantMessage:
			first = append(first, transformAssistant(*m, model))
		}
	}

	// Second pass: synthetic tool results for orphaned tool calls.
	return insertSyntheticToolResults(first)
}

func transformAssistant(m AssistantMessage, model *Model) Message {
	isSameModel := m.Provider == model.Provider && m.API == model.API && m.Model == model.ID
	blocks := make([]ContentBlock, 0, len(m.Content))
	for _, b := range m.Content {
		switch b.Type {
		case ContentTypeThinking:
			if b.Redacted {
				if isSameModel {
					blocks = append(blocks, b)
				}
				continue
			}
			if isSameModel && b.ThinkingSignature != "" {
				blocks = append(blocks, b)
				continue
			}
			if strings.TrimSpace(b.Thinking) == "" {
				continue
			}
			if isSameModel {
				blocks = append(blocks, b)
			} else {
				blocks = append(blocks, TextBlock(b.Thinking))
			}
		case ContentTypeToolCall:
			nb := b
			if !isSameModel {
				nb.ThoughtSignature = ""
			}
			blocks = append(blocks, nb)
		default:
			blocks = append(blocks, b)
		}
	}
	m.Content = blocks
	return m
}

func downgradeImages(blocks []ContentBlock, toolResult, _ bool) []ContentBlock {
	out := make([]ContentBlock, 0, len(blocks))
	consecutive := 0
	for _, b := range blocks {
		if b.Type == ContentTypeImage {
			consecutive++
			continue
		}
		if consecutive > 0 {
			text := "(image omitted: model does not support images)"
			out = append(out, TextBlock(text))
			consecutive = 0
		}
		out = append(out, b)
	}
	if consecutive > 0 {
		text := "(image omitted: model does not support images)"
		if toolResult {
			text = "(tool image omitted: model does not support images)"
		}
		out = append(out, TextBlock(text))
	}
	return out
}

type pendingToolCall struct {
	id       string
	toolName string
}

func insertSyntheticToolResults(messages []Message) []Message {
	var pending []pendingToolCall
	existing := map[string]bool{}

	flush := func(out *[]Message) {
		for _, tc := range pending {
			if existing[tc.id] {
				continue
			}
			*out = append(*out, &ToolResultMessage{
				Role:       "toolResult",
				ToolCallID: tc.id,
				ToolName:   tc.toolName,
				Content:    []ContentBlock{TextBlock("No result provided")},
				IsError:    true,
				Timestamp:  nowMs(),
			})
		}
		pending = nil
	}

	out := []Message{}
	for i, msg := range messages {
		switch m := msg.(type) {
		case AssistantMessage:
			flush(&out)
			if m.StopReason == StopError || m.StopReason == StopAborted {
				continue
			}
			for _, b := range m.Content {
				if b.Type == ContentTypeToolCall {
					pending = append(pending, pendingToolCall{id: b.ID, toolName: b.Name})
				}
			}
			out = append(out, m)
		case *AssistantMessage:
			flush(&out)
			if m.StopReason == StopError || m.StopReason == StopAborted {
				continue
			}
			for _, b := range m.Content {
				if b.Type == ContentTypeToolCall {
					pending = append(pending, pendingToolCall{id: b.ID, toolName: b.Name})
				}
			}
			out = append(out, m)
		case ToolResultMessage:
			existing[m.ToolCallID] = true
			out = append(out, m)
		case *ToolResultMessage:
			existing[m.ToolCallID] = true
			out = append(out, m)
		case UserMessage:
			flush(&out)
			out = append(out, m)
		case *UserMessage:
			flush(&out)
			out = append(out, m)
		default:
			out = append(out, msg)
		}
		_ = i
	}
	flush(&out)
	return out
}

// SplitDeferredTools partitions tools into those needed immediately and those
// loaded from the transcript (deferred). Immediate tools are returned in order;
// deferred tools are returned keyed by name.
func SplitDeferredTools(context *Context, enabled bool) (immediate []Tool, deferred map[string]Tool) {
	deferred = map[string]Tool{}
	byName := map[string]Tool{}
	var order []string
	for _, t := range context.Tools {
		if _, ok := byName[t.Name]; !ok {
			order = append(order, t.Name)
		}
		byName[t.Name] = t
	}

	if !enabled {
		out := make([]Tool, 0, len(order))
		for _, n := range order {
			out = append(out, byName[n])
		}
		return out, nil
	}

	deferredNames := map[string]bool{}
	for _, msg := range context.Messages {
		switch m := msg.(type) {
		case AssistantMessage:
			for _, b := range m.Content {
				if b.Type == ContentTypeToolCall {
					deferredNames[b.Name] = true
				}
			}
		case *AssistantMessage:
			for _, b := range m.Content {
				if b.Type == ContentTypeToolCall {
					deferredNames[b.Name] = true
				}
			}
		case ToolResultMessage:
			for _, n := range m.AddedToolNames {
				deferredNames[n] = true
			}
		case *ToolResultMessage:
			for _, n := range m.AddedToolNames {
				deferredNames[n] = true
			}
		}
	}

	immediate = []Tool{}
	for _, n := range order {
		if deferredNames[n] {
			deferred[n] = byName[n]
		} else {
			immediate = append(immediate, byName[n])
		}
	}
	return immediate, deferred
}
