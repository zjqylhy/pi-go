package session

import (
	"encoding/json"
	"fmt"

	"pi-go/ai"
)

// marshalMessage serializes an ai.Message union (discriminated by Role).
func marshalMessage(m ai.Message) (json.RawMessage, error) {
	switch v := m.(type) {
	case *ai.UserMessage:
		return json.Marshal(v)
	case ai.UserMessage:
		return json.Marshal(&v)
	case *ai.AssistantMessage:
		return json.Marshal(v)
	case ai.AssistantMessage:
		return json.Marshal(&v)
	case *ai.ToolResultMessage:
		return json.Marshal(v)
	case ai.ToolResultMessage:
		return json.Marshal(&v)
	default:
		return nil, fmt.Errorf("unsupported message type %T", m)
	}
}

// unmarshalMessage reconstructs an ai.Message from JSON.
func unmarshalMessage(raw json.RawMessage) (ai.Message, error) {
	var probe struct {
		Role string `json:"role"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return nil, err
	}
	switch probe.Role {
	case "user":
		var m ai.UserMessage
		if err := json.Unmarshal(raw, &m); err != nil {
			return nil, err
		}
		return &m, nil
	case "assistant":
		var m ai.AssistantMessage
		if err := json.Unmarshal(raw, &m); err != nil {
			return nil, err
		}
		return &m, nil
	case "toolResult":
		var m ai.ToolResultMessage
		if err := json.Unmarshal(raw, &m); err != nil {
			return nil, err
		}
		return &m, nil
	default:
		return nil, fmt.Errorf("unknown message role %q", probe.Role)
	}
}

func marshalMessages(messages []ai.Message) ([]json.RawMessage, error) {
	out := make([]json.RawMessage, 0, len(messages))
	for _, m := range messages {
		raw, err := marshalMessage(m)
		if err != nil {
			return nil, err
		}
		out = append(out, raw)
	}
	return out, nil
}

func unmarshalMessages(raws []json.RawMessage) ([]ai.Message, error) {
	out := make([]ai.Message, 0, len(raws))
	for _, raw := range raws {
		m, err := unmarshalMessage(raw)
		if err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, nil
}
