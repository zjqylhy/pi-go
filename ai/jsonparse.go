package ai

import (
	"encoding/json"
	"strings"
	"unicode/utf8"
)

// parseStreamingJson parses partial JSON from a streamed tool-call argument
// buffer. It returns {} for empty input and otherwise tries progressively more
// aggressive repair before finally returning {}.
func parseStreamingJson(raw string) map[string]any {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return map[string]any{}
	}
	var v map[string]any
	if json.Unmarshal([]byte(trimmed), &v) == nil {
		return v
	}
	if json.Unmarshal([]byte(repairJson(trimmed)), &v) == nil {
		return v
	}
	// Best-effort partial-json: attempt to close open brackets/braces.
	if json.Unmarshal([]byte(closePartialJson(trimmed)), &v) == nil {
		return v
	}
	return map[string]any{}
}

// repairJson fixes the most common streaming malformations: raw control chars
// and truncated escapes.
func repairJson(s string) string {
	var b strings.Builder
	runes := []rune(s)
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		if r < 0x20 {
			b.WriteString(escapeControl(r))
			continue
		}
		if r == '\\' && i+1 >= len(runes) {
			// trailing backslash: drop it
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

func escapeControl(r rune) string {
	switch r {
	case '\n':
		return "\\n"
	case '\r':
		return "\\r"
	case '\t':
		return "\\t"
	default:
		return " "
	}
}

// closePartialJson is a tiny best-effort bracket/brace closer for truncated
// JSON fragments (e.g. streaming tool arguments).
func closePartialJson(s string) string {
	openBrace := strings.Count(s, "{") - strings.Count(s, "}")
	openBracket := strings.Count(s, "[") - strings.Count(s, "]")
	// close un-terminated string
	if trailingBackslashOdd(s) {
		s = s[:len(s)-1]
	}
	var b strings.Builder
	b.WriteString(s)
	for i := 0; i < openBracket; i++ {
		b.WriteByte(']')
	}
	for i := 0; i < openBrace; i++ {
		b.WriteByte('}')
	}
	return b.String()
}

func trailingBackslashOdd(s string) bool {
	n := 0
	for i := len(s) - 1; i >= 0 && s[i] == '\\'; i-- {
		n++
	}
	return n%2 == 1
}

// validUTF8 reports whether b is valid UTF-8.
func validUTF8(b []byte) bool { return utf8.Valid(b) }
