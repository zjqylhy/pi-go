package ai

import (
	"bufio"
	"io"
	"strings"
)

// sseEvent is a decoded Server-Sent-Events event.
type sseEvent struct {
	Event string
	Data  string
}

// scanSSE decodes a Server-Sent-Events stream, invoking fn for each event.
// Multi-line data fields are joined with newlines. Comments are ignored.
func scanSSE(r io.Reader, fn func(event, data string) error) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)

	var eventStr strings.Builder
	var dataStr strings.Builder
	hasData := false

	flush := func() error {
		if !hasData {
			return nil
		}
		data := strings.TrimSuffix(dataStr.String(), "\n")
		event := eventStr.String()
		eventStr.Reset()
		dataStr.Reset()
		hasData = false
		if event == "" && strings.TrimSpace(data) == "" {
			return nil
		}
		return fn(event, data)
	}

	for scanner.Scan() {
		line := scanner.Text()
		line = strings.TrimSuffix(line, "\r")
		if line == "" {
			if err := flush(); err != nil {
				return err
			}
			continue
		}
		switch {
		case strings.HasPrefix(line, ":"):
			// comment
		case strings.HasPrefix(line, "event:"):
			eventStr.WriteString(strings.TrimSpace(line[len("event:"):]))
		case strings.HasPrefix(line, "data:"):
			d := line[len("data:"):]
			if strings.HasPrefix(d, " ") {
				d = d[1:]
			}
			dataStr.WriteString(d)
			dataStr.WriteByte('\n')
			hasData = true
		}
	}
	if err := flush(); err != nil {
		return err
	}
	return scanner.Err()
}

// parseSSEData strips a leading "data:" line and returns its payload, or
// ("", false) for empty / [DONE] terminators. Used for JSON-lines SSE.
func parseSSEData(raw string) (string, bool) {
	for _, line := range strings.Split(raw, "\n") {
		if strings.HasPrefix(line, "data:") {
			d := line[len("data:"):]
			d = strings.TrimSpace(d)
			if d == "" || d == "[DONE]" {
				return "", false
			}
			return d, true
		}
	}
	return "", false
}

// discard drains an io.Reader.
func discard(r io.Reader) { _, _ = io.Copy(io.Discard, r) }
