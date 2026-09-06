package harness

import (
	"fmt"
	"strings"
)

const (
	DefaultMaxLines = 2000
	DefaultMaxBytes = 50 * 1024
)

// TruncationResult describes the outcome of truncation.
type TruncationResult struct {
	Content               string
	Truncated             bool
	TruncatedBy           string // "lines" | "bytes" | ""
	TotalLines            int
	TotalBytes            int
	OutputLines           int
	OutputBytes           int
	LastLinePartial       bool
	FirstLineExceedsLimit bool
	MaxLines              int
	MaxBytes              int
}

func utf8ByteLength(s string) int { return len([]byte(s)) }

func FormatSize(n int) string {
	switch {
	case n < 1024:
		return fmt.Sprintf("%dB", n)
	case n < 1024*1024:
		return fmt.Sprintf("%.1fKB", float64(n)/1024)
	default:
		return fmt.Sprintf("%.1fMB", float64(n)/(1024*1024))
	}
}

func splitLines(content string) []string {
	if content == "" {
		return nil
	}
	lines := strings.Split(content, "\n")
	if strings.HasSuffix(content, "\n") {
		lines = lines[:len(lines)-1]
	}
	return lines
}

// TruncateHead keeps the first maxLines/maxBytes of content.
func TruncateHead(content string, maxLines, maxBytes int) TruncationResult {
	if maxLines <= 0 {
		maxLines = DefaultMaxLines
	}
	if maxBytes <= 0 {
		maxBytes = DefaultMaxBytes
	}
	lines := splitLines(content)
	totalBytes := utf8ByteLength(content)
	totalLines := len(lines)

	if totalLines <= maxLines && totalBytes <= maxBytes {
		return TruncationResult{
			Content: content, Truncated: false, TotalLines: totalLines, TotalBytes: totalBytes,
			OutputLines: totalLines, OutputBytes: totalBytes, MaxLines: maxLines, MaxBytes: maxBytes,
		}
	}

	if len(lines) > 0 && utf8ByteLength(lines[0]) > maxBytes {
		return TruncationResult{
			Content: "", Truncated: true, TruncatedBy: "bytes", TotalLines: totalLines, TotalBytes: totalBytes,
			OutputLines: 0, OutputBytes: 0, FirstLineExceedsLimit: true, MaxLines: maxLines, MaxBytes: maxBytes,
		}
	}

	outputLines := []string{}
	outputBytes := 0
	truncatedBy := "lines"
	for i := 0; i < len(lines) && i < maxLines; i++ {
		line := lines[i]
		lineBytes := utf8ByteLength(line)
		if i > 0 {
			lineBytes++ // newline
		}
		if outputBytes+lineBytes > maxBytes {
			truncatedBy = "bytes"
			break
		}
		outputLines = append(outputLines, line)
		outputBytes += lineBytes
	}
	if len(outputLines) >= maxLines && outputBytes <= maxBytes {
		truncatedBy = "lines"
	}
	outContent := strings.Join(outputLines, "\n")

	return TruncationResult{
		Content: outContent, Truncated: true, TruncatedBy: truncatedBy,
		TotalLines: totalLines, TotalBytes: totalBytes,
		OutputLines: len(outputLines), OutputBytes: utf8ByteLength(outContent),
		MaxLines: maxLines, MaxBytes: maxBytes,
	}
}

// TruncateTail keeps the last maxLines/maxBytes of content.
func TruncateTail(content string, maxLines, maxBytes int) TruncationResult {
	if maxLines <= 0 {
		maxLines = DefaultMaxLines
	}
	if maxBytes <= 0 {
		maxBytes = DefaultMaxBytes
	}
	lines := splitLines(content)
	totalBytes := utf8ByteLength(content)
	totalLines := len(lines)

	if totalLines <= maxLines && totalBytes <= maxBytes {
		return TruncationResult{
			Content: content, Truncated: false, TotalLines: totalLines, TotalBytes: totalBytes,
			OutputLines: totalLines, OutputBytes: totalBytes, MaxLines: maxLines, MaxBytes: maxBytes,
		}
	}

	outputLines := []string{}
	outputBytes := 0
	truncatedBy := "lines"
	lastLinePartial := false

	for i := len(lines) - 1; i >= 0 && len(outputLines) < maxLines; i-- {
		line := lines[i]
		lineBytes := utf8ByteLength(line)
		if len(outputLines) > 0 {
			lineBytes++
		}
		if outputBytes+lineBytes > maxBytes {
			truncatedBy = "bytes"
			if len(outputLines) == 0 {
				t := truncateStringFromEnd(line, maxBytes)
				outputLines = append([]string{t}, outputLines...)
				outputBytes = utf8ByteLength(t)
				lastLinePartial = true
			}
			break
		}
		outputLines = append([]string{line}, outputLines...)
		outputBytes += lineBytes
	}
	if len(outputLines) >= maxLines && outputBytes <= maxBytes {
		truncatedBy = "lines"
	}
	outContent := strings.Join(outputLines, "\n")

	return TruncationResult{
		Content: outContent, Truncated: true, TruncatedBy: truncatedBy,
		TotalLines: totalLines, TotalBytes: totalBytes,
		OutputLines: len(outputLines), OutputBytes: utf8ByteLength(outContent),
		LastLinePartial: lastLinePartial, MaxLines: maxLines, MaxBytes: maxBytes,
	}
}

func truncateStringFromEnd(s string, maxBytes int) string {
	if maxBytes <= 0 {
		return ""
	}
	b := []byte(s)
	if len(b) <= maxBytes {
		return s
	}
	// Keep the last maxBytes bytes, adjusted to a valid rune boundary.
	cut := b[len(b)-maxBytes:]
	for len(cut) > 0 && (cut[0]&0xC0) == 0x80 {
		cut = cut[1:]
	}
	return string(cut)
}
