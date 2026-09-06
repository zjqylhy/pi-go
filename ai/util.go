package ai

import (
	"time"
)

// nowMs returns the current unix time in milliseconds.
func nowMs() int64 { return time.Now().UnixMilli() }

// contentText flattens text content blocks into a single string.
func contentText(blocks []ContentBlock) string {
	var out string
	for _, b := range blocks {
		if b.Type == ContentTypeText {
			out += b.Text
		}
	}
	return out
}

// emptyUsage returns an all-zeros Usage.
func emptyUsage() Usage { return Usage{} }
