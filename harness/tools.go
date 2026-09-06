package harness

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"pi-go/agent"
	"pi-go/ai"
)

func objSchema(props map[string]any, required ...string) map[string]any {
	return map[string]any{"type": "object", "properties": props, "required": required}
}

func strVal(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func numVal(m map[string]any, key string) int {
	switch v := m[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	case int64:
		return int(v)
	}
	return 0
}

// ReadToolOptions customize the read tool.
type ReadToolOptions struct {
	AutoResizeImages bool
	// ImageProcessor converts binary image data; returning ok=false yields a
	// text note instead of an image attachment.
	ImageProcessor func(bytes []byte, mimeType string) (data, outMime string, hints []string, ok bool)
}

// CreateReadTool builds the `read` tool.
func CreateReadTool(env ExecutionEnv, options *ReadToolOptions) *agent.AgentTool {
	return &agent.AgentTool{
		Tool: ai.Tool{
			Name:        "read",
			Description: fmt.Sprintf("Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to %d lines or %dKB (whichever is hit first). Use offset/limit for large files.", DefaultMaxLines, DefaultMaxBytes/1024),
			Parameters: objSchema(map[string]any{
				"path":   map[string]any{"type": "string"},
				"offset": map[string]any{"type": "number"},
				"limit":  map[string]any{"type": "number"},
			}, "path"),
		},
		Label: "read",
		Execute: func(toolCallID string, params map[string]any, ctx context.Context, onUpdate agent.AgentToolUpdateCallback) (*agent.AgentToolResult, error) {
			path := strVal(params, "path")
			abs, err := env.AbsolutePath(path)
			if err != nil {
				return nil, err
			}
			bytes, err := env.ReadBinary(abs)
			if err != nil {
				return nil, err
			}
			if mime := DetectImageMimeType(bytes); mime != "" {
				if options != nil && options.ImageProcessor != nil {
					data, outMime, hints, ok := options.ImageProcessor(bytes, mime)
					if !ok {
						return &agent.AgentToolResult{Content: []ai.ContentBlock{ai.TextBlock(fmt.Sprintf("Read image file [%s]\n%s", mime, "cannot process image"))}}, nil
					}
					text := fmt.Sprintf("Read image file [%s]", outMime)
					if len(hints) > 0 {
						text += "\n" + strings.Join(hints, "\n")
					}
					return &agent.AgentToolResult{Content: []ai.ContentBlock{ai.TextBlock(text), ai.ImageBlock(data, outMime)}}, nil
				}
				return &agent.AgentToolResult{Content: []ai.ContentBlock{
					ai.TextBlock(fmt.Sprintf("Read image file [%s]", mime)),
					ai.ImageBlock(base64.StdEncoding.EncodeToString(bytes), mime),
				}}, nil
			}

			return &agent.AgentToolResult{Content: []ai.ContentBlock{ai.TextBlock(readTextOutputWithParams(string(bytes), path, numVal(params, "offset"), numVal(params, "limit")))}}, nil
		},
	}
}

// readTextOutputWithParams applies offset/limit and truncation to a text file.
func readTextOutputWithParams(text string, path string, offset, limit int) string {
	lines := strings.Split(text, "\n")
	if strings.HasSuffix(text, "\n") {
		lines = lines[:len(lines)-1]
	}
	totalLines := len(lines)
	start := offset
	if start > 0 {
		start--
	}
	if start < 0 {
		start = 0
	}
	if start >= totalLines {
		return fmt.Sprintf("[Offset %d is beyond end of file (%d lines total)]", offset, totalLines)
	}
	end := totalLines
	if limit > 0 && start+limit < totalLines {
		end = start + limit
	}
	selected := strings.Join(lines[start:end], "\n")
	trunc := TruncateHead(selected, DefaultMaxLines, DefaultMaxBytes)
	if trunc.FirstLineExceedsLimit {
		return fmt.Sprintf("[Line %d exceeds %s limit. Use bash: sed -n '%dp' %s | head -c %d]", start+1, FormatSize(DefaultMaxBytes), start+1, path, DefaultMaxBytes)
	}
	out := trunc.Content
	if trunc.Truncated {
		endLineDisp := start + trunc.OutputLines
		next := endLineDisp + 1
		if trunc.TruncatedBy == "lines" {
			out += fmt.Sprintf("\n\n[Showing lines %d-%d of %d. Use offset=%d to continue.]", start+1, endLineDisp, totalLines, next)
		} else {
			out += fmt.Sprintf("\n\n[Showing lines %d-%d of %d (%s limit). Use offset=%d to continue.]", start+1, endLineDisp, totalLines, FormatSize(DefaultMaxBytes), next)
		}
	} else if limit > 0 && start+limit < totalLines {
		remaining := totalLines - (start + limit)
		out += fmt.Sprintf("\n\n[%d more lines in file. Use offset=%d to continue.]", remaining, start+limit+1)
	}
	return out
}

// CreateWriteTool builds the `write` tool.
func CreateWriteTool(env ExecutionEnv) *agent.AgentTool {
	return &agent.AgentTool{
		Tool: ai.Tool{
			Name:        "write",
			Description: "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
			Parameters: objSchema(map[string]any{
				"path":    map[string]any{"type": "string"},
				"content": map[string]any{"type": "string"},
			}, "path", "content"),
		},
		Label: "write",
		Execute: func(toolCallID string, params map[string]any, ctx context.Context, onUpdate agent.AgentToolUpdateCallback) (*agent.AgentToolResult, error) {
			path := strVal(params, "path")
			content := strVal(params, "content")
			abs, err := env.AbsolutePath(path)
			if err != nil {
				return nil, err
			}
			err = withFileMutation(abs, func() error {
				return env.WriteFile(abs, []byte(content))
			})
			if err != nil {
				return nil, err
			}
			return &agent.AgentToolResult{Content: []ai.ContentBlock{ai.TextBlock(fmt.Sprintf("Successfully wrote to %s", path))}}, nil
		},
	}
}

// CreateEditTool builds the `edit` tool.
func CreateEditTool(env ExecutionEnv) *agent.AgentTool {
	return &agent.AgentTool{
		Tool: ai.Tool{
			Name:        "edit",
			Description: "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file.",
			Parameters: objSchema(map[string]any{
				"path":  map[string]any{"type": "string"},
				"edits": map[string]any{"type": "array", "items": map[string]any{"type": "object", "properties": map[string]any{"oldText": map[string]any{"type": "string"}, "newText": map[string]any{"type": "string"}}, "required": []any{"oldText", "newText"}}},
			}, "path", "edits"),
		},
		Label:            "edit",
		PrepareArguments: prepareEditArgs,
		Execute: func(toolCallID string, params map[string]any, ctx context.Context, onUpdate agent.AgentToolUpdateCallback) (*agent.AgentToolResult, error) {
			path := strVal(params, "path")
			edits, err := editsFromParams(params)
			if err != nil {
				return nil, err
			}
			abs, err := env.AbsolutePath(path)
			if err != nil {
				return nil, err
			}
			err = withFileMutation(abs, func() error {
				info, err := env.FileInfo(abs)
				if err != nil {
					return fmt.Errorf("could not edit file: %s: %v", path, err)
				}
				if info.Kind != FileKindFile && info.Kind != FileKindSymlink {
					return fmt.Errorf("could not edit file: %s. Path is not a file.", path)
				}
				content, err := env.ReadText(abs)
				if err != nil {
					return fmt.Errorf("could not edit file: %s: %v", path, err)
				}
				bom, text := StripBom(content)
				ending := DetectLineEnding(text)
				normalized := NormalizeToLF(text)
				applied, err := ApplyEditsToNormalizedContent(normalized, edits, path)
				if err != nil {
					return err
				}
				final := bom + RestoreLineEndings(applied.NewContent, ending)
				if err := env.WriteFile(abs, []byte(final)); err != nil {
					return fmt.Errorf("could not edit file: %s: %v", path, err)
				}
				return nil
			})
			if err != nil {
				return nil, err
			}
			// Regenerate diff for details.
			info, _ := env.FileInfo(abs)
			_ = info
			return &agent.AgentToolResult{
				Content: []ai.ContentBlock{ai.TextBlock(fmt.Sprintf("Successfully replaced %d block(s) in %s.", len(edits), path))},
				Details: map[string]any{"edits": len(edits)},
			}, nil
		},
	}
}

func prepareEditArgs(args map[string]any) map[string]any {
	if raw, ok := args["edits"]; ok {
		switch v := raw.(type) {
		case string:
			var parsed any
			if json.Unmarshal([]byte(v), &parsed) == nil {
				if arr, ok := parsed.([]any); ok {
					args["edits"] = arr
				} else if m, ok := parsed.(map[string]any); ok && isSingleEdit(m) {
					args["edits"] = []any{m}
				}
			}
		case map[string]any:
			if isSingleEdit(v) {
				args["edits"] = []any{v}
			}
		}
	}
	// legacy fields
	if old, ok := args["oldText"].(string); ok {
		if newT, ok2 := args["newText"].(string); ok2 {
			var edits []any
			if e, ok := args["edits"].([]any); ok {
				edits = e
			}
			edits = append(edits, map[string]any{"oldText": old, "newText": newT})
			args["edits"] = edits
			delete(args, "oldText")
			delete(args, "newText")
		}
	}
	return args
}

func isSingleEdit(m map[string]any) bool {
	_, ok1 := m["oldText"].(string)
	_, ok2 := m["newText"].(string)
	return ok1 && ok2
}

func editsFromParams(params map[string]any) ([]Edit, error) {
	raw, ok := params["edits"]
	arr, ok := raw.([]any)
	if !ok || len(arr) == 0 {
		return nil, fmt.Errorf("edit tool input is invalid: edits must contain at least one replacement")
	}
	edits := make([]Edit, 0, len(arr))
	for _, e := range arr {
		m, ok := e.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("edit tool input is invalid")
		}
		edits = append(edits, Edit{OldText: strVal(m, "oldText"), NewText: strVal(m, "newText")})
	}
	return edits, nil
}

// BashToolOptions customize the bash tool.
type BashToolOptions struct {
	CommandPrefix string
}

// CreateBashTool builds the `bash` tool.
func CreateBashTool(env ExecutionEnv, options *BashToolOptions) *agent.AgentTool {
	return &agent.AgentTool{
		Tool: ai.Tool{
			Name:        "bash",
			Description: fmt.Sprintf("Execute a shell command in the current working directory. Returns combined stdout and stderr. Output is truncated to last %d lines or %dKB (whichever is hit first). Optionally provide a timeout in seconds.", DefaultMaxLines, DefaultMaxBytes/1024),
			Parameters: objSchema(map[string]any{
				"command": map[string]any{"type": "string"},
				"timeout": map[string]any{"type": "number"},
			}, "command"),
		},
		Label: "bash",
		Execute: func(toolCallID string, params map[string]any, ctx context.Context, onUpdate agent.AgentToolUpdateCallback) (*agent.AgentToolResult, error) {
			command := strVal(params, "command")
			timeout := numVal(params, "timeout")
			if timeout < 0 {
				return nil, fmt.Errorf("invalid timeout: must be a positive number of seconds")
			}
			if options != nil && options.CommandPrefix != "" {
				command = options.CommandPrefix + "\n" + command
			}
			result, err := env.Exec(command, ExecOptions{InheritEnv: true, TimeoutSec: timeout})
			output := result.Output
			trunc := TruncateTail(output, DefaultMaxLines, DefaultMaxBytes)
			text := trunc.Content
			if trunc.Truncated {
				text += fmt.Sprintf("\n\n[Showing last %d lines of %d (%s limit)]", trunc.OutputLines, trunc.TotalLines, FormatSize(DefaultMaxBytes))
			}
			if err != nil {
				return nil, fmt.Errorf("%s\n\n%s", text, err.Error())
			}
			if result.ExitCode != 0 {
				return nil, fmt.Errorf("%s\n\nCommand exited with code %d", text, result.ExitCode)
			}
			if text == "" {
				text = "(no output)"
			}
			return &agent.AgentToolResult{Content: []ai.ContentBlock{ai.TextBlock(text)}}, nil
		},
	}
}
