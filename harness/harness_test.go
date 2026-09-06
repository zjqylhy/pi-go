package harness

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func testEnv(t *testing.T) ExecutionEnv {
	t.Helper()
	dir := t.TempDir()
	return NewLocalEnv(dir)
}

func TestWriteAndReadTool(t *testing.T) {
	env := testEnv(t)
	write := CreateWriteTool(env)
	read := CreateReadTool(env, nil)

	if _, err := write.Execute("id1", map[string]any{"path": "a.txt", "content": "line1\nline2\nline3\n"}, context.Background(), nil); err != nil {
		t.Fatalf("write: %v", err)
	}
	res, err := read.Execute("id2", map[string]any{"path": "a.txt"}, context.Background(), nil)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	text := res.Content[0].Text
	if !strings.Contains(text, "line1") || !strings.Contains(text, "line3") {
		t.Fatalf("read output missing content: %q", text)
	}
}

func TestEditTool(t *testing.T) {
	env := testEnv(t)
	write := CreateWriteTool(env)
	edit := CreateEditTool(env)
	read := CreateReadTool(env, nil)

	path := filepath.Join(env.Cwd(), "f.txt")
	if err := os.WriteFile(path, []byte("hello world\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	_ = write

	_, err := edit.Execute("id", map[string]any{
		"path":  "f.txt",
		"edits": []any{map[string]any{"oldText": "world", "newText": "there"}},
	}, context.Background(), nil)
	if err != nil {
		t.Fatalf("edit: %v", err)
	}

	res, _ := read.Execute("id", map[string]any{"path": "f.txt"}, context.Background(), nil)
	if !strings.Contains(res.Content[0].Text, "hello there") {
		t.Fatalf("expected edited content, got %q", res.Content[0].Text)
	}
}

func TestEditDuplicateRejected(t *testing.T) {
	path := "x.txt"
	_, err := ApplyEditsToNormalizedContent("a\na\n", []Edit{{OldText: "a", NewText: "b"}}, path)
	if err == nil || !strings.Contains(err.Error(), "unique") {
		t.Fatalf("expected duplicate error, got %v", err)
	}
}

func TestEditOverlapRejected(t *testing.T) {
	_, err := ApplyEditsToNormalizedContent("hello world", []Edit{
		{OldText: "hello", NewText: "hi"},
		{OldText: "hello world", NewText: "x"},
	}, "x.txt")
	if err == nil || !strings.Contains(err.Error(), "overlap") {
		t.Fatalf("expected overlap error, got %v", err)
	}
}

func TestEditBasicApply(t *testing.T) {
	res, err := ApplyEditsToNormalizedContent("a\nb\nc\n", []Edit{{OldText: "b", NewText: "B"}}, "x.txt")
	if err != nil {
		t.Fatalf("apply: %v", err)
	}
	if res.NewContent != "a\nB\nc\n" {
		t.Fatalf("unexpected new content %q", res.NewContent)
	}
}

func TestBashTool(t *testing.T) {
	env := testEnv(t)
	bash := CreateBashTool(env, nil)
	res, err := bash.Execute("id", map[string]any{"command": "echo hello"}, context.Background(), nil)
	if err != nil {
		t.Fatalf("bash: %v", err)
	}
	if !strings.Contains(res.Content[0].Text, "hello") {
		t.Fatalf("expected bash output to contain hello, got %q", res.Content[0].Text)
	}
}

func TestBashNonZeroExit(t *testing.T) {
	env := testEnv(t)
	bash := CreateBashTool(env, nil)
	_, err := bash.Execute("id", map[string]any{"command": "exit 3"}, context.Background(), nil)
	if err == nil || !strings.Contains(err.Error(), "code 3") {
		t.Fatalf("expected exit code error, got %v", err)
	}
}

func TestDetectImageMimeType(t *testing.T) {
	png := []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 'I', 'H', 'D', 'R'}
	if DetectImageMimeType(png) != "image/png" {
		t.Fatalf("expected png")
	}
	if DetectImageMimeType([]byte("plain text content")) != "" {
		t.Fatalf("expected no image")
	}
}
