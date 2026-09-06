package tui

import (
	"bytes"
	"strings"
	"testing"
)

func TestDiffOnlyChangedRows(t *testing.T) {
	prev := []string{"a", "b", "c"}
	next := []string{"a", "B", "c"}
	ops := Diff(prev, next)
	if len(ops) != 1 || ops[0].Row != 1 || ops[0].Text != "B" {
		t.Fatalf("unexpected diff ops: %+v", ops)
	}
}

func TestDiffAddedAndRemovedRows(t *testing.T) {
	prev := []string{"a"}
	next := []string{"a", "b"}
	ops := Diff(prev, next)
	if len(ops) != 1 || ops[0].Row != 1 || ops[0].Text != "b" {
		t.Fatalf("unexpected diff ops: %+v", ops)
	}

	ops = Diff([]string{"a", "b", "c"}, []string{"a"})
	if len(ops) != 0 {
		t.Fatalf("removals are handled by Screen clearing, not diff ops: %+v", ops)
	}
}

func TestScreenRenderWritesChangedRow(t *testing.T) {
	var buf bytes.Buffer
	s := NewScreen(&buf)
	s.Render([]string{"a", "b"})
	buf.Reset()
	s.Render([]string{"a", "c"})

	out := buf.String()
	if !strings.Contains(out, "\x1b[2;1H") || !strings.Contains(out, "c") {
		t.Fatalf("expected row 2 update, got %q", out)
	}
}

func TestScreenRenderClearsRemovedRows(t *testing.T) {
	var buf bytes.Buffer
	s := NewScreen(&buf)
	s.Render([]string{"a", "b", "c"})
	buf.Reset()
	s.Render([]string{"a"})

	out := buf.String()
	if !strings.Contains(out, "\x1b[2;1H\x1b[2K") {
		t.Fatalf("expected trailing row clear, got %q", out)
	}
}

func TestTextAndVStack(t *testing.T) {
	v := &VStack{Children: []Component{NewText("one", "two"), NewText("three")}}
	frame := v.Render(80)
	if len(frame) != 3 || frame[0] != "one" || frame[2] != "three" {
		t.Fatalf("unexpected vstack frame %q", frame)
	}
}

func TestTextTruncates(t *testing.T) {
	got := NewText("abcdef").Render(3)
	if got[0] != "abc" {
		t.Fatalf("expected truncation to abc, got %q", got[0])
	}
}

func TestBottom(t *testing.T) {
	lines := []string{"1", "2", "3", "4", "5"}
	got := Bottom(lines, 3)
	if len(got) != 3 || got[0] != "3" || got[2] != "5" {
		t.Fatalf("unexpected bottom lines %q", got)
	}
}
