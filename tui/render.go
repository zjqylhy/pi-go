package tui

import (
	"fmt"
	"io"
	"strings"
)

// Op is a single line-level render operation.
type Op struct {
	Row  int    // zero-based row
	Text string // full new content for that row
}

// Diff returns the render operations needed to transform the terminal from
// prev to next. It covers kept/added rows only; rows that disappear are cleared
// by Screen.Render itself.
func Diff(prev, next []string) []Op {
	var ops []Op
	for i := 0; i < len(next); i++ {
		var p string
		if i < len(prev) {
			p = prev[i]
		}
		if n := next[i]; p != n {
			ops = append(ops, Op{Row: i, Text: n})
		}
	}
	return ops
}

// Screen writes differential line updates to a terminal output stream.
type Screen struct {
	Out  io.Writer
	prev []string
}

func NewScreen(out io.Writer) *Screen { return &Screen{Out: out} }

// Render writes only the changed rows, then clears any rows that disappeared.
func (s *Screen) Render(next []string) {
	var b strings.Builder
	for _, op := range Diff(s.prev, next) {
		fmt.Fprintf(&b, "\x1b[%d;1H%s\x1b[K", op.Row+1, op.Text)
	}
	if len(s.prev) > len(next) {
		for r := len(next); r < len(s.prev); r++ {
			fmt.Fprintf(&b, "\x1b[%d;1H\x1b[2K", r+1)
		}
	}
	if len(next) > 0 {
		fmt.Fprintf(&b, "\x1b[%d;1H", len(next))
	}
	if b.Len() > 0 {
		_, _ = io.WriteString(s.Out, b.String())
	}
	s.prev = append([]string{}, next...)
}

// Clear clears the whole screen and resets the frame.
func (s *Screen) Clear() {
	_, _ = io.WriteString(s.Out, "\x1b[2J\x1b[H")
	s.prev = nil
}

// HideCursor hides the terminal cursor.
func (s *Screen) HideCursor() { _, _ = io.WriteString(s.Out, "\x1b[?25l") }

// ShowCursor shows the terminal cursor.
func (s *Screen) ShowCursor() { _, _ = io.WriteString(s.Out, "\x1b[?25h") }
