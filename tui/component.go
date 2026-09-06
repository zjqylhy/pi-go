// Package tui is a terminal UI library built around differential rendering:
// components render to a frame of text lines, and a Screen re-draws only the
// lines that changed between frames. It mirrors the core idea of pi-tui.
package tui

// Component renders itself to terminal lines for a given viewport width.
type Component interface {
	Render(width int) []string
}

// Text is a static block of lines.
type Text struct {
	Lines []string
}

func NewText(lines ...string) *Text { return &Text{Lines: lines} }

func (t *Text) Render(width int) []string {
	out := make([]string, 0, len(t.Lines))
	for _, l := range t.Lines {
		out = append(out, truncateToWidth(l, width))
	}
	return out
}

// VStack renders children stacked vertically, each getting the full width.
type VStack struct {
	Children []Component
}

func (v *VStack) Render(width int) []string {
	var out []string
	for _, c := range v.Children {
		if c == nil {
			continue
		}
		out = append(out, c.Render(width)...)
	}
	return out
}

// Filler pads vertical space with empty lines.
type Filler struct {
	Count int
}

func (f *Filler) Render(width int) []string {
	if f.Count <= 0 {
		return nil
	}
	out := make([]string, f.Count)
	for i := range out {
		out[i] = ""
	}
	return out
}

func truncateToWidth(s string, width int) string {
	if width <= 0 {
		return s
	}
	// Count runes, not bytes, for terminal width (approximately).
	runes := []rune(s)
	if len(runes) <= width {
		return s
	}
	return string(runes[:width])
}
