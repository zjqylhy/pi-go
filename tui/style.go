package tui

import (
	"fmt"
	"strings"
)

// Color is an ANSI color: basic (0-15), 256-color, or truecolor.
type Color struct {
	kind    uint8 // 0 = basic, 1 = 256, 2 = truecolor
	idx     uint8 // basic: 0-15, 256: 0-255
	r, g, b uint8 // truecolor components
}

const (
	colorBasic uint8 = iota
	color256
	colorTrue
)

// Basic named colors.
var (
	Black   = Color{kind: colorBasic, idx: 0}
	Red     = Color{kind: colorBasic, idx: 1}
	Green   = Color{kind: colorBasic, idx: 2}
	Yellow  = Color{kind: colorBasic, idx: 3}
	Blue    = Color{kind: colorBasic, idx: 4}
	Magenta = Color{kind: colorBasic, idx: 5}
	Cyan    = Color{kind: colorBasic, idx: 6}
	White   = Color{kind: colorBasic, idx: 7}

	BrightBlack   = Color{kind: colorBasic, idx: 8}
	BrightRed     = Color{kind: colorBasic, idx: 9}
	BrightGreen   = Color{kind: colorBasic, idx: 10}
	BrightYellow  = Color{kind: colorBasic, idx: 11}
	BrightBlue    = Color{kind: colorBasic, idx: 12}
	BrightMagenta = Color{kind: colorBasic, idx: 13}
	BrightCyan    = Color{kind: colorBasic, idx: 14}
	BrightWhite   = Color{kind: colorBasic, idx: 15}
)

// RGB returns a truecolor Color from 8-bit components.
func RGB(r, g, b int) Color {
	return Color{kind: colorTrue, r: clip8(r), g: clip8(g), b: clip8(b)}
}

// Color256 returns a 256-palette Color.
func Color256(i int) Color {
	return Color{kind: color256, idx: clip8(i)}
}

func clip8(v int) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

func (c Color) fgParams() string {
	switch c.kind {
	case colorBasic:
		if c.idx < 8 {
			return fmt.Sprintf("%d", 30+int(c.idx))
		}
		return fmt.Sprintf("%d", 90+int(c.idx-8))
	case color256:
		return fmt.Sprintf("38;5;%d", c.idx)
	case colorTrue:
		return fmt.Sprintf("38;2;%d;%d;%d", c.r, c.g, c.b)
	}
	return ""
}

func (c Color) bgParams() string {
	switch c.kind {
	case colorBasic:
		if c.idx < 8 {
			return fmt.Sprintf("%d", 40+int(c.idx))
		}
		return fmt.Sprintf("%d", 100+int(c.idx-8))
	case color256:
		return fmt.Sprintf("48;5;%d", c.idx)
	case colorTrue:
		return fmt.Sprintf("48;2;%d;%d;%d", c.r, c.g, c.b)
	}
	return ""
}

// Style describes text attributes for a run of text.
type Style struct {
	Fg, Bg    *Color
	Bold      bool
	Dim       bool
	Italic    bool
	Underline bool
	Blink     bool
	Reverse   bool
}

// Reset is the zero style (no attributes).
var Reset = Style{}

func (s Style) IsZero() bool {
	return s.Fg == nil && s.Bg == nil && !s.Bold && !s.Dim && !s.Italic && !s.Underline && !s.Blink && !s.Reverse
}

// SGR returns the ANSI SGR sequence for the style, or "" if it is the reset style.
func (s Style) SGR() string {
	var codes []string
	if s.Bold {
		codes = append(codes, "1")
	}
	if s.Dim {
		codes = append(codes, "2")
	}
	if s.Italic {
		codes = append(codes, "3")
	}
	if s.Underline {
		codes = append(codes, "4")
	}
	if s.Blink {
		codes = append(codes, "5")
	}
	if s.Reverse {
		codes = append(codes, "7")
	}
	if s.Fg != nil {
		codes = append(codes, s.Fg.fgParams())
	}
	if s.Bg != nil {
		codes = append(codes, s.Bg.bgParams())
	}
	if len(codes) == 0 {
		return ""
	}
	return "\x1b[" + strings.Join(codes, ";") + "m"
}

// Wrap applies the style around text, resetting at the end.
func (s Style) Wrap(text string) string {
	if s.IsZero() {
		return text
	}
	return s.SGR() + text + "\x1b[0m"
}

// StripAnsi removes CSI/SGR escape sequences from a string.
func StripAnsi(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); {
		if s[i] == 0x1b && i+1 < len(s) && s[i+1] == '[' {
			j := i + 2
			for j < len(s) && (s[j] < 0x40 || s[j] > 0x7e) {
				j++
			}
			if j < len(s) {
				i = j + 1
				continue
			}
		}
		b.WriteByte(s[i])
		i++
	}
	return b.String()
}

// VisibleWidth reports the terminal width of s in runes (ignoring ANSI codes).
func VisibleWidth(s string) int {
	return len([]rune(StripAnsi(s)))
}
