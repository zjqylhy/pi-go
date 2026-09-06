package tui

import (
	"os"
	"strconv"
)

// TerminalSize reports the terminal size from COLUMNS/LINES, defaulting to
// 80x24 when the environment is not informative.
func TerminalSize() (w, h int) {
	w, h = 80, 24
	if c := os.Getenv("COLUMNS"); c != "" {
		if v, err := strconv.Atoi(c); err == nil && v > 0 {
			w = v
		}
	}
	if l := os.Getenv("LINES"); l != "" {
		if v, err := strconv.Atoi(l); err == nil && v > 0 {
			h = v
		}
	}
	return
}

// Bottom returns the last `height` lines (scrolled to the bottom).
func Bottom(lines []string, height int) []string {
	if height <= 0 {
		return nil
	}
	if len(lines) <= height {
		return lines
	}
	return lines[len(lines)-height:]
}
