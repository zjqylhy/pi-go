package tui

import "testing"

func TestColorParams(t *testing.T) {
	if got := Red.fgParams(); got != "31" {
		t.Fatalf("Red fg = %q", got)
	}
	if got := Red.bgParams(); got != "41" {
		t.Fatalf("Red bg = %q", got)
	}
	if got := BrightGreen.fgParams(); got != "92" {
		t.Fatalf("BrightGreen fg = %q", got)
	}
	if got := Color256(208).fgParams(); got != "38;5;208" {
		t.Fatalf("256 fg = %q", got)
	}
	if got := RGB(255, 0, 100).bgParams(); got != "48;2;255;0;100" {
		t.Fatalf("truecolor bg = %q", got)
	}
}

func TestStyleSGRAndWrap(t *testing.T) {
	fg := Red
	s := Style{Fg: &fg, Bold: true}
	if got := s.SGR(); got != "\x1b[1;31m" {
		t.Fatalf("SGR = %q", got)
	}
	if got := s.Wrap("hi"); got != "\x1b[1;31mhi\x1b[0m" {
		t.Fatalf("Wrap = %q", got)
	}
	if Reset.Wrap("plain") != "plain" {
		t.Fatalf("reset wrap should be plain")
	}
}

func TestStripAnsiAndVisibleWidth(t *testing.T) {
	s := "\x1b[1;31mred\x1b[0m word"
	if got := StripAnsi(s); got != "red word" {
		t.Fatalf("StripAnsi = %q", got)
	}
	if got := VisibleWidth(s); got != len([]rune("red word")) {
		t.Fatalf("VisibleWidth = %d", got)
	}
}
