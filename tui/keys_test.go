package tui

import "testing"

func mustKey(t *testing.T, buf []byte) Key {
	t.Helper()
	k, _, ok := parseOne(buf)
	if !ok {
		t.Fatalf("incomplete key for %v", buf)
	}
	if k == nil {
		t.Fatalf("nil key for %v", buf)
	}
	return *k
}

func TestParseRunAndControl(t *testing.T) {
	if k := mustKey(t, []byte("a")); k != (Key{Code: KeyRune, Rune: 'a'}) {
		t.Fatalf("ascii key = %+v", k)
	}
	if k := mustKey(t, []byte{'\r'}); k.Code != KeyEnter {
		t.Fatalf("enter = %+v", k)
	}
	if k := mustKey(t, []byte{'\t'}); k.Code != KeyTab {
		t.Fatalf("tab = %+v", k)
	}
	if k := mustKey(t, []byte{0x7f}); k.Code != KeyBackspace {
		t.Fatalf("backspace = %+v", k)
	}
	if k := mustKey(t, []byte{0x03}); k != (Key{Code: KeyRune, Rune: 'c', Ctrl: true}) {
		t.Fatalf("ctrl-c = %+v", k)
	}
}

func TestParseArrowsAndModifiers(t *testing.T) {
	if k := mustKey(t, []byte("\x1b[A")); k != (Key{Code: KeyUp}) {
		t.Fatalf("up = %+v", k)
	}
	if k := mustKey(t, []byte("\x1b[1;5C")); k != (Key{Code: KeyRight, Ctrl: true}) {
		t.Fatalf("ctrl-right = %+v", k)
	}
	if k := mustKey(t, []byte("\x1b[1;2D")); k != (Key{Code: KeyLeft, Shift: true}) {
		t.Fatalf("shift-left = %+v", k)
	}
}

func TestParseFunctionAndNav(t *testing.T) {
	if k := mustKey(t, []byte("\x1b[3~")); k.Code != KeyDelete {
		t.Fatalf("delete = %+v", k)
	}
	if k := mustKey(t, []byte("\x1b[5~")); k.Code != KeyPageUp {
		t.Fatalf("pgup = %+v", k)
	}
	if k := mustKey(t, []byte("\x1b[H")); k.Code != KeyHome {
		t.Fatalf("home = %+v", k)
	}
	if k := mustKey(t, []byte("\x1bOP")); k.Code != KeyF1 {
		t.Fatalf("f1 = %+v", k)
	}
	if k := mustKey(t, []byte("\x1b[Z")); k != (Key{Code: KeyTab, Shift: true}) {
		t.Fatalf("shift-tab = %+v", k)
	}
}

func TestParserAcceptsPartial(t *testing.T) {
	var p KeyParser
	got := p.Process([]byte("\x1b["))
	if len(got) != 0 {
		t.Fatalf("partial should yield no keys, got %+v", got)
	}
	got = p.Process([]byte("A"))
	if len(got) != 1 || got[0] != (Key{Code: KeyUp}) {
		t.Fatalf("completed arrow = %+v", got)
	}
}

func TestParserMultiKey(t *testing.T) {
	var p KeyParser
	got := p.Process([]byte("ab\r"))
	if len(got) != 3 {
		t.Fatalf("expected 3 keys, got %+v", got)
	}
	if got[0].Rune != 'a' || got[1].Rune != 'b' || got[2].Code != KeyEnter {
		t.Fatalf("multi keys = %+v", got)
	}
}
