package tui

import "unicode/utf8"

// KeyCode identifies a non-printable key.
type KeyCode uint8

const (
	KeyUnknown KeyCode = iota
	KeyRune            // printable rune; see Key.Rune
	KeyEnter
	KeyTab
	KeyBackspace
	KeyEscape
	KeyDelete
	KeyInsert
	KeyHome
	KeyEnd
	KeyPageUp
	KeyPageDown
	KeyUp
	KeyDown
	KeyLeft
	KeyRight
	KeyF1
	KeyF2
	KeyF3
	KeyF4
	KeyF5
	KeyF6
	KeyF7
	KeyF8
	KeyF9
	KeyF10
	KeyF11
	KeyF12
)

// Key is a normalized key event.
type Key struct {
	Code  KeyCode
	Rune  rune
	Ctrl  bool
	Alt   bool
	Shift bool
}

func (k Key) IsZero() bool { return k.Code == KeyUnknown && k.Rune == 0 }

// KeyParser converts a raw byte stream into Key events, handling multi-byte
// UTF-8 and ANSI escape sequences (arrows, function keys, modifiers).
type KeyParser struct {
	pending []byte
}

// Process feeds raw bytes and returns any complete keys that were parsed.
func (p *KeyParser) Process(b []byte) []Key {
	p.pending = append(p.pending, b...)
	var out []Key
	for {
		k, n, ok := parseOne(p.pending)
		if !ok {
			break
		}
		p.pending = p.pending[n:]
		if k != nil {
			out = append(out, *k)
		}
	}
	return out
}

// Flush emits a lone pending escape byte as KeyEscape (used on timeout).
func (p *KeyParser) Flush() []Key {
	if len(p.pending) == 1 && p.pending[0] == 0x1b {
		p.pending = nil
		return []Key{{Code: KeyEscape}}
	}
	return nil
}

// parseOne parses the first complete key from buf, returning the key, the bytes
// consumed, and whether a complete key was available. ok=false means the buffer
// is incomplete (e.g. a partial escape sequence).
func parseOne(buf []byte) (*Key, int, bool) {
	if len(buf) == 0 {
		return nil, 0, false
	}
	b0 := buf[0]

	if b0 == 0x1b { // escape sequence
		if len(buf) < 2 {
			return nil, 0, false
		}
		switch buf[1] {
		case '[':
			return parseCSI(buf)
		case 'O':
			if len(buf) < 3 {
				return nil, 0, false
			}
			return parseSS3(buf)
		default:
			return &Key{Code: KeyEscape, Alt: true}, 2, true
		}
	}

	if b0 == '\r' || b0 == '\n' {
		return &Key{Code: KeyEnter}, 1, true
	}
	if b0 == '\t' {
		return &Key{Code: KeyTab}, 1, true
	}
	if b0 == 0x7f || b0 == 0x08 {
		return &Key{Code: KeyBackspace}, 1, true
	}
	if b0 < 0x20 || b0 == 0x7f {
		// Ctrl+letter: control byte 0x01..0x1a maps to 'a'..'z'.
		r := rune(b0 + 0x60)
		return &Key{Code: KeyRune, Rune: r, Ctrl: true}, 1, true
	}

	if b0 < 0x80 {
		return &Key{Code: KeyRune, Rune: rune(b0)}, 1, true
	}

	if !utf8.FullRune(buf) {
		return nil, 0, false
	}
	r, size := utf8.DecodeRune(buf)
	if r == utf8.RuneError && size == 1 {
		return &Key{Code: KeyUnknown}, 1, true
	}
	return &Key{Code: KeyRune, Rune: r}, size, true
}

// parseCSI handles ESC [ ... final sequences.
func parseCSI(buf []byte) (*Key, int, bool) {
	// find final byte (0x40-0x7e)
	end := -1
	for i := 2; i < len(buf); i++ {
		if buf[i] >= 0x40 && buf[i] <= 0x7e {
			end = i
			break
		}
	}
	if end < 0 {
		return nil, 0, false
	}
	final := buf[end]
	params := parseParams(buf[2:end])
	consume := end + 1

	ctrl, alt, shift := parseMods(params)

	mk := func(code KeyCode) *Key {
		return &Key{Code: code, Ctrl: ctrl, Alt: alt, Shift: shift}
	}

	switch final {
	case 'A':
		return mk(KeyUp), consume, true
	case 'B':
		return mk(KeyDown), consume, true
	case 'C':
		return mk(KeyRight), consume, true
	case 'D':
		return mk(KeyLeft), consume, true
	case 'H':
		return mk(KeyHome), consume, true
	case 'F':
		return mk(KeyEnd), consume, true
	case 'Z':
		return &Key{Code: KeyTab, Shift: true}, consume, true
	case '~':
		switch firstParam(params) {
		case 1, 7:
			return mk(KeyHome), consume, true
		case 2:
			return mk(KeyInsert), consume, true
		case 3:
			return mk(KeyDelete), consume, true
		case 4, 8:
			return mk(KeyEnd), consume, true
		case 5:
			return mk(KeyPageUp), consume, true
		case 6:
			return mk(KeyPageDown), consume, true
		case 11:
			return mk(KeyF1), consume, true
		case 12:
			return mk(KeyF2), consume, true
		case 13:
			return mk(KeyF3), consume, true
		case 14:
			return mk(KeyF4), consume, true
		case 15:
			return mk(KeyF5), consume, true
		case 17:
			return mk(KeyF6), consume, true
		case 18:
			return mk(KeyF7), consume, true
		case 19:
			return mk(KeyF8), consume, true
		case 20:
			return mk(KeyF9), consume, true
		case 21:
			return mk(KeyF10), consume, true
		case 23:
			return mk(KeyF11), consume, true
		case 24:
			return mk(KeyF12), consume, true
		}
	}
	return nil, consume, true
}

// parseSS3 handles ESC O ... sequences (F1-F4).
func parseSS3(buf []byte) (*Key, int, bool) {
	switch buf[2] {
	case 'P':
		return &Key{Code: KeyF1}, 3, true
	case 'Q':
		return &Key{Code: KeyF2}, 3, true
	case 'R':
		return &Key{Code: KeyF3}, 3, true
	case 'S':
		return &Key{Code: KeyF4}, 3, true
	}
	return nil, 3, true
}

func parseParams(b []byte) []int {
	if len(b) == 0 {
		return nil
	}
	var out []int
	cur := 0
	for i := 0; i <= len(b); i++ {
		if i == len(b) || b[i] == ';' {
			out = append(out, cur)
			cur = 0
			continue
		}
		cur = cur*10 + int(b[i]-'0')
	}
	return out
}

func firstParam(params []int) int {
	if len(params) == 0 {
		return 0
	}
	return params[0]
}

func parseMods(params []int) (ctrl, alt, shift bool) {
	m := 0
	if len(params) >= 2 {
		m = params[1]
	} else if len(params) == 1 {
		m = params[0]
	}
	return m >= 5,
		m == 3 || m == 4 || m == 7 || m == 8,
		m == 2 || m == 4 || m == 6 || m == 8
}
