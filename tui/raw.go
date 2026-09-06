package tui

import "errors"

// ErrRawModeUnsupported is returned when raw terminal mode is unavailable.
var ErrRawModeUnsupported = errors.New("tui: raw terminal mode is not supported on this platform")

// KeyReader reads normalized key events from the terminal.
type KeyReader interface {
	// ReadKey blocks until a key is available.
	ReadKey() (Key, error)
	// Close restores the terminal to its previous mode.
	Close() error
}

// NewKeyReader opens the terminal in raw mode (if supported) and returns a
// key reader. Callers must call Close to restore the terminal.
func NewKeyReader() (KeyReader, error) {
	return newKeyReader()
}
