//go:build !windows

package tui

// newKeyReader is the non-Windows fallback: raw console mode is not yet
// implemented for this platform.
func newKeyReader() (KeyReader, error) {
	return nil, ErrRawModeUnsupported
}
