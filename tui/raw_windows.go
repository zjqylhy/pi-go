//go:build windows

package tui

import (
	"os"
	"syscall"
	"unsafe"
)

const (
	stdInputHandle             = uintptr(0xfffffff6)
	enableProcessedInput       = 0x0001
	enableLineInput            = 0x0002
	enableEchoInput            = 0x0004
	enableVirtualTerminalInput = 0x0200
)

var (
	kernel32           = syscall.NewLazyDLL("kernel32.dll")
	procGetStdHandle   = kernel32.NewProc("GetStdHandle")
	procGetConsoleMode = kernel32.NewProc("GetConsoleMode")
	procSetConsoleMode = kernel32.NewProc("SetConsoleMode")
)

func stdinConsoleHandle() uintptr {
	h, _, _ := procGetStdHandle.Call(stdInputHandle)
	return h
}

func getConsoleMode() (uint32, error) {
	var mode uint32
	r, _, err := procGetConsoleMode.Call(stdinConsoleHandle(), uintptr(unsafe.Pointer(&mode)))
	if r == 0 {
		return 0, err
	}
	return mode, nil
}

func setConsoleMode(mode uint32) error {
	r, _, err := procSetConsoleMode.Call(stdinConsoleHandle(), uintptr(mode))
	if r == 0 {
		return err
	}
	return nil
}

type windowsKeyReader struct {
	in       *os.File
	origMode uint32
	parser   KeyParser
	pending  []Key
}

func newKeyReader() (KeyReader, error) {
	orig, err := getConsoleMode()
	if err != nil {
		return nil, err
	}
	raw := orig &^ (enableEchoInput | enableLineInput | enableProcessedInput)
	raw |= enableVirtualTerminalInput
	if err := setConsoleMode(raw); err != nil {
		return nil, err
	}
	return &windowsKeyReader{in: os.Stdin, origMode: orig, parser: KeyParser{}}, nil
}

func (r *windowsKeyReader) ReadKey() (Key, error) {
	for len(r.pending) == 0 {
		buf := make([]byte, 16)
		n, err := r.in.Read(buf)
		if err != nil {
			return Key{}, err
		}
		r.pending = append(r.pending, r.parser.Process(buf[:n])...)
	}
	k := r.pending[0]
	r.pending = r.pending[1:]
	return k, nil
}

func (r *windowsKeyReader) Close() error {
	return setConsoleMode(r.origMode)
}
