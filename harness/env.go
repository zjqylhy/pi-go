// Package harness provides the built-in execution tools (read, write, edit,
// bash) plus the filesystem/shell abstraction they operate on.
package harness

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// FileKind is the kind of a filesystem object (symlinks not followed).
type FileKind = string

const (
	FileKindFile      FileKind = "file"
	FileKindDirectory FileKind = "directory"
	FileKindSymlink   FileKind = "symlink"
)

// FileInfo is stable filesystem metadata for one object.
type FileInfo struct {
	Name    string
	Path    string
	Kind    FileKind
	Size    int64
	MtimeMs int64
}

// ExecOptions configure a shell command execution.
type ExecOptions struct {
	Cwd        string
	Env        map[string]string
	InheritEnv bool
	TimeoutSec int
}

// ExecResult is the outcome of a shell command.
type ExecResult struct {
	Output   string // combined stdout + stderr
	ExitCode int
}

// ExecutionEnv is the filesystem and shell capability used by the built-in
// tools. Paths may be absolute or relative to Cwd().
type ExecutionEnv interface {
	Cwd() string
	AbsolutePath(path string) (string, error)
	ReadBinary(path string) ([]byte, error)
	ReadText(path string) (string, error)
	WriteFile(path string, data []byte) error
	FileInfo(path string) (FileInfo, error)
	Exists(path string) bool
	CanonicalPath(path string) (string, error)
	Exec(command string, opts ExecOptions) (ExecResult, error)
}

// LocalEnv is the OS-backed ExecutionEnv.
type LocalEnv struct {
	cwd string
}

// NewLocalEnv returns a LocalEnv rooted at cwd (defaults to the process cwd).
func NewLocalEnv(cwd string) *LocalEnv {
	if cwd == "" {
		if wd, err := os.Getwd(); err == nil {
			cwd = wd
		}
	}
	return &LocalEnv{cwd: cwd}
}

func (e *LocalEnv) Cwd() string { return e.cwd }

func (e *LocalEnv) AbsolutePath(path string) (string, error) {
	if filepath.IsAbs(path) {
		return filepath.Clean(path), nil
	}
	return filepath.Abs(filepath.Join(e.cwd, path))
}

func (e *LocalEnv) ReadBinary(path string) ([]byte, error) {
	abs, err := e.AbsolutePath(path)
	if err != nil {
		return nil, err
	}
	return os.ReadFile(abs)
}

func (e *LocalEnv) ReadText(path string) (string, error) {
	b, err := e.ReadBinary(path)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func (e *LocalEnv) WriteFile(path string, data []byte) error {
	abs, err := e.AbsolutePath(path)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		return err
	}
	return os.WriteFile(abs, data, 0o644)
}

func (e *LocalEnv) FileInfo(path string) (FileInfo, error) {
	abs, err := e.AbsolutePath(path)
	if err != nil {
		return FileInfo{}, err
	}
	li, err := os.Lstat(abs)
	if err != nil {
		return FileInfo{}, err
	}
	kind := FileKindFile
	switch {
	case li.Mode()&os.ModeSymlink != 0:
		kind = FileKindSymlink
	case li.IsDir():
		kind = FileKindDirectory
	}
	return FileInfo{
		Name:    li.Name(),
		Path:    abs,
		Kind:    kind,
		Size:    li.Size(),
		MtimeMs: li.ModTime().UnixMilli(),
	}, nil
}

func (e *LocalEnv) Exists(path string) bool {
	abs, err := e.AbsolutePath(path)
	if err != nil {
		return false
	}
	_, err = os.Stat(abs)
	return err == nil
}

func (e *LocalEnv) CanonicalPath(path string) (string, error) {
	abs, err := e.AbsolutePath(path)
	if err != nil {
		return "", err
	}
	return filepath.EvalSymlinks(abs)
}

func (e *LocalEnv) Exec(command string, opts ExecOptions) (ExecResult, error) {
	ctx := context.Background()
	var cancel context.CancelFunc
	if opts.TimeoutSec > 0 {
		ctx, cancel = context.WithTimeout(ctx, time.Duration(opts.TimeoutSec)*time.Second)
		defer cancel()
	}

	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.CommandContext(ctx, "cmd", "/C", command)
	} else {
		cmd = exec.CommandContext(ctx, "sh", "-c", command)
	}
	if opts.Cwd != "" {
		cmd.Dir = opts.Cwd
	}
	if opts.InheritEnv {
		cmd.Env = os.Environ()
		for k, v := range opts.Env {
			cmd.Env = append(cmd.Env, k+"="+v)
		}
	}

	var out strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &out

	err := cmd.Run()
	switch {
	case ctx.Err() == context.DeadlineExceeded:
		return ExecResult{Output: out.String()}, fmt.Errorf("command timed out after %d seconds", opts.TimeoutSec)
	case ctx.Err() == context.Canceled:
		return ExecResult{Output: out.String()}, fmt.Errorf("command aborted")
	}
	exitCode := 0
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else if ctx.Err() != nil {
			return ExecResult{Output: out.String()}, ctx.Err()
		} else {
			return ExecResult{Output: out.String()}, err
		}
	}
	return ExecResult{Output: out.String(), ExitCode: exitCode}, nil
}

// File mutation serialization keyed by canonical path, so concurrent
// read-modify-write tools targeting the same file do not interleave.
var mutationLocks sync.Map

func lockForPath(path string) *sync.Mutex {
	m, _ := mutationLocks.LoadOrStore(path, &sync.Mutex{})
	return m.(*sync.Mutex)
}

func withFileMutation(path string, fn func() error) error {
	mu := lockForPath(path)
	mu.Lock()
	defer mu.Unlock()
	return fn()
}
