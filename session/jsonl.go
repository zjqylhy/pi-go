package session

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"pi-go/ai"
)

const (
	jsonlFormatVersion  = 4
	jsonlStorageVersion = 1
)

// committedWrite is the on-disk representation of a single committed write.
type committedWrite struct {
	Kind string `json:"kind"`
	Op   string `json:"op,omitempty"`
	Seq  int    `json:"seq"`

	ID         string          `json:"id,omitempty"`
	ParentID   *string         `json:"parentId,omitempty"`
	Timestamp  int64           `json:"timestamp,omitempty"`
	Type       string          `json:"type,omitempty"`
	CustomType string          `json:"customType,omitempty"`
	Message    json.RawMessage `json:"message,omitempty"`
	Terminate  bool            `json:"terminate,omitempty"`

	Summary      string            `json:"summary,omitempty"`
	RetainedTail []json.RawMessage `json:"retainedTail,omitempty"`
	FromID       string            `json:"fromId,omitempty"`
	TokensBefore int               `json:"tokensBefore,omitempty"`
	FromHook     bool              `json:"fromHook,omitempty"`
	Data         json.RawMessage   `json:"data,omitempty"`

	Usage      *ai.Usage       `json:"usage,omitempty"`
	EntryID    string          `json:"entryId,omitempty"`
	Adjustment bool            `json:"adjustment,omitempty"`
	Details    json.RawMessage `json:"details,omitempty"`

	Namespace string          `json:"namespace,omitempty"`
	Key       string          `json:"key,omitempty"`
	Value     json.RawMessage `json:"value,omitempty"`
}

func toCommittedWrite(w Write, seq int, ts int64) (committedWrite, error) {
	cw := committedWrite{Kind: w.Kind, Op: w.Op, Seq: seq}
	switch w.Kind {
	case "entry":
		e := w.Entry
		if e == nil {
			return cw, fmt.Errorf("nil entry")
		}
		cw.ID = e.ID
		cw.Type = e.Type
		cw.CustomType = e.CustomType
		cw.Timestamp = ts
		if e.ParentID != "" {
			p := e.ParentID
			cw.ParentID = &p
		}
		cw.Terminate = e.Terminate
		cw.Summary = e.Summary
		cw.FromID = e.FromID
		cw.TokensBefore = e.TokensBefore
		cw.FromHook = e.FromHook
		if e.Message != nil {
			raw, err := marshalMessage(e.Message)
			if err != nil {
				return cw, err
			}
			cw.Message = raw
		}
		if len(e.RetainedTail) > 0 {
			raws, err := marshalMessages(e.RetainedTail)
			if err != nil {
				return cw, err
			}
			cw.RetainedTail = raws
		}
		if e.Data != nil {
			b, _ := json.Marshal(e.Data)
			cw.Data = b
		}
	case "usage":
		r := w.Row
		if r == nil {
			return cw, fmt.Errorf("nil usage row")
		}
		cw.ID = r.ID
		u := r.Usage
		cw.Usage = &u
		cw.EntryID = r.EntryID
		cw.Adjustment = r.Adjustment
		if r.Details != nil {
			b, _ := json.Marshal(r.Details)
			cw.Details = b
		}
	case "value", "list":
		cw.Namespace = w.Namespace
		cw.Key = w.Key
		if w.Op == "set" || w.Op == "append" {
			b, _ := json.Marshal(w.Value)
			cw.Value = b
		}
	}
	return cw, nil
}

func (cw committedWrite) toEntry() *Entry {
	e := &Entry{
		ID: cw.ID, Seq: cw.Seq, Timestamp: cw.Timestamp, Type: cw.Type, CustomType: cw.CustomType,
		Terminate: cw.Terminate, Summary: cw.Summary, FromID: cw.FromID,
		TokensBefore: cw.TokensBefore, FromHook: cw.FromHook,
	}
	if cw.ParentID != nil {
		e.ParentID = *cw.ParentID
	}
	if len(cw.Message) > 0 {
		if m, err := unmarshalMessage(cw.Message); err == nil {
			e.Message = m
		}
	}
	if len(cw.RetainedTail) > 0 {
		e.RetainedTail, _ = unmarshalMessages(cw.RetainedTail)
	}
	if len(cw.Data) > 0 {
		var d any
		_ = json.Unmarshal(cw.Data, &d)
		e.Data = d
	}
	return e
}

func (cw committedWrite) toUsageRow() UsageRow {
	r := UsageRow{ID: cw.ID, Seq: cw.Seq, EntryID: cw.EntryID, Adjustment: cw.Adjustment}
	if cw.Usage != nil {
		r.Usage = *cw.Usage
	}
	if len(cw.Details) > 0 {
		var d any
		_ = json.Unmarshal(cw.Details, &d)
		r.Details = d
	}
	return r
}

// CommitPrepared assigns seq/timestamp, validates, applies, and returns the
// serializable committed rows plus the commit result (for file persistence).
func (s *InMemoryStorage) CommitPrepared(writes []Write, ts int64) ([]committedWrite, CommitResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	firstSeq := s.nextSeq
	rows := make([]committedWrite, len(writes))
	committedIDs := map[string]bool{}

	for i, w := range writes {
		seq := firstSeq + i
		cw, err := toCommittedWrite(w, seq, ts)
		if err != nil {
			return nil, CommitResult{}, err
		}
		switch w.Kind {
		case "entry":
			if committedIDs[w.Entry.ID] {
				return nil, CommitResult{}, fmt.Errorf("duplicate entry id: %s", w.Entry.ID)
			}
			if _, exists := s.entries[w.Entry.ID]; exists {
				return nil, CommitResult{}, fmt.Errorf("duplicate entry id: %s", w.Entry.ID)
			}
			if w.Entry.ParentID != "" {
				if _, ok := s.entries[w.Entry.ParentID]; !ok && !committedIDs[w.Entry.ParentID] {
					return nil, CommitResult{}, fmt.Errorf("missing parent entry: %s", w.Entry.ParentID)
				}
			}
			committedIDs[w.Entry.ID] = true
		case "usage":
			if committedIDs[w.Row.ID] {
				return nil, CommitResult{}, fmt.Errorf("duplicate usage id: %s", w.Row.ID)
			}
			if _, exists := s.usage[w.Row.ID]; exists {
				return nil, CommitResult{}, fmt.Errorf("duplicate usage id: %s", w.Row.ID)
			}
			committedIDs[w.Row.ID] = true
		}
		rows[i] = cw
		s.applyRow(cw)
	}

	s.nextSeq = firstSeq + len(writes)
	seqs := make([]int, len(rows))
	for i, r := range rows {
		seqs[i] = r.Seq
	}
	return rows, CommitResult{FirstSeq: firstSeq, Seqs: seqs, Timestamp: ts, Stats: s.stats}, nil
}

// applyRow applies a single materialized committed write to the state.
func (s *InMemoryStorage) applyRow(cw committedWrite) {
	switch cw.Kind {
	case "entry":
		e := cw.toEntry()
		s.entries[e.ID] = e
		s.entriesBySeq = append(s.entriesBySeq, e)
		if e.Type == EntryTypeMessage {
			s.stats.MessageCount++
		}
	case "usage":
		r := cw.toUsageRow()
		s.usage[r.ID] = r
		s.stats.Usage = AddUsage(s.stats.Usage, r.Usage)
	case "value":
		key := physicalKey(cw.Namespace, cw.Key)
		if cw.Op == "delete" {
			delete(s.scalarValues, key)
		} else {
			var v any
			_ = json.Unmarshal(cw.Value, &v)
			s.scalarValues[key] = &StoredValue{Address: Address{Namespace: cw.Namespace, Key: cw.Key}, Value: v, Seq: cw.Seq}
		}
	case "list":
		key := physicalKey(cw.Namespace, cw.Key)
		if cw.Op == "delete" {
			delete(s.listValues, key)
		} else {
			var v any
			_ = json.Unmarshal(cw.Value, &v)
			s.listValues[key] = append(s.listValues[key], ListElement{Seq: cw.Seq, Value: v})
		}
	}
}

// ReplayRows applies already-committed rows during open, advancing nextSeq.
func (s *InMemoryStorage) ReplayRows(rows []committedWrite) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, cw := range rows {
		s.applyRow(cw)
		if cw.Seq >= s.nextSeq {
			s.nextSeq = cw.Seq + 1
		}
	}
}

// JSONLHeader is the first line of a JSONL session file.
type JSONLHeader struct {
	V               int    `json:"v"`
	Kind            string `json:"kind"`
	ID              string `json:"id"`
	StorageVersion  int    `json:"storageVersion"`
	CreatedAt       int64  `json:"createdAt"`
	Cwd             string `json:"cwd"`
	ParentSessionID string `json:"parentSessionId,omitempty"`
	NextSeq         int    `json:"nextSeq,omitempty"`
}

func parseHeader(line string) (JSONLHeader, error) {
	var h JSONLHeader
	if err := json.Unmarshal([]byte(line), &h); err != nil {
		return h, fmt.Errorf("invalid JSONL header: %v", err)
	}
	if h.Kind != "header" || h.V != jsonlFormatVersion {
		return h, fmt.Errorf("unsupported JSONL header")
	}
	return h, nil
}

func serializeCommit(rows []committedWrite) (string, error) {
	if len(rows) == 1 {
		b, err := json.Marshal(rows[0])
		return string(b), err
	}
	b, err := json.Marshal(rows)
	return string(b), err
}

func parseCommit(line string) ([]committedWrite, error) {
	var arr []committedWrite
	if err := json.Unmarshal([]byte(line), &arr); err == nil {
		return arr, nil
	}
	var single committedWrite
	if err := json.Unmarshal([]byte(line), &single); err != nil {
		return nil, fmt.Errorf("invalid JSONL transaction: %v", err)
	}
	return []committedWrite{single}, nil
}

func splitCompleteLines(content string) []string {
	if content == "" {
		return nil
	}
	content = strings.TrimSuffix(content, "\n")
	return strings.Split(content, "\n")
}

// FileSystem is the minimal filesystem capability JSONL persistence needs.
type FileSystem interface {
	ReadTextFile(path string) (string, error)
	ReadTextLines(path string, maxLines int) ([]string, error)
	WriteFile(path, content string) error
	AppendFile(path, content string) error
	RenameFile(src, dst string) error
	Remove(path string) error
	ListDir(path string) ([]FSFileInfo, error)
	CreateDir(path string) error
	JoinPath(parts ...string) string
	AbsolutePath(path string) (string, error)
	Exists(path string) bool
}

// FSFileInfo is minimal filesystem metadata.
type FSFileInfo struct {
	Name    string
	Path    string
	Kind    string
	Size    int64
	MtimeMs int64
}

// OSFileSystem is the os-backed FileSystem.
type OSFileSystem struct{}

func (OSFileSystem) ReadTextFile(path string) (string, error) {
	b, err := os.ReadFile(path)
	return string(b), err
}

func (OSFileSystem) ReadTextLines(path string, maxLines int) ([]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	lines := splitCompleteLines(string(b))
	if maxLines > 0 && len(lines) > maxLines {
		lines = lines[:maxLines]
	}
	return lines, nil
}

func (OSFileSystem) WriteFile(path, content string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(content), 0o644)
}

func (OSFileSystem) AppendFile(path, content string) error {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(content)
	return err
}

func (OSFileSystem) RenameFile(src, dst string) error { return os.Rename(src, dst) }
func (OSFileSystem) Remove(path string) error         { return os.Remove(path) }

func (OSFileSystem) ListDir(path string) ([]FSFileInfo, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}
	out := make([]FSFileInfo, 0, len(entries))
	for _, e := range entries {
		info, err := e.Info()
		if err != nil {
			continue
		}
		kind := "file"
		if e.IsDir() {
			kind = "directory"
		}
		p := filepath.Join(path, e.Name())
		out = append(out, FSFileInfo{Name: e.Name(), Path: p, Kind: kind, Size: info.Size(), MtimeMs: info.ModTime().UnixMilli()})
	}
	return out, nil
}

func (OSFileSystem) CreateDir(path string) error { return os.MkdirAll(path, 0o755) }

func (OSFileSystem) JoinPath(parts ...string) string { return filepath.Join(parts...) }

func (OSFileSystem) AbsolutePath(path string) (string, error) { return filepath.Abs(path) }

func (OSFileSystem) Exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// JSONLStorage is a file-backed Storage that keeps an in-memory replica and
// appends committed transactions to a JSONL file.
type JSONLStorage struct {
	fs     FileSystem
	path   string
	mem    *InMemoryStorage
	header JSONLHeader
	mu     sync.Mutex
	closed bool
}

func createJSONLStorage(fs FileSystem, path string, header JSONLHeader) (*JSONLStorage, error) {
	line, err := json.Marshal(header)
	if err != nil {
		return nil, err
	}
	if err := fs.WriteFile(path, string(line)+"\n"); err != nil {
		return nil, err
	}
	return &JSONLStorage{fs: fs, path: path, mem: NewInMemoryStorage(), header: header}, nil
}

func openJSONLStorage(fs FileSystem, path string) (*JSONLStorage, error) {
	content, err := fs.ReadTextFile(path)
	if err != nil {
		return nil, err
	}
	lines := splitCompleteLines(content)
	if len(lines) == 0 || lines[0] == "" {
		return nil, fmt.Errorf("invalid JSONL storage: missing header")
	}
	header, err := parseHeader(lines[0])
	if err != nil {
		return nil, err
	}
	mem := NewInMemoryStorage()
	for i := 1; i < len(lines); i++ {
		if lines[i] == "" {
			continue
		}
		rows, err := parseCommit(lines[i])
		if err != nil {
			return nil, fmt.Errorf("invalid JSONL storage line %d: %v", i+1, err)
		}
		mem.ReplayRows(rows)
	}
	return &JSONLStorage{fs: fs, path: path, mem: mem, header: header}, nil
}

func (s *JSONLStorage) Commit(writes []Write, ctx context.Context) (CommitResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return CommitResult{}, fmt.Errorf("JSONLStorage is closed")
	}
	rows, res, err := s.mem.CommitPrepared(writes, time.Now().UnixMilli())
	if err != nil {
		return res, err
	}
	if len(rows) > 0 {
		line, err := serializeCommit(rows)
		if err != nil {
			return res, err
		}
		if err := s.fs.AppendFile(s.path, line+"\n"); err != nil {
			return res, err
		}
	}
	return res, nil
}

func (s *JSONLStorage) GetEntries(ids []string, ctx context.Context) (map[string]*Entry, error) {
	return s.mem.GetEntries(ids, ctx)
}
func (s *JSONLStorage) GetValue(addr Address, ctx context.Context) (*StoredValue, error) {
	return s.mem.GetValue(addr, ctx)
}
func (s *JSONLStorage) ScanValues(addr Address, ctx context.Context) ([]*StoredValue, error) {
	return s.mem.ScanValues(addr, ctx)
}
func (s *JSONLStorage) ReadList(addr Address, ctx context.Context) ([]ListElement, error) {
	return s.mem.ReadList(addr, ctx)
}
func (s *JSONLStorage) ScanEntries(query EntryQuery, ctx context.Context) ([]*Entry, error) {
	return s.mem.ScanEntries(query, ctx)
}
func (s *JSONLStorage) ScanUsage(fromSeq, toSeq int, order string, limit int, ctx context.Context) ([]UsageRow, error) {
	return s.mem.ScanUsage(fromSeq, toSeq, order, limit, ctx)
}
func (s *JSONLStorage) GetStats(ctx context.Context) (SessionStats, error) {
	return s.mem.GetStats(ctx)
}

// JSONLSessionMetadata extends SessionMetadata with the on-disk location.
type JSONLSessionMetadata struct {
	SessionMetadata
	Path       string
	ModifiedAt int64
}

// JSONLRepo is a file-backed SessionRepo using one file per session.
type JSONLRepo struct {
	fs   FileSystem
	root string
	mu   sync.Mutex
	open map[string]*Session
}

func NewJSONLRepo(fs FileSystem, root string) *JSONLRepo {
	return &JSONLRepo{fs: fs, root: root, open: map[string]*Session{}}
}

// Create opens a new persistent session.
func (r *JSONLRepo) Create(cwd string, ctx context.Context) (*Session, JSONLSessionMetadata, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.fs.CreateDir(r.root); err != nil {
		return nil, JSONLSessionMetadata{}, err
	}
	id := NewIdGenerator().Next()
	now := time.Now().UnixMilli()
	header := JSONLHeader{V: jsonlFormatVersion, Kind: "header", ID: id, StorageVersion: jsonlStorageVersion, CreatedAt: now, Cwd: cwd}
	path := r.fs.JoinPath(r.root, id+".jsonl")
	storage, err := createJSONLStorage(r.fs, path, header)
	if err != nil {
		return nil, JSONLSessionMetadata{}, err
	}
	meta := JSONLSessionMetadata{
		SessionMetadata: SessionMetadata{ID: id, CreatedAt: now, StorageVersion: jsonlStorageVersion, Cwd: cwd},
		Path:            path,
		ModifiedAt:      now,
	}
	s := NewSession(meta.SessionMetadata, storage)
	r.open[id] = s
	return s, meta, nil
}

// Open reopens a session from its on-disk metadata.
func (r *JSONLRepo) Open(meta JSONLSessionMetadata, ctx context.Context) (*Session, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if s, ok := r.open[meta.ID]; ok {
		return s, nil
	}
	storage, err := openJSONLStorage(r.fs, meta.Path)
	if err != nil {
		return nil, err
	}
	if storage.header.ID != meta.ID {
		return nil, fmt.Errorf("session identity does not match header: %s", meta.ID)
	}
	s := NewSession(meta.SessionMetadata, storage)
	r.open[meta.ID] = s
	return s, nil
}

// List returns metadata for all sessions in the root.
func (r *JSONLRepo) List(ctx context.Context) ([]JSONLSessionMetadata, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.fs.Exists(r.root) {
		return nil, nil
	}
	entries, err := r.fs.ListDir(r.root)
	if err != nil {
		return nil, err
	}
	var out []JSONLSessionMetadata
	for _, e := range entries {
		if e.Kind == "directory" || !strings.HasSuffix(e.Name, ".jsonl") {
			continue
		}
		lines, err := r.fs.ReadTextLines(e.Path, 1)
		if err != nil || len(lines) == 0 {
			continue
		}
		header, err := parseHeader(lines[0])
		if err != nil {
			continue
		}
		out = append(out, JSONLSessionMetadata{
			SessionMetadata: SessionMetadata{ID: header.ID, CreatedAt: header.CreatedAt, StorageVersion: header.StorageVersion, Cwd: header.Cwd, ParentSessionID: header.ParentSessionID},
			Path:            e.Path,
			ModifiedAt:      e.MtimeMs,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].CreatedAt > out[j].CreatedAt })
	return out, nil
}

// Delete removes a session file.
func (r *JSONLRepo) Delete(meta JSONLSessionMetadata, ctx context.Context) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.open[meta.ID]; ok {
		return fmt.Errorf("session is open: %s", meta.ID)
	}
	return r.fs.Remove(meta.Path)
}
