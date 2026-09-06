package session

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/zjqylhy/pi-go/ai"
)

const tipValueKey = "pi.session.tip"

// Session is the high-level durable session facade over a Storage. It is safe
// for concurrent use; commits are serialized.
type Session struct {
	mu       sync.Mutex
	metadata SessionMetadata
	storage  Storage
	ids      *IdGenerator
}

func NewSession(metadata SessionMetadata, storage Storage) *Session {
	return &Session{metadata: metadata, storage: storage, ids: NewIdGenerator()}
}

func (s *Session) Metadata() SessionMetadata { return s.metadata }

func (s *Session) GetStats(ctx context.Context) (SessionStats, error) {
	return s.storage.GetStats(ctx)
}

func (s *Session) GetEntry(id string, ctx context.Context) (*Entry, error) {
	entries, err := s.storage.GetEntries([]string{id}, ctx)
	if err != nil {
		return nil, err
	}
	return entries[id], nil
}

func (s *Session) FindEntries(query EntryQuery, ctx context.Context) ([]*Entry, error) {
	return s.storage.ScanEntries(query, ctx)
}

// Messages returns the linear transcript (message entries in ascending order).
func (s *Session) Messages(ctx context.Context) ([]ai.Message, error) {
	entries, err := s.storage.ScanEntries(EntryQuery{Type: EntryTypeMessage, Order: "asc"}, ctx)
	if err != nil {
		return nil, err
	}
	out := make([]ai.Message, 0, len(entries))
	for _, e := range entries {
		if e.Message != nil {
			out = append(out, e.Message)
		}
	}
	return out, nil
}

// Entries returns all entries in ascending sequence order.
func (s *Session) Entries(ctx context.Context) ([]*Entry, error) {
	return s.storage.ScanEntries(EntryQuery{Order: "asc"}, ctx)
}

// AppendMessage appends a message entry as a child of the current tip and
// returns the new entry id.
func (s *Session) AppendMessage(msg ai.Message, ctx context.Context) (string, error) {
	return s.appendEntry(&Entry{Type: EntryTypeMessage, Message: msg}, ctx)
}

// AppendCompaction appends a compaction (summary) entry and returns its id.
func (s *Session) AppendCompaction(summary string, tokensBefore int, retainedTail []ai.Message, ctx context.Context) (string, error) {
	return s.appendEntry(&Entry{Type: EntryTypeCompaction, Summary: summary, TokensBefore: tokensBefore, RetainedTail: retainedTail}, ctx)
}

// appendEntry writes any entry type as a child of the current tip.
func (s *Session) appendEntry(entry *Entry, ctx context.Context) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	tip, _ := s.storage.GetValue(Value(tipValueKey, ""), ctx)
	parent := ""
	if tip != nil {
		if p, ok := tip.Value.(string); ok {
			parent = p
		}
	}
	entry.ID = s.ids.Next()
	entry.ParentID = parent
	_, err := s.storage.Commit([]Write{
		InsertEntry(entry),
		SetValue(Value(tipValueKey, ""), entry.ID),
	}, ctx)
	if err != nil {
		return "", err
	}
	return entry.ID, nil
}

func (s *Session) GetValue(addr Address, ctx context.Context) (*StoredValue, error) {
	return s.storage.GetValue(addr, ctx)
}

func (s *Session) ScanValues(addr Address, ctx context.Context) ([]*StoredValue, error) {
	return s.storage.ScanValues(addr, ctx)
}

func (s *Session) ReadList(addr Address, ctx context.Context) ([]ListElement, error) {
	return s.storage.ReadList(addr, ctx)
}

func (s *Session) SetValue(addr Address, v any, ctx context.Context) error {
	return s.singleCommit(SetValue(addr, v), ctx)
}

func (s *Session) DeleteValue(addr Address, ctx context.Context) error {
	return s.singleCommit(DeleteValue(addr), ctx)
}

func (s *Session) AppendList(addr Address, v any, ctx context.Context) error {
	return s.singleCommit(AppendList(addr, v), ctx)
}

func (s *Session) DeleteList(addr Address, ctx context.Context) error {
	return s.singleCommit(DeleteList(addr), ctx)
}

func (s *Session) singleCommit(w Write, ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.storage.Commit([]Write{w}, ctx)
	return err
}

// Mutator provides exactly one Commit within an exclusive mutation barrier.
type Mutator struct {
	s    *Session
	used bool
}

func (m *Mutator) Commit(writes []Write, ctx context.Context) (CommitResult, error) {
	if m.used {
		return CommitResult{}, fmt.Errorf("mutator already committed")
	}
	m.used = true
	return m.s.storage.Commit(writes, ctx)
}

// Mutate runs fn under the session's exclusive mutation barrier.
func (s *Session) Mutate(ctx context.Context, fn func(m *Mutator) error) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return fn(&Mutator{s: s})
}

// MemoryRepo is an in-memory SessionRepo.
type MemoryRepo struct {
	mu  sync.Mutex
	rec map[string]*repoRecord
}

type repoRecord struct {
	metadata SessionMetadata
	storage  *InMemoryStorage
}

func NewMemoryRepo() *MemoryRepo {
	return &MemoryRepo{rec: map[string]*repoRecord{}}
}

func (r *MemoryRepo) Create(ctx context.Context) (*Session, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	id := NewIdGenerator().Next()
	meta := SessionMetadata{ID: id, CreatedAt: time.Now().UnixMilli(), StorageVersion: 1}
	r.rec[id] = &repoRecord{metadata: meta, storage: NewInMemoryStorage()}
	return NewSession(meta, r.rec[id].storage), nil
}

func (r *MemoryRepo) Open(metadata SessionMetadata, ctx context.Context) (*Session, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	rec, ok := r.rec[metadata.ID]
	if !ok {
		return nil, fmt.Errorf("session not found: %s", metadata.ID)
	}
	return NewSession(rec.metadata, rec.storage), nil
}

func (r *MemoryRepo) List(ctx context.Context) []SessionMetadata {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]SessionMetadata, 0, len(r.rec))
	for _, rec := range r.rec {
		out = append(out, rec.metadata)
	}
	return out
}

func (r *MemoryRepo) Delete(metadata SessionMetadata, ctx context.Context) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.rec, metadata.ID)
	return nil
}
