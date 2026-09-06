package session

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// EntryQuery filters a scan over entries.
type EntryQuery struct {
	Type       string
	CustomType string
	FromSeq    int
	ToSeq      int
	Order      string // "asc" | "desc"
	Limit      int    // 0 = unlimited
}

// Storage is the durable backing for a Session.
type Storage interface {
	Commit(writes []Write, ctx context.Context) (CommitResult, error)
	GetEntries(ids []string, ctx context.Context) (map[string]*Entry, error)
	GetValue(addr Address, ctx context.Context) (*StoredValue, error)
	ScanValues(addr Address, ctx context.Context) ([]*StoredValue, error)
	ReadList(addr Address, ctx context.Context) ([]ListElement, error)
	ScanEntries(query EntryQuery, ctx context.Context) ([]*Entry, error)
	ScanUsage(fromSeq, toSeq int, order string, limit int, ctx context.Context) ([]UsageRow, error)
	GetStats(ctx context.Context) (SessionStats, error)
}

func physicalKey(namespace, key string) string { return namespace + "\x00" + key }

// InMemoryStorage is a fully materialized, non-persistent Storage.
type InMemoryStorage struct {
	mu           sync.Mutex
	entries      map[string]*Entry
	entriesBySeq []*Entry
	scalarValues map[string]*StoredValue
	listValues   map[string][]ListElement
	usage        map[string]UsageRow
	stats        SessionStats
	nextSeq      int
}

func NewInMemoryStorage() *InMemoryStorage {
	return &InMemoryStorage{
		entries:      map[string]*Entry{},
		scalarValues: map[string]*StoredValue{},
		listValues:   map[string][]ListElement{},
		usage:        map[string]UsageRow{},
		nextSeq:      1,
	}
}

func (s *InMemoryStorage) Commit(writes []Write, ctx context.Context) (CommitResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	ts := time.Now().UnixMilli()
	firstSeq := s.nextSeq
	seqs := make([]int, len(writes))
	committedIDs := map[string]bool{}

	for i, w := range writes {
		seq := firstSeq + i
		seqs[i] = seq

		switch w.Kind {
		case "entry":
			e := *w.Entry
			e.Seq = seq
			e.Timestamp = ts
			if committedIDs[e.ID] {
				return CommitResult{}, fmt.Errorf("duplicate entry id: %s", e.ID)
			}
			if _, exists := s.entries[e.ID]; exists {
				return CommitResult{}, fmt.Errorf("duplicate entry id: %s", e.ID)
			}
			if e.ParentID != "" && e.ParentID != e.ID {
				if _, ok := s.entries[e.ParentID]; !ok && !committedIDs[e.ParentID] {
					return CommitResult{}, fmt.Errorf("missing parent entry: %s", e.ParentID)
				}
			}
			s.entries[e.ID] = &e
			s.entriesBySeq = append(s.entriesBySeq, &e)
			if e.Type == EntryTypeMessage {
				s.stats.MessageCount++
			}
			committedIDs[e.ID] = true

		case "usage":
			row := *w.Row
			row.Seq = seq
			if committedIDs[row.ID] {
				return CommitResult{}, fmt.Errorf("duplicate usage id: %s", row.ID)
			}
			if _, exists := s.usage[row.ID]; exists {
				return CommitResult{}, fmt.Errorf("duplicate usage id: %s", row.ID)
			}
			s.usage[row.ID] = row
			s.stats.Usage = AddUsage(s.stats.Usage, row.Usage)
			committedIDs[row.ID] = true

		case "value":
			key := physicalKey(w.Namespace, w.Key)
			if w.Op == "delete" {
				delete(s.scalarValues, key)
			} else {
				s.scalarValues[key] = &StoredValue{Address: Address{Namespace: w.Namespace, Key: w.Key}, Value: w.Value, Seq: seq}
			}

		case "list":
			key := physicalKey(w.Namespace, w.Key)
			if w.Op == "delete" {
				delete(s.listValues, key)
			} else {
				s.listValues[key] = append(s.listValues[key], ListElement{Seq: seq, Value: w.Value})
			}
		}
	}

	s.nextSeq = firstSeq + len(writes)
	return CommitResult{FirstSeq: firstSeq, Seqs: seqs, Timestamp: ts, Stats: s.stats}, nil
}

func (s *InMemoryStorage) GetEntries(ids []string, ctx context.Context) (map[string]*Entry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := map[string]*Entry{}
	for _, id := range ids {
		if e, ok := s.entries[id]; ok {
			out[id] = e
		}
	}
	return out, nil
}

func (s *InMemoryStorage) GetValue(addr Address, ctx context.Context) (*StoredValue, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.scalarValues[physicalKey(addr.Namespace, addr.Key)], nil
}

func (s *InMemoryStorage) ScanValues(addr Address, ctx context.Context) ([]*StoredValue, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []*StoredValue
	for _, v := range s.scalarValues {
		if v.Address.Namespace == addr.Namespace && strings.HasPrefix(v.Address.Key, addr.Key) {
			out = append(out, v)
		}
	}
	sortStoredValues(out)
	return out, nil
}

func sortStoredValues(values []*StoredValue) {
	for i := 1; i < len(values); i++ {
		for j := i; j > 0 && values[j].Address.Key < values[j-1].Address.Key; j-- {
			values[j], values[j-1] = values[j-1], values[j]
		}
	}
}

func (s *InMemoryStorage) ReadList(addr Address, ctx context.Context) ([]ListElement, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]ListElement{}, s.listValues[physicalKey(addr.Namespace, addr.Key)]...), nil
}

func (s *InMemoryStorage) ScanEntries(query EntryQuery, ctx context.Context) ([]*Entry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := s.filterEntries(query)
	if query.Limit > 0 && len(out) > query.Limit {
		out = out[:query.Limit]
	}
	return out, nil
}

func (s *InMemoryStorage) filterEntries(query EntryQuery) []*Entry {
	var out []*Entry
	for _, e := range s.entriesBySeq {
		if query.Type != "" && e.Type != query.Type {
			continue
		}
		if query.CustomType != "" && e.CustomType != query.CustomType {
			continue
		}
		if query.FromSeq != 0 && e.Seq < query.FromSeq {
			continue
		}
		if query.ToSeq != 0 && e.Seq > query.ToSeq {
			continue
		}
		out = append(out, e)
	}
	if query.Order == "desc" {
		for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
			out[i], out[j] = out[j], out[i]
		}
	}
	return out
}

func (s *InMemoryStorage) ScanUsage(fromSeq, toSeq int, order string, limit int, ctx context.Context) ([]UsageRow, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []UsageRow
	for _, r := range s.usage {
		if fromSeq != 0 && r.Seq < fromSeq {
			continue
		}
		if toSeq != 0 && r.Seq > toSeq {
			continue
		}
		out = append(out, r)
	}
	sortUsageRows(out, order == "desc")
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

func sortUsageRows(rows []UsageRow, desc bool) {
	for i := 1; i < len(rows); i++ {
		for j := i; j > 0 && ((desc && rows[j].Seq > rows[j-1].Seq) || (!desc && rows[j].Seq < rows[j-1].Seq)); j-- {
			rows[j], rows[j-1] = rows[j-1], rows[j]
		}
	}
}

func (s *InMemoryStorage) GetStats(ctx context.Context) (SessionStats, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stats, nil
}
