package session

import (
	"context"
	"testing"

	"github.com/zjqylhy/pi-go/ai"
)

func userMsg(text string) ai.Message {
	return &ai.UserMessage{Role: "user", Content: []ai.ContentBlock{ai.TextBlock(text)}}
}

func TestSessionAppendAndScan(t *testing.T) {
	ctx := context.Background()
	repo := NewMemoryRepo()
	s, err := repo.Create(ctx)
	if err != nil {
		t.Fatal(err)
	}

	id1, _ := s.AppendMessage(userMsg("hello"), ctx)
	id2, _ := s.AppendMessage(userMsg("world"), ctx)
	if id1 == "" || id2 == "" {
		t.Fatalf("expected entry ids, got %q %q", id1, id2)
	}

	msgs, err := s.Messages(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}

	stats, _ := s.GetStats(ctx)
	if stats.MessageCount != 2 {
		t.Fatalf("expected messageCount 2, got %d", stats.MessageCount)
	}

	// parent chain: second entry's parent is the first.
	e1, _ := s.GetEntry(id1, ctx)
	e2, _ := s.GetEntry(id2, ctx)
	if e2.ParentID != id1 {
		t.Fatalf("expected parent %q, got %q", id1, e2.ParentID)
	}
	if e1.ParentID != "" {
		t.Fatalf("expected root parent, got %q", e1.ParentID)
	}
}

func TestSessionValuesAndLists(t *testing.T) {
	ctx := context.Background()
	repo := NewMemoryRepo()
	s, _ := repo.Create(ctx)

	addr := Value("pi.test", "k")
	if err := s.SetValue(addr, "v1", ctx); err != nil {
		t.Fatal(err)
	}
	got, _ := s.GetValue(addr, ctx)
	if got.Value != "v1" {
		t.Fatalf("expected v1, got %v", got.Value)
	}

	listAddr := List("pi.test.list", "")
	_ = s.AppendList(listAddr, "a", ctx)
	_ = s.AppendList(listAddr, "b", ctx)
	items, err := s.ReadList(listAddr, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 2 {
		t.Fatalf("expected 2 list elements, got %d", len(items))
	}
}

func TestSessionUsageAggregates(t *testing.T) {
	ctx := context.Background()
	repo := NewMemoryRepo()
	s, _ := repo.Create(ctx)

	before := ai.Usage{}
	_ = before
	err := s.Mutate(ctx, func(m *Mutator) error {
		_, err := m.Commit([]Write{
			InsertUsage(&UsageRow{ID: "u1", Usage: ai.Usage{Input: 10, Output: 5}}),
			InsertUsage(&UsageRow{ID: "u2", Usage: ai.Usage{Input: 20, Output: 7}}),
		}, ctx)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	stats, _ := s.GetStats(ctx)
	if stats.Usage.Input != 30 || stats.Usage.Output != 12 {
		t.Fatalf("expected aggregated usage input=30 output=12, got %+v", stats.Usage)
	}
}

func TestMemoryRepoListAndOpen(t *testing.T) {
	ctx := context.Background()
	repo := NewMemoryRepo()
	s, _ := repo.Create(ctx)
	meta := s.Metadata()

	opened, err := repo.Open(meta, ctx)
	if err != nil {
		t.Fatal(err)
	}
	_ = opened

	if len(repo.List(ctx)) != 1 {
		t.Fatalf("expected 1 session in list")
	}
	if err := repo.Delete(meta, ctx); err != nil {
		t.Fatal(err)
	}
	if len(repo.List(ctx)) != 0 {
		t.Fatalf("expected 0 sessions after delete")
	}
}
