package session

import (
	"context"
	"testing"

	"pi-go/ai"
)

func TestJSONLRoundTrip(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	repo := NewJSONLRepo(OSFileSystem{}, dir)
	s, meta, err := repo.Create("/work", ctx)
	if err != nil {
		t.Fatal(err)
	}

	_, _ = s.AppendMessage(&ai.UserMessage{Role: "user", Content: []ai.ContentBlock{ai.TextBlock("hello")}}, ctx)
	_, _ = s.AppendMessage(&ai.AssistantMessage{Role: "assistant", Content: []ai.ContentBlock{ai.TextBlock("hi there")}, StopReason: ai.StopEnd}, ctx)
	if err := s.SetValue(Value("pi.test", "k"), "persisted", ctx); err != nil {
		t.Fatal(err)
	}
	if err := s.AppendList(List("pi.test.list", ""), "a", ctx); err != nil {
		t.Fatal(err)
	}

	// Reopen from disk.
	repo2 := NewJSONLRepo(OSFileSystem{}, dir)
	s2, err := repo2.Open(meta, ctx)
	if err != nil {
		t.Fatal(err)
	}

	msgs, err := s2.Messages(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages after reopen, got %d", len(msgs))
	}

	stored, err := s2.GetValue(Value("pi.test", "k"), ctx)
	if err != nil || stored == nil {
		t.Fatalf("expected stored value, got %v err=%v", stored, err)
	}
	if stored.Value != "persisted" {
		t.Fatalf("expected persisted value, got %v", stored.Value)
	}

	items, err := s2.ReadList(List("pi.test.list", ""), ctx)
	if err != nil || len(items) != 1 || items[0].Value != "a" {
		t.Fatalf("expected 1 list element, got %v err=%v", items, err)
	}

	stats, _ := s2.GetStats(ctx)
	if stats.MessageCount != 2 {
		t.Fatalf("expected messageCount 2, got %d", stats.MessageCount)
	}
}

func TestJSONLListAndDelete(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	repo := NewJSONLRepo(OSFileSystem{}, dir)

	_, meta, err := repo.Create("/work", ctx)
	if err != nil {
		t.Fatal(err)
	}

	list, err := repo.List(ctx)
	if err != nil || len(list) != 1 {
		t.Fatalf("expected 1 session, got %d err=%v", len(list), err)
	}

	// Delete requires the session to first be closed; memory repo keeps it open.
	// For JSONL the metadata.Metadata().ID must match.
	if err := repo.Delete(meta, ctx); err == nil {
		t.Fatalf("expected delete of open session to error")
	}
}
