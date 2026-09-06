package compaction

import (
	"context"
	"strings"
	"testing"

	"github.com/zjqylhy/pi-go/ai"
	"github.com/zjqylhy/pi-go/session"
)

func user(text string) ai.Message {
	return &ai.UserMessage{Role: "user", Content: []ai.ContentBlock{ai.TextBlock(text)}}
}

func assistant(text string) ai.Message {
	return &ai.AssistantMessage{Role: "assistant", Content: []ai.ContentBlock{ai.TextBlock(text)}, StopReason: ai.StopEnd, Usage: ai.Usage{Input: 10, Output: 5, TotalTokens: 15}}
}

func TestEstimateAndShouldCompact(t *testing.T) {
	messages := []ai.Message{user("hello world this is a test"), assistant("ok got it")}
	est := EstimateContextTokens(messages)
	if est.Tokens <= 0 {
		t.Fatalf("expected positive token estimate, got %d", est.Tokens)
	}
	if !ShouldCompact(100000, 100000, CompactionSettings{Enabled: true, ReserveTokens: 16384}) {
		t.Fatalf("expected shouldCompact true")
	}
	if ShouldCompact(100000, 200000, CompactionSettings{Enabled: true, ReserveTokens: 16384}) {
		t.Fatalf("expected shouldCompact false")
	}
}

func buildEntries(turns int) []*session.Entry {
	var entries []*session.Entry
	for i := 0; i < turns; i++ {
		entries = append(entries,
			&session.Entry{Type: session.EntryTypeMessage, ID: "u", Seq: i * 2, Message: user("turn")},
			&session.Entry{Type: session.EntryTypeMessage, ID: "a", Seq: i*2 + 1, Message: assistant("resp")},
		)
	}
	return entries
}

func TestFindCutPointKeepsRecent(t *testing.T) {
	entries := buildEntries(50)
	cut := FindCutPoint(entries, 0, len(entries), 40)
	if cut.FirstKeptEntryIndex <= 0 || cut.FirstKeptEntryIndex >= len(entries) {
		t.Fatalf("unexpected cut index %d", cut.FirstKeptEntryIndex)
	}
	// cut point must be a user message
	if messageRole(entries[cut.FirstKeptEntryIndex].Message) != "user" {
		t.Fatalf("cut point should be a user message")
	}
}

func TestPrepareCompaction(t *testing.T) {
	entries := buildEntries(100)
	prep := PrepareCompaction(entries, CompactionSettings{Enabled: true, ReserveTokens: 100, KeepRecentTokens: 20})
	if prep == nil {
		t.Fatal("expected preparation")
	}
	if len(prep.MessagesToSummarize) == 0 {
		t.Fatal("expected messages to summarize")
	}
	if len(prep.RetainedTail) == 0 {
		t.Fatal("expected retained tail")
	}
}

func TestSerializeConversation(t *testing.T) {
	out := SerializeConversation([]ai.Message{user("hi there"), assistant("hello!")})
	if !strings.Contains(out, "[User]: hi there") {
		t.Fatalf("missing user text: %s", out)
	}
	if !strings.Contains(out, "[Assistant]: hello!") {
		t.Fatalf("missing assistant text: %s", out)
	}
}

func fakeRequest(text string) SummaryRequest {
	return func(ctx *ai.Context, opts *ai.SimpleStreamOptions) (*ai.AssistantMessage, error) {
		return &ai.AssistantMessage{Content: []ai.ContentBlock{ai.TextBlock(text)}, StopReason: ai.StopEnd, Usage: ai.Usage{Input: 50, Output: 25}}, nil
	}
}

func TestGenerateSummary(t *testing.T) {
	model := &ai.Model{ID: "m", MaxTokens: 4096}
	text, usage, err := GenerateSummaryWithRequest(
		[]ai.Message{user("build a cli"), assistant("done")},
		model, 16000, "", "", "", fakeRequest("## Goal\nbuild a cli"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(text, "build a cli") {
		t.Fatalf("unexpected summary %q", text)
	}
	if usage.Output != 25 {
		t.Fatalf("expected usage output 25, got %d", usage.Output)
	}
}

func TestBuildContextWithCompaction(t *testing.T) {
	ctx := context.Background()
	repo := session.NewMemoryRepo()
	s, _ := repo.Create(ctx)

	_, _ = s.AppendMessage(user("hello"), ctx)
	_, _ = s.AppendMessage(assistant("world"), ctx)
	_, _ = s.AppendCompaction("summary here", 100, []ai.Message{assistant("recent")}, ctx)
	_, _ = s.AppendMessage(user("after"), ctx)

	entries, _ := s.Entries(ctx)
	msgs := BuildContext(entries)
	if len(msgs) != 3 {
		t.Fatalf("expected 3 projected messages, got %d", len(msgs))
	}
	if !strings.Contains(textOfBlocks(msgs[0].(*ai.UserMessage).Content), "summary here") {
		t.Fatalf("first projected message should be the compaction summary")
	}
}

func TestCompactWithRequest(t *testing.T) {
	model := &ai.Model{ID: "m", MaxTokens: 4096}
	prep := &Preparation{
		MessagesToSummarize: []ai.Message{user("summarize me")},
		RetainedTail:        []ai.Message{assistant("recent")},
		Settings:            DefaultCompactionSettings,
		ModifiedFiles:       []string{"main.go"},
	}
	res, err := CompactWithRequest(prep, model, "", "", fakeRequest("## Goal\nx"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(res.Summary, "## Goal") {
		t.Fatalf("unexpected summary %q", res.Summary)
	}
	if !strings.Contains(res.Summary, "main.go") {
		t.Fatalf("expected modified file in summary")
	}
	if len(res.RetainedTail) != 1 {
		t.Fatalf("expected 1 retained tail message")
	}
}
