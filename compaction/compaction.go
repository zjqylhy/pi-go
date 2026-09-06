// Package compaction implements context compaction: token estimation, cut-point
// selection, transcript serialization, and LLM-driven summary generation. The
// LLM boundary is injected as a callback so this package stays transport-free.
package compaction

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/zjqylhy/pi-go/ai"
	"github.com/zjqylhy/pi-go/session"
)

// CompactionSettings controls automatic compaction thresholds.
type CompactionSettings struct {
	Enabled          bool
	ReserveTokens    int
	KeepRecentTokens int
}

// DefaultCompactionSettings are the harness defaults.
var DefaultCompactionSettings = CompactionSettings{Enabled: true, ReserveTokens: 16384, KeepRecentTokens: 20000}

// CutPointResult is the compaction cut-point selection.
type CutPointResult struct {
	FirstKeptEntryIndex int
	TurnStartIndex      int
	IsSplitTurn         bool
}

// ContextUsageEstimate is an estimated context-token breakdown.
type ContextUsageEstimate struct {
	Tokens         int
	UsageTokens    int
	TrailingTokens int
	LastUsageIndex int // -1 = none
}

// CalculateContextTokens returns the total context tokens from usage.
func CalculateContextTokens(usage ai.Usage) int {
	if usage.TotalTokens > 0 {
		return usage.TotalTokens
	}
	return usage.Input + usage.Output + usage.CacheRead + usage.CacheWrite
}

func messageRole(m ai.Message) string {
	switch m.(type) {
	case *ai.UserMessage:
		return "user"
	case *ai.AssistantMessage:
		return "assistant"
	case *ai.ToolResultMessage:
		return "toolResult"
	default:
		return "custom"
	}
}

func contentChars(blocks []ai.ContentBlock) int {
	chars := 0
	for _, b := range blocks {
		switch b.Type {
		case ai.ContentTypeText:
			chars += len(b.Text)
		case ai.ContentTypeImage:
			chars += 4800
		}
	}
	return chars
}

func textOfBlocks(blocks []ai.ContentBlock) string {
	var b strings.Builder
	for _, block := range blocks {
		if block.Type == ai.ContentTypeText {
			b.WriteString(block.Text)
		}
	}
	return b.String()
}

func safeJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil || string(b) == "null" {
		return "undefined"
	}
	return string(b)
}

// EstimateTokens estimates tokens for one message via a char/4 heuristic.
func EstimateTokens(m ai.Message) int {
	chars := 0
	switch v := m.(type) {
	case *ai.UserMessage:
		chars = contentChars(v.Content)
	case *ai.AssistantMessage:
		for _, b := range v.Content {
			switch b.Type {
			case ai.ContentTypeText:
				chars += len(b.Text)
			case ai.ContentTypeThinking:
				chars += len(b.Thinking)
			case ai.ContentTypeToolCall:
				chars += len(b.Name) + len(safeJSON(b.Arguments))
			}
		}
	case *ai.ToolResultMessage:
		chars = contentChars(v.Content)
	}
	return (chars + 3) / 4
}

func getAssistantUsage(m ai.Message) *ai.Usage {
	am, ok := m.(*ai.AssistantMessage)
	if !ok {
		return nil
	}
	if am.StopReason == ai.StopAborted || am.StopReason == ai.StopError {
		return nil
	}
	if CalculateContextTokens(am.Usage) > 0 {
		u := am.Usage
		return &u
	}
	return nil
}

// GetLastAssistantUsage returns usage from the last valid assistant message.
func GetLastAssistantUsage(entries []*session.Entry) *ai.Usage {
	for i := len(entries) - 1; i >= 0; i-- {
		if entries[i].Type == session.EntryTypeMessage {
			if u := getAssistantUsage(entries[i].Message); u != nil {
				return u
			}
		}
	}
	return nil
}

// EstimateContextTokens estimates context tokens using provider usage when available.
func EstimateContextTokens(messages []ai.Message) ContextUsageEstimate {
	lastUsage, lastIndex := -1, -1
	for i := len(messages) - 1; i >= 0; i-- {
		if u := getAssistantUsage(messages[i]); u != nil {
			lastUsage = CalculateContextTokens(*u)
			lastIndex = i
			break
		}
	}
	if lastIndex < 0 {
		total := 0
		for _, m := range messages {
			total += EstimateTokens(m)
		}
		return ContextUsageEstimate{Tokens: total, TrailingTokens: total, LastUsageIndex: -1}
	}
	trailing := 0
	for i := lastIndex + 1; i < len(messages); i++ {
		trailing += EstimateTokens(messages[i])
	}
	return ContextUsageEstimate{Tokens: lastUsage + trailing, UsageTokens: lastUsage, TrailingTokens: trailing, LastUsageIndex: lastIndex}
}

// ShouldCompact reports whether context usage exceeds the threshold.
func ShouldCompact(contextTokens, contextWindow int, settings CompactionSettings) bool {
	if !settings.Enabled {
		return false
	}
	return contextTokens > contextWindow-settings.ReserveTokens
}

func messageForEntry(e *session.Entry, forCompaction bool) ai.Message {
	switch e.Type {
	case session.EntryTypeMessage:
		return e.Message
	case session.EntryTypeBranchSummary:
		return &ai.UserMessage{Role: "user", Content: []ai.ContentBlock{ai.TextBlock(branchPrefix + e.Summary + branchSuffix)}}
	case session.EntryTypeCompaction:
		if forCompaction {
			return nil
		}
		return &ai.UserMessage{Role: "user", Content: []ai.ContentBlock{ai.TextBlock(compactionPrefix + e.Summary + compactionSuffix)}}
	}
	return nil
}

const (
	branchPrefix     = "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n"
	branchSuffix     = "\n</summary>"
	compactionPrefix = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
	compactionSuffix = "\n</summary>"
)

func findValidCutPoints(entries []*session.Entry, start, end int) []int {
	var points []int
	for i := start; i < end; i++ {
		e := entries[i]
		switch e.Type {
		case session.EntryTypeMessage:
			role := messageRole(e.Message)
			if role == "user" || role == "assistant" {
				points = append(points, i)
			}
		case session.EntryTypeBranchSummary:
			points = append(points, i)
		}
	}
	return points
}

// FindTurnStartIndex returns the user-visible message that starts the turn
// containing entryIndex.
func FindTurnStartIndex(entries []*session.Entry, entryIndex, start int) int {
	for i := entryIndex; i >= start; i-- {
		e := entries[i]
		if e.Type == session.EntryTypeBranchSummary {
			return i
		}
		if e.Type == session.EntryTypeMessage && messageRole(e.Message) == "user" {
			return i
		}
	}
	return -1
}

// FindCutPoint selects the compaction cut point keeping ~keepRecentTokens.
func FindCutPoint(entries []*session.Entry, start, end, keepRecentTokens int) CutPointResult {
	points := findValidCutPoints(entries, start, end)
	if len(points) == 0 {
		return CutPointResult{FirstKeptEntryIndex: start, TurnStartIndex: -1, IsSplitTurn: false}
	}
	cutIndex := points[0]
	accumulated := 0
	for i := end - 1; i >= start; i-- {
		if entries[i].Type != session.EntryTypeMessage {
			continue
		}
		accumulated += EstimateTokens(entries[i].Message)
		if accumulated >= keepRecentTokens {
			for _, c := range points {
				if c >= i {
					cutIndex = c
					break
				}
			}
			break
		}
	}
	for cutIndex > start {
		prev := entries[cutIndex-1]
		if prev.Type == session.EntryTypeCompaction || prev.Type == session.EntryTypeMessage {
			break
		}
		cutIndex--
	}
	cut := entries[cutIndex]
	isUser := cut.Type == session.EntryTypeMessage && messageRole(cut.Message) == "user"
	turnStart := -1
	if !isUser {
		turnStart = FindTurnStartIndex(entries, cutIndex, start)
	}
	return CutPointResult{
		FirstKeptEntryIndex: cutIndex,
		TurnStartIndex:      turnStart,
		IsSplitTurn:         !isUser && turnStart != -1,
	}
}

// Preparation is the prepared input for a compaction run.
type Preparation struct {
	MessagesToSummarize []ai.Message
	TurnPrefixMessages  []ai.Message
	RetainedTail        []ai.Message
	IsSplitTurn         bool
	TokensBefore        int
	PreviousSummary     string
	ReadFiles           []string
	ModifiedFiles       []string
	Settings            CompactionSettings
}

func allMessages(entries []*session.Entry, forCompaction bool) []ai.Message {
	out := make([]ai.Message, 0, len(entries))
	for _, e := range entries {
		if m := messageForEntry(e, forCompaction); m != nil {
			out = append(out, m)
		}
	}
	return out
}

func virtualRetained(prev *session.Entry) []*session.Entry {
	out := make([]*session.Entry, 0, len(prev.RetainedTail))
	for i, m := range prev.RetainedTail {
		out = append(out, &session.Entry{
			Type:    session.EntryTypeMessage,
			ID:      fmt.Sprintf("%s:retained:%d", prev.ID, i),
			Seq:     prev.Seq,
			Message: m,
		})
	}
	return out
}

// PrepareCompaction prepares entries for compaction, or returns nil when not
// applicable.
func PrepareCompaction(entries []*session.Entry, settings CompactionSettings) *Preparation {
	if len(entries) == 0 || entries[len(entries)-1].Type == session.EntryTypeCompaction {
		return nil
	}
	prevIndex := -1
	for i := len(entries) - 1; i >= 0; i-- {
		if entries[i].Type == session.EntryTypeCompaction {
			prevIndex = i
			break
		}
	}
	var previousSummary string
	compactable := entries
	if prevIndex >= 0 {
		previousSummary = entries[prevIndex].Summary
		compactable = append(virtualRetained(entries[prevIndex]), entries[prevIndex+1:]...)
	}
	boundaryEnd := len(compactable)

	tokensBefore := EstimateContextTokens(allMessages(entries, false)).Tokens
	cut := FindCutPoint(compactable, 0, boundaryEnd, settings.KeepRecentTokens)
	historyEnd := cut.FirstKeptEntryIndex
	if cut.IsSplitTurn {
		historyEnd = cut.TurnStartIndex
	}

	var toSummarize, prefix, tail []ai.Message
	for i := 0; i < historyEnd; i++ {
		if m := messageForEntry(compactable[i], true); m != nil {
			toSummarize = append(toSummarize, m)
		}
	}
	if cut.IsSplitTurn {
		for i := cut.TurnStartIndex; i < cut.FirstKeptEntryIndex; i++ {
			if m := messageForEntry(compactable[i], true); m != nil {
				prefix = append(prefix, m)
			}
		}
	}
	for i := cut.FirstKeptEntryIndex; i < boundaryEnd; i++ {
		if m := messageForEntry(compactable[i], true); m != nil {
			tail = append(tail, m)
		}
	}

	read, modified := extractFileOps(append(append([]ai.Message{}, toSummarize...), prefix...))

	return &Preparation{
		MessagesToSummarize: toSummarize,
		TurnPrefixMessages:  prefix,
		RetainedTail:        tail,
		IsSplitTurn:         cut.IsSplitTurn,
		TokensBefore:        tokensBefore,
		PreviousSummary:     previousSummary,
		ReadFiles:           read,
		ModifiedFiles:       modified,
		Settings:            settings,
	}
}

func extractFileOps(msgs []ai.Message) (read, modified []string) {
	readSet := map[string]bool{}
	modSet := map[string]bool{}
	for _, m := range msgs {
		am, ok := m.(*ai.AssistantMessage)
		if !ok {
			continue
		}
		for _, b := range am.Content {
			if b.Type != ai.ContentTypeToolCall {
				continue
			}
			path, _ := b.Arguments["path"].(string)
			if path == "" {
				continue
			}
			switch b.Name {
			case "read":
				readSet[path] = true
			case "write", "edit":
				modSet[path] = true
			}
		}
	}
	var readOnly []string
	for p := range readSet {
		if !modSet[p] {
			readOnly = append(readOnly, p)
		}
	}
	sort.Strings(readOnly)
	var mod []string
	for p := range modSet {
		mod = append(mod, p)
	}
	sort.Strings(mod)
	return readOnly, mod
}

func formatFileOperations(readFiles, modifiedFiles []string) string {
	var sections []string
	if len(readFiles) > 0 {
		sections = append(sections, "<read-files>\n"+strings.Join(readFiles, "\n")+"\n</read-files>")
	}
	if len(modifiedFiles) > 0 {
		sections = append(sections, "<modified-files>\n"+strings.Join(modifiedFiles, "\n")+"\n</modified-files>")
	}
	if len(sections) == 0 {
		return ""
	}
	return "\n\n" + strings.Join(sections, "\n\n")
}

// SerializeConversation renders LLM messages as plain text for summarization.
func SerializeConversation(messages []ai.Message) string {
	const toolResultMaxChars = 2000
	var parts []string
	for _, m := range messages {
		switch messageRole(m) {
		case "user":
			u := m.(*ai.UserMessage)
			if c := textOfBlocks(u.Content); c != "" {
				parts = append(parts, "[User]: "+c)
			}
		case "assistant":
			a := m.(*ai.AssistantMessage)
			var thinking []string
			var toolCalls []string
			hasText := false
			for _, b := range a.Content {
				switch b.Type {
				case ai.ContentTypeThinking:
					thinking = append(thinking, b.Thinking)
				case ai.ContentTypeToolCall:
					var argParts []string
					for k, v := range b.Arguments {
						argParts = append(argParts, fmt.Sprintf("%s=%s", k, safeJSON(v)))
					}
					toolCalls = append(toolCalls, fmt.Sprintf("%s(%s)", b.Name, strings.Join(argParts, ", ")))
				case ai.ContentTypeText:
					hasText = true
				}
			}
			if len(thinking) > 0 {
				parts = append(parts, "[Assistant thinking]: "+strings.Join(thinking, "\n"))
			}
			if hasText {
				parts = append(parts, "[Assistant]: "+textOfBlocks(a.Content))
			}
			if len(toolCalls) > 0 {
				parts = append(parts, "[Assistant tool calls]: "+strings.Join(toolCalls, "; "))
			}
		case "toolResult":
			tr := m.(*ai.ToolResultMessage)
			if c := textOfBlocks(tr.Content); c != "" {
				parts = append(parts, "[Tool result]: "+truncateForSummary(c, toolResultMaxChars))
			}
		}
	}
	return strings.Join(parts, "\n\n")
}

func truncateForSummary(text string, maxChars int) string {
	if len(text) <= maxChars {
		return text
	}
	return text[:maxChars] + fmt.Sprintf("\n\n[... %d more characters truncated]", len(text)-maxChars)
}

// SummaryRequest is the LLM request boundary for summary generation.
type SummaryRequest func(ctx *ai.Context, opts *ai.SimpleStreamOptions) (*ai.AssistantMessage, error)

const summarizationSystemPrompt = `You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.`

const summarizationPrompt = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.`

const updateSummarizationPrompt = `The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages.`

const turnPrefixPrompt = `This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix.`

// GenerateSummaryWithRequest generates (or updates) a conversation summary.
func GenerateSummaryWithRequest(messages []ai.Message, model *ai.Model, reserveTokens int, customInstructions, previousSummary string, thinkingLevel ai.ThinkingLevel, request SummaryRequest) (string, ai.Usage, error) {
	maxTokens := int(float64(reserveTokens) * 0.8)
	if model.MaxTokens > 0 && maxTokens > model.MaxTokens {
		maxTokens = model.MaxTokens
	}

	basePrompt := summarizationPrompt
	if previousSummary != "" {
		basePrompt = updateSummarizationPrompt
	}
	if customInstructions != "" {
		basePrompt += "\n\nAdditional focus: " + customInstructions
	}

	conversation := SerializeConversation(messages)
	prompt := "<conversation>\n" + conversation + "\n</conversation>\n\n"
	if previousSummary != "" {
		prompt += "<previous-summary>\n" + previousSummary + "\n</previous-summary>\n\n"
	}
	prompt += basePrompt

	summarizationMessages := []ai.Message{
		&ai.UserMessage{Role: "user", Content: []ai.ContentBlock{ai.TextBlock(prompt)}},
	}

	opts := &ai.SimpleStreamOptions{}
	opts.MaxTokens = maxTokens
	if model.Reasoning && thinkingLevel != "" && thinkingLevel != ai.ThinkingOff {
		opts.Reasoning = thinkingLevel
	}

	response, err := request(&ai.Context{SystemPrompt: summarizationSystemPrompt, Messages: summarizationMessages}, opts)
	if err != nil {
		return "", ai.Usage{}, err
	}
	if response.StopReason == ai.StopAborted || response.StopReason == ai.StopError {
		return "", ai.Usage{}, fmt.Errorf("summarization failed: %s", response.ErrorMessage)
	}
	return textOfBlocks(response.Content), response.Usage, nil
}

// CompactResult is the generated compaction data.
type CompactResult struct {
	Summary       string
	TokensBefore  int
	Usage         ai.Usage
	RetainedTail  []ai.Message
	ReadFiles     []string
	ModifiedFiles []string
}

// CompactWithRequest generates compaction data from a prepared history.
func CompactWithRequest(prep *Preparation, model *ai.Model, customInstructions string, thinkingLevel ai.ThinkingLevel, request SummaryRequest) (CompactResult, error) {
	var summary string
	var usage ai.Usage

	if prep.IsSplitTurn && len(prep.TurnPrefixMessages) > 0 {
		historyText := "No prior history."
		if len(prep.MessagesToSummarize) > 0 {
			s, u, err := GenerateSummaryWithRequest(prep.MessagesToSummarize, model, prep.Settings.ReserveTokens, customInstructions, prep.PreviousSummary, thinkingLevel, request)
			if err != nil {
				return CompactResult{}, err
			}
			historyText = s
			usage = u
		}
		prefixText, pu, err := generateTurnPrefixSummary(prep.TurnPrefixMessages, model, prep.Settings.ReserveTokens, thinkingLevel, request)
		if err != nil {
			return CompactResult{}, err
		}
		usage = aiUsageAdd(usage, pu)
		summary = historyText + "\n\n---\n\n**Turn Context (split turn):**\n\n" + prefixText
	} else {
		s, u, err := GenerateSummaryWithRequest(prep.MessagesToSummarize, model, prep.Settings.ReserveTokens, customInstructions, prep.PreviousSummary, thinkingLevel, request)
		if err != nil {
			return CompactResult{}, err
		}
		summary = s
		usage = u
	}

	summary += formatFileOperations(prep.ReadFiles, prep.ModifiedFiles)

	return CompactResult{
		Summary:       summary,
		TokensBefore:  prep.TokensBefore,
		Usage:         usage,
		RetainedTail:  prep.RetainedTail,
		ReadFiles:     prep.ReadFiles,
		ModifiedFiles: prep.ModifiedFiles,
	}, nil
}

func generateTurnPrefixSummary(messages []ai.Message, model *ai.Model, reserveTokens int, thinkingLevel ai.ThinkingLevel, request SummaryRequest) (string, ai.Usage, error) {
	maxTokens := int(float64(reserveTokens) * 0.5)
	if model.MaxTokens > 0 && maxTokens > model.MaxTokens {
		maxTokens = model.MaxTokens
	}
	conversation := SerializeConversation(messages)
	prompt := "<conversation>\n" + conversation + "\n</conversation>\n\n" + turnPrefixPrompt
	summarizationMessages := []ai.Message{
		&ai.UserMessage{Role: "user", Content: []ai.ContentBlock{ai.TextBlock(prompt)}},
	}
	opts := &ai.SimpleStreamOptions{}
	opts.MaxTokens = maxTokens
	if model.Reasoning && thinkingLevel != "" && thinkingLevel != ai.ThinkingOff {
		opts.Reasoning = thinkingLevel
	}
	response, err := request(&ai.Context{SystemPrompt: summarizationSystemPrompt, Messages: summarizationMessages}, opts)
	if err != nil {
		return "", ai.Usage{}, err
	}
	if response.StopReason == ai.StopAborted || response.StopReason == ai.StopError {
		return "", ai.Usage{}, fmt.Errorf("turn prefix summarization failed: %s", response.ErrorMessage)
	}
	return textOfBlocks(response.Content), response.Usage, nil
}

func aiUsageAdd(a, b ai.Usage) ai.Usage {
	return session.AddUsage(a, b)
}

// CompactionSummaryMessage renders a compaction summary as a model message.
func CompactionSummaryMessage(summary string) ai.Message {
	return &ai.UserMessage{Role: "user", Content: []ai.ContentBlock{ai.TextBlock(compactionPrefix + summary + compactionSuffix)}}
}

// BranchSummaryMessage renders a branch summary as a model message.
func BranchSummaryMessage(summary string) ai.Message {
	return &ai.UserMessage{Role: "user", Content: []ai.ContentBlock{ai.TextBlock(branchPrefix + summary + branchSuffix)}}
}

// BuildContext projects session entries into the model-visible message list.
// A compaction entry resets the context to [summary]+[retained tail].
func BuildContext(entries []*session.Entry) []ai.Message {
	var ctx []ai.Message
	for _, e := range entries {
		switch e.Type {
		case session.EntryTypeMessage:
			if e.Message != nil {
				ctx = append(ctx, e.Message)
			}
		case session.EntryTypeCompaction:
			ctx = []ai.Message{CompactionSummaryMessage(e.Summary)}
			ctx = append(ctx, e.RetainedTail...)
		case session.EntryTypeBranchSummary:
			ctx = append(ctx, BranchSummaryMessage(e.Summary))
		}
	}
	return ctx
}
