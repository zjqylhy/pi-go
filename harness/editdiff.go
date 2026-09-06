package harness

import (
	"fmt"
	"strings"
	"unicode"
)

// Edit is a single targeted text replacement.
type Edit struct {
	OldText string
	NewText string
}

func DetectLineEnding(content string) string {
	crlf := strings.Index(content, "\r\n")
	lf := strings.Index(content, "\n")
	if lf == -1 {
		return "\n"
	}
	if crlf == -1 {
		return "\n"
	}
	if crlf < lf {
		return "\r\n"
	}
	return "\n"
}

func NormalizeToLF(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	return strings.ReplaceAll(text, "\r", "\n")
}

func RestoreLineEndings(text, ending string) string {
	if ending == "\r\n" {
		return strings.ReplaceAll(text, "\n", "\r\n")
	}
	return text
}

func StripBom(content string) (bom, text string) {
	if strings.HasPrefix(content, "\uFEFF") {
		return "\uFEFF", content[1:]
	}
	return "", content
}

// normalizeForFuzzyMatch normalizes smart quotes, unicode dashes, and special
// spaces to ASCII, and trims trailing whitespace per line.
func normalizeForFuzzyMatch(text string) string {
	var b strings.Builder
	for _, r := range text {
		switch r {
		case '\u2018', '\u2019', '\u201A', '\u201B':
			b.WriteRune('\'')
		case '\u201C', '\u201D', '\u201E', '\u201F':
			b.WriteRune('"')
		case '\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015', '\u2212':
			b.WriteRune('-')
		case '\u00A0', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', '\u200A', '\u202F', '\u205F', '\u3000':
			b.WriteRune(' ')
		default:
			b.WriteRune(r)
		}
	}
	normalized := b.String()
	lines := strings.Split(normalized, "\n")
	for i, l := range lines {
		lines[i] = strings.TrimRightFunc(l, unicode.IsSpace)
	}
	return strings.Join(lines, "\n")
}

// AppliedEditsResult is the outcome of applying edits.
type AppliedEditsResult struct {
	BaseContent string
	NewContent  string
}

type matchedEdit struct {
	editIndex   int
	matchIndex  int
	matchLength int
	newText     string
}

// ApplyEditsToNormalizedContent applies exact-text replacements to LF-normalized
// content. Each oldText must match a unique, non-overlapping region.
func ApplyEditsToNormalizedContent(content string, edits []Edit, path string) (AppliedEditsResult, error) {
	normEdits := make([]Edit, len(edits))
	for i, e := range edits {
		normEdits[i] = Edit{OldText: NormalizeToLF(e.OldText), NewText: NormalizeToLF(e.NewText)}
	}
	for i, e := range normEdits {
		if e.OldText == "" {
			return AppliedEditsResult{}, newEditError(path, i, len(normEdits), "oldText must not be empty")
		}
	}

	// Determine replacement base via fuzzy match when any exact match fails.
	exactBase := content
	replacementBase := exactBase
	usedFuzzy := false
	for _, e := range normEdits {
		if strings.Index(exactBase, e.OldText) == -1 {
			usedFuzzy = true
			break
		}
	}
	if usedFuzzy {
		replacementBase = normalizeForFuzzyMatch(content)
	}

	matched := make([]matchedEdit, 0, len(normEdits))
	for i, e := range normEdits {
		oldText := e.OldText
		if usedFuzzy {
			oldText = normalizeForFuzzyMatch(e.OldText)
		}
		idx := strings.Index(replacementBase, oldText)
		if idx == -1 {
			return AppliedEditsResult{}, newEditError(path, i, len(normEdits), "could not find the exact text; oldText must match exactly including whitespace and newlines")
		}
		occurrences := strings.Count(replacementBase, oldText)
		if occurrences > 1 {
			return AppliedEditsResult{}, newEditError(path, i, len(normEdits), fmt.Sprintf("found %d occurrences; oldText must be unique", occurrences))
		}
		matched = append(matched, matchedEdit{
			editIndex: i, matchIndex: idx, matchLength: len(oldText), newText: e.NewText,
		})
	}

	// Sort by index and check for overlap.
	sorted := append([]matchedEdit{}, matched...)
	sortEdits(sorted)
	for i := 1; i < len(sorted); i++ {
		prev := sorted[i-1]
		cur := sorted[i]
		if prev.matchIndex+prev.matchLength > cur.matchIndex {
			return AppliedEditsResult{}, fmt.Errorf("edits[%d] and edits[%d] overlap in %s; merge them or target disjoint regions", prev.editIndex, cur.editIndex, path)
		}
	}

	baseContent := exactBase
	newContent := replacementBase
	// Apply in reverse order so offsets stay stable.
	for i := len(sorted) - 1; i >= 0; i-- {
		r := sorted[i]
		newContent = newContent[:r.matchIndex] + r.newText + newContent[r.matchIndex+r.matchLength:]
	}

	if baseContent == newContent {
		return AppliedEditsResult{}, newEditError(path, -1, len(normEdits), "no changes made; replacement produced identical content")
	}

	return AppliedEditsResult{BaseContent: baseContent, NewContent: newContent}, nil
}

func sortEdits(edits []matchedEdit) {
	for i := 1; i < len(edits); i++ {
		for j := i; j > 0 && edits[j].matchIndex < edits[j-1].matchIndex; j-- {
			edits[j], edits[j-1] = edits[j-1], edits[j]
		}
	}
}

func newEditError(path string, editIndex, total int, message string) error {
	if editIndex < 0 {
		if total == 1 {
			return fmt.Errorf("%s in %s", message, path)
		}
		return fmt.Errorf("%s in %s", message, path)
	}
	if total == 1 {
		return fmt.Errorf("%s in %s", message, path)
	}
	return fmt.Errorf("edits[%d]: %s in %s", editIndex, message, path)
}

// GenerateUnifiedPatch returns a basic unified diff between old and new content.
func GenerateUnifiedPatch(path, oldContent, newContent string) string {
	oldLines := strings.Split(oldContent, "\n")
	newLines := strings.Split(newContent, "\n")
	ops := diffLines(oldLines, newLines)

	var b strings.Builder
	fmt.Fprintf(&b, "--- %s\n+++ %s\n", path, path)
	for _, op := range ops {
		switch op.kind {
		case diffEqual:
			fmt.Fprintf(&b, " %s\n", op.line)
		case diffDelete:
			fmt.Fprintf(&b, "-%s\n", op.line)
		case diffInsert:
			fmt.Fprintf(&b, "+%s\n", op.line)
		}
	}
	return b.String()
}

type diffOpKind int

const (
	diffEqual diffOpKind = iota
	diffDelete
	diffInsert
)

type diffOp struct {
	kind diffOpKind
	line string
}

func diffLines(a, b []string) []diffOp {
	n, m := len(a), len(b)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, m+1)
	}
	for i := n - 1; i >= 0; i-- {
		for j := m - 1; j >= 0; j-- {
			if a[i] == b[j] {
				dp[i][j] = dp[i+1][j+1] + 1
			} else if dp[i+1][j] >= dp[i][j+1] {
				dp[i][j] = dp[i+1][j]
			} else {
				dp[i][j] = dp[i][j+1]
			}
		}
	}
	var ops []diffOp
	i, j := 0, 0
	for i < n && j < m {
		switch {
		case a[i] == b[j]:
			ops = append(ops, diffOp{diffEqual, a[i]})
			i++
			j++
		case dp[i+1][j] >= dp[i][j+1]:
			ops = append(ops, diffOp{diffDelete, a[i]})
			i++
		default:
			ops = append(ops, diffOp{diffInsert, b[j]})
			j++
		}
	}
	for i < n {
		ops = append(ops, diffOp{diffDelete, a[i]})
		i++
	}
	for j < m {
		ops = append(ops, diffOp{diffInsert, b[j]})
		j++
	}
	return ops
}

// GenerateDiffString returns a display diff with line numbers plus the first
// changed line number in the new content (1-indexed).
func GenerateDiffString(oldContent, newContent string) (string, int) {
	oldLines := strings.Split(oldContent, "\n")
	newLines := strings.Split(newContent, "\n")
	ops := diffLines(oldLines, newLines)

	var b strings.Builder
	oldLn, newLn := 1, 1
	firstChanged := 0
	width := len(fmt.Sprintf("%d", maxInt(len(oldLines), len(newLines))))
	for _, op := range ops {
		switch op.kind {
		case diffEqual:
			fmt.Fprintf(&b, "%*d %s\n", width, oldLn, op.line)
			oldLn++
			newLn++
		case diffDelete:
			if firstChanged == 0 {
				firstChanged = newLn
			}
			fmt.Fprintf(&b, "-%*d %s\n", width, oldLn, op.line)
			oldLn++
		case diffInsert:
			if firstChanged == 0 {
				firstChanged = newLn
			}
			fmt.Fprintf(&b, "+%*d %s\n", width, newLn, op.line)
			newLn++
		}
	}
	return strings.TrimRight(b.String(), "\n"), firstChanged
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
