// Package session implements durable session storage for the agent: an
// append-only transcript of entries, scoped key/value and list state, usage
// accounting, and an in-memory repository. It mirrors the essential semantics
// of @earendil-works/pi-agent-core's session storage.
package session

import "github.com/zjqylhy/pi-go/ai"

// EntryType discriminates the entry union.
type EntryType = string

const (
	EntryTypeMessage       EntryType = "message"
	EntryTypeCompaction    EntryType = "compaction"
	EntryTypeBranchSummary EntryType = "branch_summary"
	EntryTypeCustom        EntryType = "custom"
)

// Entry is a single durable transcript entry. Payload fields are used
// depending on Type.
type Entry struct {
	ID         string
	ParentID   string // "" = branch root
	Seq        int
	Timestamp  int64
	Type       EntryType
	CustomType string

	// message
	Message   ai.Message
	Terminate bool

	// compaction / branch_summary
	Summary      string
	RetainedTail []ai.Message
	FromID       string
	TokensBefore int
	FromHook     bool

	Data any

	Usage *ai.Usage
}

// UsageRow is a single committed usage record.
type UsageRow struct {
	ID         string
	Seq        int
	Usage      ai.Usage
	EntryID    string
	Adjustment bool
	Details    any
}

// Address addresses a scoped value or list.
type Address struct {
	Namespace string
	Key       string
	List      bool
}

// Value returns a scalar value address.
func Value(namespace, key string) Address { return Address{Namespace: namespace, Key: key} }

// List returns a list address.
func List(namespace, key string) Address { return Address{Namespace: namespace, Key: key, List: true} }

// StoredValue is a committed scalar value with its sequence.
type StoredValue struct {
	Address Address
	Value   any
	Seq     int
}

// ListElement is a single committed list element.
type ListElement struct {
	Seq   int
	Value any
}

// SessionStats is the running session aggregate.
type SessionStats struct {
	MessageCount int
	Usage        ai.Usage
}

// CommitResult reports the outcome of a commit.
type CommitResult struct {
	FirstSeq  int
	Seqs      []int
	Timestamp int64
	Stats     SessionStats
}

// Write is a single mutation submitted in a commit. Kind/Op select the variant.
type Write struct {
	Kind string // "entry" | "usage" | "value" | "list"
	Op   string // for value/list: "set" | "delete" | "append"

	Entry *Entry    // kind "entry" (Seq/Timestamp assigned at commit)
	Row   *UsageRow // kind "usage" (Seq assigned at commit)

	Namespace string // value/list
	Key       string // value/list
	Value     any    // value set / list append
}

// Entry write constructors.
func InsertEntry(entry *Entry) Write { return Write{Kind: "entry", Entry: entry} }

func InsertUsage(row *UsageRow) Write { return Write{Kind: "usage", Row: row} }

func SetValue(addr Address, v any) Write {
	return Write{Kind: "value", Op: "set", Namespace: addr.Namespace, Key: addr.Key, Value: v}
}

func DeleteValue(addr Address) Write {
	return Write{Kind: "value", Op: "delete", Namespace: addr.Namespace, Key: addr.Key}
}

func AppendList(addr Address, v any) Write {
	return Write{Kind: "list", Op: "append", Namespace: addr.Namespace, Key: addr.Key, Value: v}
}

func DeleteList(addr Address) Write {
	return Write{Kind: "list", Op: "delete", Namespace: addr.Namespace, Key: addr.Key}
}

// SessionMetadata identifies a persisted session.
type SessionMetadata struct {
	ID              string
	CreatedAt       int64
	StorageVersion  int
	Cwd             string
	ParentSessionID string
}

// AddUsage sums two usage records field-by-field.
func AddUsage(a, b ai.Usage) ai.Usage {
	return ai.Usage{
		Input:        a.Input + b.Input,
		Output:       a.Output + b.Output,
		CacheRead:    a.CacheRead + b.CacheRead,
		CacheWrite:   a.CacheWrite + b.CacheWrite,
		CacheWrite1h: a.CacheWrite1h + b.CacheWrite1h,
		Reasoning:    a.Reasoning + b.Reasoning,
		TotalTokens:  a.TotalTokens + b.TotalTokens,
		Cost: ai.UsageCost{
			Input:      a.Cost.Input + b.Cost.Input,
			Output:     a.Cost.Output + b.Cost.Output,
			CacheRead:  a.Cost.CacheRead + b.Cost.CacheRead,
			CacheWrite: a.Cost.CacheWrite + b.Cost.CacheWrite,
			Total:      a.Cost.Total + b.Cost.Total,
		},
	}
}
