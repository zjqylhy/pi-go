package ai

// Api is the wire-protocol family a Model speaks. It mirrors the open
// string union in the TypeScript source and is extensible with arbitrary
// provider-specific values.
type Api = string

// Known API ids.
const (
	ApiOpenAICompletions     Api = "openai-completions"
	ApiMistralConversations  Api = "mistral-conversations"
	ApiOpenAIResponses       Api = "openai-responses"
	ApiAzureOpenAIResponses  Api = "azure-openai-responses"
	ApiOpenAICodexResponses  Api = "openai-codex-responses"
	ApiAnthropicMessages     Api = "anthropic-messages"
	ApiBedrockConverseStream Api = "bedrock-converse-stream"
	ApiGoogleGenerativeAI    Api = "google-generative-ai"
	ApiGoogleVertex          Api = "google-vertex"
	ApiPiMessages            Api = "pi-messages"
)

// Model describes a single concrete model offered by a provider.
type Model struct {
	ID   string `json:"id"`
	Name string `json:"name"`
	API  Api    `json:"api"`

	Provider string `json:"provider"`
	BaseURL  string `json:"baseUrl,omitempty"`

	Reasoning bool `json:"reasoning"`

	// ThinkingLevelMap maps pi thinking levels ("off", "minimal", "low",
	// "medium", "high", "xhigh", "max") to provider-native values. A key
	// present with a nil pointer means the level is explicitly unsupported
	// (JSON null); an absent key uses the provider default.
	ThinkingLevelMap map[string]*string `json:"thinkingLevelMap,omitempty"`

	Input []string `json:"input"`

	Cost          ModelCost `json:"cost"`
	ContextWindow int       `json:"contextWindow"`
	MaxTokens     int       `json:"maxTokens"`

	SamplingParams map[string]any    `json:"samplingParams,omitempty"`
	Headers        map[string]string `json:"headers,omitempty"`

	// Compat holds provider-specific capability flags (e.g. OpenAI/Anthropic
	// compat). Kept as an open map to mirror the per-api compat unions.
	Compat map[string]any `json:"compat,omitempty"`
}

// ModelCostRates are per-million-token prices in USD.
type ModelCostRates struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cacheRead"`
	CacheWrite float64 `json:"cacheWrite"`
}

// ModelCostTier is a volume-discount tier.
type ModelCostTier struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cacheRead"`
	CacheWrite float64 `json:"cacheWrite"`

	InputTokensAbove int `json:"inputTokensAbove"`
}

// ModelCost extends the base rates with optional tiers.
type ModelCost struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cacheRead"`
	CacheWrite float64 `json:"cacheWrite"`

	Tiers []ModelCostTier `json:"tiers,omitempty"`
}

// ContentBlock is a tagged union of assistant/user content blocks. The Type
// field discriminates among "text", "thinking", "image", and "toolCall".
type ContentBlock struct {
	Type string `json:"type"`

	// text block
	Text          string `json:"text,omitempty"`
	TextSignature string `json:"textSignature,omitempty"`

	// thinking block
	Thinking          string `json:"thinking,omitempty"`
	ThinkingSignature string `json:"thinkingSignature,omitempty"`
	Redacted          bool   `json:"redacted,omitempty"`

	// image block
	Data     string `json:"data,omitempty"`     // base64
	MimeType string `json:"mimeType,omitempty"` // e.g. image/jpeg

	// toolCall block
	ID               string         `json:"id,omitempty"`
	Name             string         `json:"name,omitempty"`
	Arguments        map[string]any `json:"arguments,omitempty"`
	ThoughtSignature string         `json:"thoughtSignature,omitempty"`
	Namespace        string         `json:"namespace,omitempty"`
}

// Content block type tags.
const (
	ContentTypeText     = "text"
	ContentTypeThinking = "thinking"
	ContentTypeImage    = "image"
	ContentTypeToolCall = "toolCall"
)

// Content block constructors.
func TextBlock(text string) ContentBlock { return ContentBlock{Type: ContentTypeText, Text: text} }
func ThinkingBlock(t string) ContentBlock {
	return ContentBlock{Type: ContentTypeThinking, Thinking: t}
}
func ImageBlock(data, mime string) ContentBlock {
	return ContentBlock{Type: ContentTypeImage, Data: data, MimeType: mime}
}
func ToolCallBlock(id, name string, args map[string]any) ContentBlock {
	return ContentBlock{Type: ContentTypeToolCall, ID: id, Name: name, Arguments: args}
}

// Message is the union of the three transcript message kinds.
type Message interface{ messageMarker() }

type UserMessage struct {
	Role      string         `json:"role"` // "user"
	Content   []ContentBlock `json:"content"`
	Timestamp int64          `json:"timestamp"`
}

type AssistantMessage struct {
	Role                  string          `json:"role"` // "assistant"
	Content               []ContentBlock  `json:"content"`
	API                   Api             `json:"api"`
	Provider              string          `json:"provider"`
	Model                 string          `json:"model"`
	ResponseModel         string          `json:"responseModel,omitempty"`
	ResponseID            string          `json:"responseId,omitempty"`
	ProviderThinkingLevel string          `json:"providerThinkingLevel,omitempty"`
	Diagnostics           []Diagnostic    `json:"diagnostics,omitempty"`
	Usage                 Usage           `json:"usage"`
	StopReason            StopReason      `json:"stopReason"`
	Deferred              *DeferredHandle `json:"deferred,omitempty"`
	ErrorMessage          string          `json:"errorMessage,omitempty"`
	RawStopReason         string          `json:"rawStopReason,omitempty"`
	EndTurn               bool            `json:"endTurn,omitempty"`
	Timestamp             int64           `json:"timestamp"`
}

type ToolResultMessage struct {
	Role           string         `json:"role"` // "toolResult"
	ToolCallID     string         `json:"toolCallId"`
	ToolName       string         `json:"toolName"`
	Content        []ContentBlock `json:"content"`
	Details        any            `json:"details,omitempty"`
	Usage          *Usage         `json:"usage,omitempty"`
	AddedToolNames []string       `json:"addedToolNames,omitempty"`
	IsError        bool           `json:"isError"`
	Timestamp      int64          `json:"timestamp"`
}

func (UserMessage) messageMarker()       {}
func (AssistantMessage) messageMarker()  {}
func (ToolResultMessage) messageMarker() {}

// Diagnostic is a structured, non-secret annotation attached to a message
// (e.g. rewrites, response failures).
type Diagnostic struct {
	Type    string         `json:"type"`
	Details map[string]any `json:"details,omitempty"`
}

// Usage aggregates token counters and computed cost.
type Usage struct {
	Input        int       `json:"input"`
	Output       int       `json:"output"`
	CacheRead    int       `json:"cacheRead"`
	CacheWrite   int       `json:"cacheWrite"`
	CacheWrite1h int       `json:"cacheWrite1h,omitempty"`
	Reasoning    int       `json:"reasoning,omitempty"`
	TotalTokens  int       `json:"totalTokens"`
	Cost         UsageCost `json:"cost"`
}

type UsageCost struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cacheRead"`
	CacheWrite float64 `json:"cacheWrite"`
	Total      float64 `json:"total"`
}

// StopReason is the terminal reason for an assistant turn.
type StopReason = string

const (
	StopPending  StopReason = "pending"
	StopEnd      StopReason = "stop"
	StopLength   StopReason = "length"
	StopToolUse  StopReason = "toolUse"
	StopError    StopReason = "error"
	StopAborted  StopReason = "aborted"
	StopDeferred StopReason = "deferred"
)

// DeferredHandle points at a durable, provider-side asynchronous response.
type DeferredHandle struct {
	Provider    string `json:"provider"`
	ModelID     string `json:"modelId"`
	API         string `json:"api"`
	ID          string `json:"id"`
	ExpiresAt   int64  `json:"expiresAt,omitempty"`
	PollAfterMs int64  `json:"pollAfterMs,omitempty"`
	Data        any    `json:"data,omitempty"`
}

// Tool describes a tool schema handed to a model.
type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
	// ConstrainedSampling is optional; kept as a raw value.
	ConstrainedSampling any `json:"constrainedSampling,omitempty"`
}

// Context is the input bundle sent to a model.
type Context struct {
	SystemPrompt string    `json:"systemPrompt,omitempty"`
	Messages     []Message `json:"messages"`
	Tools        []Tool    `json:"tools,omitempty"`
}

// ThinkingLevel is the extended reasoning-effort axis.
type ThinkingLevel = string

const (
	ThinkingOff     ThinkingLevel = "off"
	ThinkingMinimal ThinkingLevel = "minimal"
	ThinkingLow     ThinkingLevel = "low"
	ThinkingMedium  ThinkingLevel = "medium"
	ThinkingHigh    ThinkingLevel = "high"
	ThinkingXHigh   ThinkingLevel = "xhigh"
	ThinkingMax     ThinkingLevel = "max"
)

// ThinkingBudgets optionally pins per-level token budgets.
type ThinkingBudgets struct {
	Minimal int `json:"minimal,omitempty"`
	Low     int `json:"low,omitempty"`
	Medium  int `json:"medium,omitempty"`
	High    int `json:"high,omitempty"`
}

// ProviderHeaders maps header names to values; a nil value suppresses a
// provider/API default header.
type ProviderHeaders = map[string]*string

// ProviderEnv is a provider-scoped environment override.
type ProviderEnv = map[string]string

// Streaming event type tags (the public AssistantMessageEvent contract).
const (
	EventStart         = "start"
	EventTextStart     = "text_start"
	EventTextDelta     = "text_delta"
	EventTextEnd       = "text_end"
	EventThinkingStart = "thinking_start"
	EventThinkingDelta = "thinking_delta"
	EventThinkingEnd   = "thinking_end"
	EventToolCallStart = "toolcall_start"
	EventToolCallDelta = "toolcall_delta"
	EventToolCallEnd   = "toolcall_end"
	EventDone          = "done"
	EventError         = "error"
)

// Event is the public assistant-message streaming event, represented as a
// tagged struct. Partial is the shared live accumulator owned by the producer.
type Event struct {
	Type         string `json:"type"`
	ContentIndex int    `json:"contentIndex,omitempty"`

	// text/thinking deltas and ends
	Delta   string `json:"delta,omitempty"`
	Content string `json:"content,omitempty"`

	// toolcall_end
	ToolCall *ContentBlock `json:"toolCall,omitempty"`

	// terminal
	Reason  StopReason        `json:"reason,omitempty"`
	Message *AssistantMessage `json:"message,omitempty"`
	Error   *AssistantMessage `json:"error,omitempty"`

	// shared accumulator for every event
	Partial *AssistantMessage `json:"partial,omitempty"`
}

func isTerminalEvent(e Event) bool { return e.Type == EventDone || e.Type == EventError }

// Transport selects the wire transport.
type Transport = string

const (
	TransportSSE             Transport = "sse"
	TransportWebSocket       Transport = "websocket"
	TransportWebSocketCached Transport = "websocket-cached"
	TransportAuto            Transport = "auto"
)

// CacheRetention controls provider-side prompt caching.
type CacheRetention = string

const (
	CacheRetentionNone  CacheRetention = "none"
	CacheRetentionShort CacheRetention = "short"
	CacheRetentionLong  CacheRetention = "long"
)

// ToolChoice controls forced/auto tool selection.
type ToolChoice = string

const (
	ToolChoiceAuto ToolChoice = "auto"
	ToolChoiceNone ToolChoice = "none"
)
