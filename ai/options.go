package ai

import "context"

// ProviderRequestOptions are request-scoped controls shared by all provider
// calls. Cancellation is expressed through Ctx; the TypeScript AbortSignal
// maps to context.Context in idiomatic Go.
type ProviderRequestOptions struct {
	Ctx     context.Context `json:"-"`
	APIKey  string          `json:"-"`
	Env     ProviderEnv     `json:"-"`
	Headers ProviderHeaders `json:"-"`
	// TimeoutMs bounds the entire request; 0 means no explicit deadline.
	TimeoutMs       int `json:"-"`
	MaxRetries      int `json:"-"`
	MaxRetryDelayMs int `json:"-"`

	// OnPayload lets callers rewrite the outbound JSON payload before send.
	OnPayload func(payload any, model *Model) (any, error) `json:"-"`
	// OnResponse observes the HTTP status/headers of the wire response.
	OnResponse func(status int, headers map[string]string, model *Model) `json:"-"`
}

// StreamOptions are the options for a provider stream request.
type StreamOptions struct {
	ProviderRequestOptions

	Temperature    *float64       `json:"-"`
	SamplingParams map[string]any `json:"-"`
	MaxTokens      int            `json:"-"`
	Transport      Transport      `json:"-"`
	CacheRetention CacheRetention `json:"-"`
	SessionID      string         `json:"-"`
	Metadata       map[string]any `json:"-"`
}

// SimpleStreamOptions are the options for the simplified stream request where
// reasoning/tool choice are expressed uniformly.
type SimpleStreamOptions struct {
	StreamOptions

	ToolChoice      ToolChoice      `json:"-"`
	Reasoning       ThinkingLevel   `json:"-"`
	Deferred        any             `json:"-"`
	ThinkingBudgets ThinkingBudgets `json:"-"`
}

// DeferredFetchOptions bound a deferred-response poll.
type DeferredFetchOptions struct {
	ProviderRequestOptions
	// Wait is the max long-poll duration in milliseconds; 0 means one check.
	Wait int `json:"-"`
}

// DeferredCancelOptions cancels a deferred response.
type DeferredCancelOptions struct {
	ProviderRequestOptions
}
