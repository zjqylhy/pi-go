package ai

import "sync"

// StreamSimpleHook wraps a StreamSimple call so plugins can observe or decorate
// requests without modifying call sites. Call next to proceed down the chain and
// return the stream it produces. Hooks run in registration order, outermost first.
//
// This is the global extension point used by opt-in plugins (e.g. llmlog), which
// register themselves via an init() side effect and are enabled by a blank import.
type StreamSimpleHook func(model *Model, context *Context, opts *SimpleStreamOptions, next func() *AssistantMessageEventStream) *AssistantMessageEventStream

var (
	hooksMu           sync.Mutex
	streamSimpleHooks []StreamSimpleHook
)

// RegisterStreamSimpleHook appends a hook to the global chain.
func RegisterStreamSimpleHook(h StreamSimpleHook) {
	hooksMu.Lock()
	defer hooksMu.Unlock()
	streamSimpleHooks = append(streamSimpleHooks, h)
}

func applyStreamSimpleHooks(model *Model, context *Context, opts *SimpleStreamOptions, raw func() *AssistantMessageEventStream) *AssistantMessageEventStream {
	hooksMu.Lock()
	chain := append([]StreamSimpleHook{}, streamSimpleHooks...)
	hooksMu.Unlock()

	next := raw
	for i := len(chain) - 1; i >= 0; i-- {
		h := chain[i]
		prev := next
		next = func() *AssistantMessageEventStream {
			return h(model, context, opts, prev)
		}
	}
	return next()
}