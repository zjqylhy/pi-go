package ai

import (
	"sync"
	"time"
)

// AssistantMessageEventStream is the streaming contract returned by every
// provider. It delivers ordered Event values and resolves a final
// *AssistantMessage via Result. Terminal events are still delivered before the
// iterator reports the stream is finished.
//
// The implementation mirrors the TypeScript EventStream: an unbounded queue,
// a terminal flag, and a single-slot future for the final result.
type AssistantMessageEventStream struct {
	mu    sync.Mutex
	cond  *sync.Cond
	queue []Event
	done  bool

	finalOnce sync.Once
	final     *AssistantMessage
	finalDone chan struct{}
}

// NewAssistantMessageEventStream returns an empty, live stream.
func NewAssistantMessageEventStream() *AssistantMessageEventStream {
	s := &AssistantMessageEventStream{finalDone: make(chan struct{})}
	s.cond = sync.NewCond(&s.mu)
	return s
}

// Push delivers an event. Terminal events mark the stream done and resolve
// Result immediately; events after a terminal event are dropped.
func (s *AssistantMessageEventStream) Push(e Event) {
	s.mu.Lock()
	if s.done {
		s.mu.Unlock()
		return
	}
	if isTerminalEvent(e) {
		s.done = true
		s.finalOnce.Do(func() {
			s.final = extractResult(e)
			close(s.finalDone)
		})
	}
	s.queue = append(s.queue, e)
	s.cond.Signal()
	s.mu.Unlock()
}

// End marks the stream done. If result is non-nil it resolves Result (a no-op
// when a terminal event already resolved it).
func (s *AssistantMessageEventStream) End(result *AssistantMessage) {
	s.mu.Lock()
	s.done = true
	if result != nil {
		s.finalOnce.Do(func() {
			s.final = result
			close(s.finalDone)
		})
	}
	s.cond.Broadcast()
	s.mu.Unlock()
}

// Next returns the next event in order. ok is false once the stream is done
// and the queue is drained.
func (s *AssistantMessageEventStream) Next() (Event, bool) {
	s.mu.Lock()
	for len(s.queue) == 0 && !s.done {
		s.cond.Wait()
	}
	if len(s.queue) > 0 {
		e := s.queue[0]
		s.queue = s.queue[1:]
		s.mu.Unlock()
		return e, true
	}
	s.mu.Unlock()
	return Event{}, false
}

// Result blocks until the stream is terminal and returns the final message.
func (s *AssistantMessageEventStream) Result() *AssistantMessage {
	<-s.finalDone
	return s.final
}

// Events returns a channel that emits every event then closes. It supports
// `for e := range stream.Events() { ... }`.
func (s *AssistantMessageEventStream) Events() <-chan Event {
	ch := make(chan Event)
	go func() {
		for {
			e, ok := s.Next()
			if !ok {
				close(ch)
				return
			}
			ch <- e
		}
	}()
	return ch
}

// extractResult derives the final message from a terminal event.
func extractResult(e Event) *AssistantMessage {
	switch e.Type {
	case EventDone:
		return e.Message
	case EventError:
		return e.Error
	default:
		panic("ai: unexpected terminal event type " + e.Type)
	}
}

// setupErrorMessage builds a zeroed error message for a failed lazy setup.
func setupErrorMessage(model *Model, err error) *AssistantMessage {
	msg := "unknown error"
	if err != nil {
		msg = err.Error()
	}
	return &AssistantMessage{
		Role:         "assistant",
		Content:      []ContentBlock{},
		API:          model.API,
		Provider:     model.Provider,
		Model:        model.ID,
		Usage:        Usage{},
		StopReason:   StopError,
		ErrorMessage: msg,
		Timestamp:    time.Now().UnixMilli(),
	}
}

// forwardStream re-pumps an inner stream into an outer stream, then ends the
// outer with the inner's final result.
func forwardStream(target, source *AssistantMessageEventStream) {
	for {
		e, ok := source.Next()
		if !ok {
			break
		}
		target.Push(e)
	}
	target.End(source.Result())
}

// lazyStream returns a live stream immediately while setup runs behind it.
// A setup error surfaces as an `error` event rather than a panic/throw.
func lazyStream(model *Model, setup func() (*AssistantMessageEventStream, error)) *AssistantMessageEventStream {
	outer := NewAssistantMessageEventStream()
	go func() {
		inner, err := setup()
		if err != nil {
			msg := setupErrorMessage(model, err)
			outer.Push(Event{Type: EventError, Reason: StopError, Error: msg})
			outer.End(msg)
			return
		}
		forwardStream(outer, inner)
	}()
	return outer
}
