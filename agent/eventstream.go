package agent

import "sync"

// EventStream is the agent loop's observable output: an ordered sequence of
// Event values that resolves to the produced agent messages. agent_end is the
// terminal event and also resolves Result.
type EventStream struct {
	mu        sync.Mutex
	cond      *sync.Cond
	queue     []Event
	done      bool
	finalOnce sync.Once
	final     []AgentMessage
	finalDone chan struct{}
}

func newEventStream() *EventStream {
	s := &EventStream{finalDone: make(chan struct{})}
	s.cond = sync.NewCond(&s.mu)
	return s
}

// Push delivers an event. agent_end marks the stream done and resolves Result.
func (s *EventStream) Push(e Event) {
	s.mu.Lock()
	if s.done {
		s.mu.Unlock()
		return
	}
	if e.Type == EventAgentEnd {
		s.done = true
		s.finalOnce.Do(func() {
			s.final = e.Messages
			close(s.finalDone)
		})
	}
	s.queue = append(s.queue, e)
	s.cond.Signal()
	s.mu.Unlock()
}

// End marks the stream done and resolves Result with messages.
func (s *EventStream) End(messages []AgentMessage) {
	s.mu.Lock()
	s.done = true
	s.finalOnce.Do(func() {
		s.final = messages
		close(s.finalDone)
	})
	s.cond.Broadcast()
	s.mu.Unlock()
}

// Next returns the next event in order. ok is false once the stream is done
// and fully drained.
func (s *EventStream) Next() (Event, bool) {
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

// Result blocks until the stream is complete and returns the produced messages.
func (s *EventStream) Result() []AgentMessage {
	<-s.finalDone
	return s.final
}

// Events returns a channel that emits every event then closes.
func (s *EventStream) Events() <-chan Event {
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
