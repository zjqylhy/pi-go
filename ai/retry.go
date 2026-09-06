package ai

import (
	"context"
	"regexp"
	"time"
)

// RetryPolicy controls automatic retry of transient provider errors.
type RetryPolicy struct {
	Enabled     bool
	MaxRetries  int
	BaseDelayMs int
}

// RetryCallbacks observes retry lifecycle events.
type RetryCallbacks struct {
	OnRetryScheduled    func(attempt, maxAttempts, delayMs int, errorMessage string)
	OnRetryAttemptStart func()
	OnRetryFinished     func(success bool, attempt int, finalError string)
}

var nonRetryableLimitPattern = regexp.MustCompile(`(?i)GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|available balance|insufficient_quota|out of budget|quota exceeded|billing`)

var retryableErrorPattern = regexp.MustCompile(`(?i)overloaded|rate.?limit|too many requests|429|500|502|503|504|524|service.?unavailable|server.?error|internal.?error|provider.?returned.?error|exceeded request buffer limit while retrying upstream|network.?error|connection.?error|connection.?refused|connection.?lost|other side closed|fetch failed|getaddrinfo|ENOTFOUND|EAI_AGAIN|upstream.?connect|reset before headers|socket hang up|socket connection was closed|timed? out|timeout|terminated|websocket.?closed|websocket.?error|ended without|stream ended before message_stop|stream ended before a terminal response event|http2 request did not get a response|retry delay|you can retry your request|try your request again|please retry your request|ResourceExhausted`)

func isRetryableAssistantError(message *AssistantMessage) bool {
	if message.StopReason != StopError || message.ErrorMessage == "" {
		return false
	}
	if nonRetryableLimitPattern.MatchString(message.ErrorMessage) {
		return false
	}
	return retryableErrorPattern.MatchString(message.ErrorMessage)
}

type retryRecord struct {
	attempt      int
	errorMessage string
}

// RetryAssistantCall re-invokes produce with an exponential backoff while the
// materialized assistant message reports a retryable error. Aborted calls are
// never retried; non-retryable errors fail fast.
func RetryAssistantCall(ctx context.Context, produce func(ctx context.Context) (*AssistantMessage, error), policy *RetryPolicy, callbacks *RetryCallbacks) (*AssistantMessage, error) {
	maxAttempts := 0
	baseDelayMs := 500
	if policy != nil {
		if policy.BaseDelayMs > 0 {
			baseDelayMs = policy.BaseDelayMs
		}
		if policy.Enabled {
			maxAttempts = policy.MaxRetries
		}
	}

	attempt := 0
	var lastRetry *retryRecord
	finished := func(success bool, finalError string) {
		if lastRetry != nil && callbacks != nil && callbacks.OnRetryFinished != nil {
			callbacks.OnRetryFinished(success, lastRetry.attempt, finalError)
		}
	}

	for {
		msg, err := produce(ctx)
		if err != nil {
			if attempt >= maxAttempts {
				finished(false, err.Error())
				return nil, err
			}
			attempt++
			lastRetry = &retryRecord{attempt: attempt, errorMessage: err.Error()}
			delayMs := baseDelayMs << (attempt - 1)
			if callbacks != nil && callbacks.OnRetryScheduled != nil {
				callbacks.OnRetryScheduled(attempt, maxAttempts, delayMs, err.Error())
			}
			if sleepCtx(ctx, time.Duration(delayMs)*time.Millisecond) != nil {
				finished(false, err.Error())
				return nil, ctx.Err()
			}
			if callbacks != nil && callbacks.OnRetryAttemptStart != nil {
				callbacks.OnRetryAttemptStart()
			}
			continue
		}

		if msg.StopReason == StopAborted {
			finished(false, "")
			return msg, nil
		}
		if msg.StopReason != StopError {
			finished(true, "")
			return msg, nil
		}
		if attempt >= maxAttempts || !isRetryableAssistantError(msg) {
			finished(false, msg.ErrorMessage)
			return msg, nil
		}
		attempt++
		lastRetry = &retryRecord{attempt: attempt, errorMessage: msg.ErrorMessage}
		delayMs := baseDelayMs << (attempt - 1)
		if callbacks != nil && callbacks.OnRetryScheduled != nil {
			callbacks.OnRetryScheduled(attempt, maxAttempts, delayMs, msg.ErrorMessage)
		}
		if err := sleepCtx(ctx, time.Duration(delayMs)*time.Millisecond); err != nil {
			finished(false, msg.ErrorMessage)
			cp := *msg
			cp.StopReason = StopAborted
			cp.ErrorMessage = ""
			return &cp, nil
		}
		if callbacks != nil && callbacks.OnRetryAttemptStart != nil {
			callbacks.OnRetryAttemptStart()
		}
	}
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	if ctx == nil {
		ctx = context.Background()
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
