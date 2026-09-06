package ai

import "regexp"

// Context-overflow classification. The patterns mirror the TypeScript source;
// the non-overflow patterns are checked first (exclusions).

var nonOverflowPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^(Throttling error|Service unavailable):`),
	regexp.MustCompile(`(?i)rate limit`),
	regexp.MustCompile(`(?i)too many requests`),
}

var overflowPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)prompt is too long`),
	regexp.MustCompile(`(?i)request_too_large`),
	regexp.MustCompile(`(?i)input is too long for requested model`),
	regexp.MustCompile(`(?i)exceeds the context window`),
	regexp.MustCompile(`(?i)exceeds .* maximum context length`),
	regexp.MustCompile(`(?i)input token count.*exceeds the maximum`),
	regexp.MustCompile(`(?i)maximum prompt length is \d+`),
	regexp.MustCompile(`(?i)reduce the length of the messages`),
	regexp.MustCompile(`(?i)maximum context length is \d+ tokens`),
	regexp.MustCompile(`(?i)exceeds .* maximum allowed input length of [\d,]+ tokens?`),
	regexp.MustCompile(`(?i)input \(\d+ tokens\) is longer than the model's context length`),
	regexp.MustCompile(`(?i)exceeds the limit of \d+`),
	regexp.MustCompile(`(?i)exceeds the available context size`),
	regexp.MustCompile(`(?i)greater than the context length`),
	regexp.MustCompile(`(?i)context window exceeds limit`),
	regexp.MustCompile(`(?i)exceeded model token limit`),
	regexp.MustCompile(`(?i)too large for model with \d+ maximum context length`),
	regexp.MustCompile(`(?i)prompt has .* but the configured context size is .*`),
	regexp.MustCompile(`(?i)model_context_window_exceeded`),
	regexp.MustCompile(`(?i)prompt too long; exceeded (?:max )?context length`),
	regexp.MustCompile(`(?i)range of input length should be`),
	regexp.MustCompile(`(?i)context[ _]length[ _]exceeded`),
	regexp.MustCompile(`(?i)too many tokens`),
	regexp.MustCompile(`(?i)token limit exceeded`),
	regexp.MustCompile(`(?i)^4(?:00|13)\s*(?:status code)?\s*\(no body\)`),
}

func anyMatch(patterns []*regexp.Regexp, s string) bool {
	for _, p := range patterns {
		if p.MatchString(s) {
			return true
		}
	}
	return false
}

// IsContextOverflow classifies an assistant message as a context-overflow
// failure (via error patterns, silent input overflow, or length-stop overflow).
func IsContextOverflow(message *AssistantMessage, contextWindow int) bool {
	if message.StopReason == StopError && message.ErrorMessage != "" {
		if anyMatch(nonOverflowPatterns, message.ErrorMessage) {
			return false
		}
		return anyMatch(overflowPatterns, message.ErrorMessage)
	}
	if contextWindow > 0 && message.StopReason == StopEnd {
		inputTokens := message.Usage.Input + message.Usage.CacheRead
		if inputTokens > contextWindow {
			return true
		}
	}
	if contextWindow > 0 && message.StopReason == StopLength && message.Usage.Output == 0 &&
		message.Usage.Input+message.Usage.CacheRead >= int(float64(contextWindow)*0.99) {
		return true
	}
	return false
}

// IsRecoverableLength reports whether a length-stop output can be retried with
// a higher output budget.
func IsRecoverableLength(message *AssistantMessage, desiredMaxOutput int) bool {
	return message.StopReason == StopLength && desiredMaxOutput > 0 && message.Usage.Output < desiredMaxOutput
}
