package ai

// Built-in provider factories and a representative model catalog. The full
// generated catalog (40+ providers, exact model versions, live pricing) is out
// of MVP scope; wiring is correct and catalog entries are accurate enough for
// development and smoke-testing.

func mkModel(id, name string, api Api, provider string, in, out float64, ctxWin, maxTok int, reasoning bool) *Model {
	return &Model{
		ID:            id,
		Name:          name,
		API:           api,
		Provider:      provider,
		Input:         []string{"text"},
		Cost:          ModelCost{Input: in, Output: out, CacheRead: in * 0.1, CacheWrite: in * 1.25},
		ContextWindow: ctxWin,
		MaxTokens:     maxTok,
		Reasoning:     reasoning,
	}
}

// Anthropic.
func anthropicModels() []*Model {
	return []*Model{
		mkModel("claude-sonnet-4-5", "Claude Sonnet 4.5", ApiAnthropicMessages, "anthropic", 3, 15, 200000, 64000, true),
		mkModel("claude-opus-4-6", "Claude Opus 4.6", ApiAnthropicMessages, "anthropic", 15, 75, 200000, 64000, true),
		mkModel("claude-haiku-4-5", "Claude Haiku 4.5", ApiAnthropicMessages, "anthropic", 1, 5, 200000, 32000, false),
	}
}

func anthropicProvider() Provider {
	return CreateProvider(CreateProviderOptions{
		ID:      "anthropic",
		Name:    "Anthropic",
		BaseURL: "https://api.anthropic.com",
		Auth:    ProviderAuth{APIKey: NewEnvAPIKeyAuth("Anthropic API key", "ANTHROPIC_API_KEY", "ANTHROPIC_OAUTH_TOKEN")},
		Models:  anthropicModels(),
		API:     anthropicAPI{},
	})
}

// OpenAI (Responses API).
func openaiModels() []*Model {
	return []*Model{
		mkModel("gpt-5.2", "GPT-5.2", ApiOpenAIResponses, "openai", 1.75, 14, 400000, 128000, true),
		mkModel("gpt-5-mini", "GPT-5 mini", ApiOpenAIResponses, "openai", 0.25, 2, 400000, 128000, true),
		mkModel("gpt-5-nano", "GPT-5 nano", ApiOpenAIResponses, "openai", 0.05, 0.4, 400000, 128000, false),
	}
}

func openaiProvider() Provider {
	return CreateProvider(CreateProviderOptions{
		ID:      "openai",
		Name:    "OpenAI",
		BaseURL: "https://api.openai.com/v1",
		Auth:    ProviderAuth{APIKey: NewEnvAPIKeyAuth("OpenAI API key", "OPENAI_API_KEY")},
		Models:  openaiModels(),
		API:     openAIResponsesAPI{},
	})
}

// Google (Generative Language API).
func googleModels() []*Model {
	return []*Model{
		mkModel("gemini-3-pro-preview", "Gemini 3 Pro", ApiGoogleGenerativeAI, "google", 2, 12, 1000000, 65536, true),
		mkModel("gemini-2.5-flash", "Gemini 2.5 Flash", ApiGoogleGenerativeAI, "google", 0.3, 2.5, 1000000, 65536, false),
	}
}

func googleProvider() Provider {
	return CreateProvider(CreateProviderOptions{
		ID:      "google",
		Name:    "Google",
		BaseURL: "https://generativelanguage.googleapis.com/v1beta",
		Auth:    ProviderAuth{APIKey: NewEnvAPIKeyAuth("Gemini API key", "GEMINI_API_KEY")},
		Models:  googleModels(),
		API:     googleGenerativeAIAPI{},
	})
}

// OpenAI-compatible Chat Completions providers.
func openAICompatibleProvider(id, name, baseURL string, envVar string, models []*Model) Provider {
	return CreateProvider(CreateProviderOptions{
		ID:      id,
		Name:    name,
		BaseURL: baseURL,
		Auth:    ProviderAuth{APIKey: NewEnvAPIKeyAuth(name+" API key", envVar)},
		Models:  models,
		API:     openAICompletionsAPI{},
	})
}

func deepseekProvider() Provider {
	return openAICompatibleProvider("deepseek", "DeepSeek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", []*Model{
		mkModel("deepseek-chat", "DeepSeek Chat", ApiOpenAICompletions, "deepseek", 0.27, 1.1, 128000, 8192, false),
		mkModel("deepseek-reasoner", "DeepSeek Reasoner", ApiOpenAICompletions, "deepseek", 0.55, 2.19, 128000, 64000, true),
	})
}

func groqProvider() Provider {
	return openAICompatibleProvider("groq", "Groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", []*Model{
		mkModel("llama-4-maverick-17b", "Llama 4 Maverick", ApiOpenAICompletions, "groq", 0.2, 0.6, 131072, 8192, true),
	})
}

func xaiProvider() Provider {
	return openAICompatibleProvider("xai", "xAI", "https://api.x.ai/v1", "XAI_API_KEY", []*Model{
		mkModel("grok-4", "Grok 4", ApiOpenAICompletions, "xai", 3, 15, 256000, 32000, true),
	})
}

func openrouterProvider() Provider {
	return openAICompatibleProvider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", []*Model{
		mkModel("anthropic/claude-sonnet-4.5", "Claude Sonnet 4.5 (OpenRouter)", ApiOpenAICompletions, "openrouter", 3, 15, 200000, 64000, true),
	})
}

func mistralProvider() Provider {
	return openAICompatibleProvider("mistral", "Mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY", []*Model{
		mkModel("mistral-large-latest", "Mistral Large", ApiOpenAICompletions, "mistral", 2, 6, 128000, 8192, false),
	})
}

func moonshotaiProvider() Provider {
	return openAICompatibleProvider("moonshotai", "Moonshot AI", "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY", []*Model{
		mkModel("moonshot-v1-128k", "Moonshot v1", ApiOpenAICompletions, "moonshotai", 2, 6, 128000, 8192, false),
	})
}

func zaiProvider() Provider {
	return openAICompatibleProvider("zai", "Zhipu AI", "https://open.bigmodel.cn/api/paas/v4", "ZAI_API_KEY", []*Model{
		mkModel("glm-4.5", "GLM 4.5", ApiOpenAICompletions, "zai", 0.8, 1.6, 128000, 8192, true),
	})
}

func kimiCodingProvider() Provider {
	return openAICompatibleProvider("kimi-coding", "Kimi Coding", "https://api.moonshot.cn/v1", "KIMI_API_KEY", []*Model{
		mkModel("kimi-k2", "Kimi K2", ApiOpenAICompletions, "kimi-coding", 0.6, 2.5, 256000, 64000, true),
	})
}

func togetherProvider() Provider {
	return openAICompatibleProvider("together", "Together AI", "https://api.together.xyz/v1", "TOGETHER_API_KEY", []*Model{
		mkModel("deepseek-ai/DeepSeek-V3", "DeepSeek V3", ApiOpenAICompletions, "together", 0.27, 1.1, 128000, 8192, false),
	})
}

// builtinProviders returns all built-in provider factories.
func builtinProviders() []Provider {
	return []Provider{
		anthropicProvider(),
		openaiProvider(),
		googleProvider(),
		deepseekProvider(),
		groqProvider(),
		xaiProvider(),
		openrouterProvider(),
		mistralProvider(),
		moonshotaiProvider(),
		zaiProvider(),
		kimiCodingProvider(),
		togetherProvider(),
	}
}

// BuiltinModels returns a Models collection with all built-in providers
// registered.
func BuiltinModels() MutableModels {
	m := CreateModels(nil)
	for _, p := range builtinProviders() {
		m.SetProvider(p)
	}
	return m
}
