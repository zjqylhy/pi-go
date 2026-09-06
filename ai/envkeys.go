package ai

// Provider -> ambient api-key environment variable names (in priority order).
// Mirrors getApiKeyEnvVars in the TypeScript source.
var providerEnvKeys = map[string][]string{
	"github-copilot":         {"COPILOT_GITHUB_TOKEN"},
	"anthropic":              {"ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"},
	"ant-ling":               {"ANT_LING_API_KEY"},
	"openai":                 {"OPENAI_API_KEY"},
	"azure-openai-responses": {"AZURE_OPENAI_API_KEY"},
	"nvidia":                 {"NVIDIA_API_KEY"},
	"deepseek":               {"DEEPSEEK_API_KEY"},
	"google":                 {"GEMINI_API_KEY"},
	"google-vertex":          {"GOOGLE_CLOUD_API_KEY"},
	"groq":                   {"GROQ_API_KEY"},
	"cerebras":               {"CEREBRAS_API_KEY"},
	"xai":                    {"XAI_API_KEY"},
	"radius":                 {"RADIUS_API_KEY"},
	"openrouter":             {"OPENROUTER_API_KEY"},
	"vercel-ai-gateway":      {"AI_GATEWAY_API_KEY"},
	"zai":                    {"ZAI_API_KEY"},
	"zai-coding-cn":          {"ZAI_CODING_CN_API_KEY"},
	"mistral":                {"MISTRAL_API_KEY"},
	"minimax":                {"MINIMAX_API_KEY"},
	"minimax-cn":             {"MINIMAX_CN_API_KEY"},
	"moonshotai":             {"MOONSHOT_API_KEY"},
	"moonshotai-cn":          {"MOONSHOT_API_KEY"},
	"huggingface":            {"HF_TOKEN"},
	"fireworks":              {"FIREWORKS_API_KEY"},
	"together":               {"TOGETHER_API_KEY"},
	"baseten":                {"BASETEN_API_KEY"},
	"opencode":               {"OPENCODE_API_KEY"},
	"opencode-go":            {"OPENCODE_API_KEY"},
	"kimi-coding":            {"KIMI_API_KEY"},
	"cloudflare-workers-ai":  {"CLOUDFLARE_API_KEY"},
	"cloudflare-ai-gateway":  {"CLOUDFLARE_API_KEY"},
	"xiaomi":                 {"XIAOMI_API_KEY"},
}

// getAPIKeyEnvVars returns the ambient env-var names for a provider.
func getAPIKeyEnvVars(providerID string) []string {
	return providerEnvKeys[providerID]
}

// findEnvKeys returns which of the provider's env vars are set in the auth
// context.
func findEnvKeys(providerID string, authCtx AuthContext) []string {
	var found []string
	for _, name := range getAPIKeyEnvVars(providerID) {
		if authCtx.Env(name) != "" {
			found = append(found, name)
		}
	}
	return found
}

// getEnvAPIKey returns the first populated ambient api key for a provider.
// Anthropic's AUTH_TOKEN is skipped because it must be sent as a Bearer
// header, not x-api-key.
func getEnvAPIKey(providerID string, authCtx AuthContext) (string, string) {
	for _, name := range providerEnvKeys[providerID] {
		if providerID == "anthropic" && name == "ANTHROPIC_AUTH_TOKEN" {
			continue
		}
		if v := authCtx.Env(name); v != "" {
			return v, name
		}
	}
	return "", ""
}
