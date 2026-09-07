package ai

import (
	"context"
	"sync"
)

// ProviderStreams is the wire-level implementation of a model API. It is what
// the api implementations (anthropic, openai-*, google) provide.
type ProviderStreams interface {
	Stream(model *Model, context *Context, opts *StreamOptions) *AssistantMessageEventStream
	StreamSimple(model *Model, context *Context, opts *SimpleStreamOptions) *AssistantMessageEventStream
}

// DeferredStreams is an optional capability for async/deferred responses.
type DeferredStreams interface {
	FetchDeferred(model *Model, handle *DeferredHandle, opts *DeferredFetchOptions) *AssistantMessageEventStream
}

// DeferredCanceler is an optional capability for cancelling deferred responses.
type DeferredCanceler interface {
	CancelDeferred(model *Model, handle *DeferredHandle, opts *DeferredCancelOptions) error
}

// Provider is the runtime unit owning metadata, auth, model lists, and stream
// behavior for a single provider.
type Provider interface {
	ID() string
	Name() string
	BaseURL() string
	Headers() ProviderHeaders
	Auth() ProviderAuth
	GetModels() []*Model
	FilterModels(models []*Model, cred *Credential) []*Model
	Stream(model *Model, context *Context, opts *StreamOptions) *AssistantMessageEventStream
	StreamSimple(model *Model, context *Context, opts *SimpleStreamOptions) *AssistantMessageEventStream
}

// providerRefresher is an optional capability for dynamic model lists.
type providerRefresher interface {
	RefreshModels(ctx RefreshModelsContext) error
}

// ModelsStoreEntry is a persisted provider model catalog snapshot.
type ModelsStoreEntry struct {
	Models    []*Model `json:"models"`
	CheckedAt int64    `json:"checkedAt"`
}

// ModelsStore persists dynamic provider model catalogs.
type ModelsStore interface {
	Read(providerID string, ctx context.Context) (*ModelsStoreEntry, error)
	Write(providerID string, entry *ModelsStoreEntry, ctx context.Context) error
	Delete(providerID string, ctx context.Context) error
}

// InMemoryModelsStore is a non-persistent ModelsStore.
type InMemoryModelsStore struct {
	mu    sync.Mutex
	store map[string]*ModelsStoreEntry
}

func NewInMemoryModelsStore() *InMemoryModelsStore {
	return &InMemoryModelsStore{store: map[string]*ModelsStoreEntry{}}
}

func (s *InMemoryModelsStore) Read(providerID string, ctx context.Context) (*ModelsStoreEntry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.store[providerID], nil
}

func (s *InMemoryModelsStore) Write(providerID string, entry *ModelsStoreEntry, ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.store[providerID] = entry
	return nil
}

func (s *InMemoryModelsStore) Delete(providerID string, ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.store, providerID)
	return nil
}

// ModelsPublication controls how a provider publishes refreshed model state.
type ModelsPublication struct {
	// Delete removes the persisted snapshot.
	Delete bool
	// Persist stores a new snapshot (ignored when Delete is true).
	Persist *ModelsStoreEntry
	// Update synchronously mutates provider-private in-memory state.
	Update func()
}

// RefreshModelsContext is passed to a provider's RefreshModels.
type RefreshModelsContext struct {
	Credential   *Credential
	Stored       *ModelsStoreEntry
	Publish      func(ModelsPublication) (bool, error)
	AllowNetwork bool
	Force        bool
	Ctx          context.Context
}

// ModelsRefreshResult reports the outcome of a refresh.
type ModelsRefreshResult struct {
	Aborted bool
	Errors  map[string]error
}

// providerImpl is the concrete Provider returned by createProvider.
type providerImpl struct {
	id, name, baseURL string
	headers           ProviderHeaders
	auth              ProviderAuth

	mu          sync.Mutex
	baseline    []*Model
	dynamic     []*Model
	fetchModels func(RefreshModelsContext) ([]*Model, error)
	filter      func([]*Model, *Credential) []*Model

	single    ProviderStreams
	byAPI     map[string]ProviderStreams
	hasFetch  bool
	hasCancel bool
}

func (p *providerImpl) ID() string               { return p.id }
func (p *providerImpl) Name() string             { return p.name }
func (p *providerImpl) BaseURL() string          { return p.baseURL }
func (p *providerImpl) Headers() ProviderHeaders { return p.headers }
func (p *providerImpl) Auth() ProviderAuth       { return p.auth }

func (p *providerImpl) GetModels() []*Model {
	p.mu.Lock()
	defer p.mu.Unlock()
	merged := append([]*Model{}, p.baseline...)
	for _, dm := range p.dynamic {
		replaced := false
		for i, bm := range merged {
			if bm.ID == dm.ID {
				merged[i] = dm
				replaced = true
				break
			}
		}
		if !replaced {
			merged = append(merged, dm)
		}
	}
	return merged
}

func (p *providerImpl) FilterModels(models []*Model, cred *Credential) []*Model {
	if p.filter == nil {
		return models
	}
	return p.filter(models, cred)
}

func (p *providerImpl) apiFor(model *Model) ProviderStreams {
	if p.single != nil {
		return p.single
	}
	return p.byAPI[model.API]
}

func (p *providerImpl) Stream(model *Model, context *Context, opts *StreamOptions) *AssistantMessageEventStream {
	streams := p.apiFor(model)
	if streams == nil {
		return noAPIModelStream(p.id, model.API)
	}
	return streams.Stream(p.applyBaseURL(model), context, opts)
}

func (p *providerImpl) StreamSimple(model *Model, context *Context, opts *SimpleStreamOptions) *AssistantMessageEventStream {
	streams := p.apiFor(model)
	if streams == nil {
		return noAPIModelStream(p.id, model.API)
	}
	return streams.StreamSimple(p.applyBaseURL(model), context, opts)
}

// applyBaseURL returns the model with the provider base URL applied when the
// model does not define its own.
func (p *providerImpl) applyBaseURL(model *Model) *Model {
	if model.BaseURL != "" || p.baseURL == "" {
		return model
	}
	cp := *model
	cp.BaseURL = p.baseURL
	return &cp
}

func (p *providerImpl) RefreshModels(ctx RefreshModelsContext) error {
	if p.fetchModels == nil {
		return nil
	}
	if ctx.Stored != nil {
		restored := filterModelsByProvider(ctx.Stored.Models, p.id)
		ok, err := ctx.Publish(ModelsPublication{Update: func() {
			p.mu.Lock()
			p.dynamic = restored
			p.mu.Unlock()
		}})
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}
	}
	if !ctx.AllowNetwork || ctx.Ctx.Err() != nil {
		return nil
	}
	fresh, err := p.fetchModels(ctx)
	if err != nil {
		return err
	}
	if ctx.Ctx.Err() != nil {
		return nil
	}
	_, err = ctx.Publish(ModelsPublication{
		Persist: &ModelsStoreEntry{Models: fresh, CheckedAt: nowMs()},
		Update: func() {
			p.mu.Lock()
			p.dynamic = fresh
			p.mu.Unlock()
		},
	})
	return err
}

func (p *providerImpl) FetchDeferred(model *Model, handle *DeferredHandle, opts *DeferredFetchOptions) *AssistantMessageEventStream {
	impl, ok := p.apiFor(model).(DeferredStreams)
	if !ok {
		return noDeferredStream(p.id, model.API)
	}
	return impl.FetchDeferred(model, handle, opts)
}

func (p *providerImpl) CancelDeferred(model *Model, handle *DeferredHandle, opts *DeferredCancelOptions) error {
	impl, ok := p.apiFor(model).(DeferredCanceler)
	if !ok {
		return newModelsError(ModelsErrorProvider, "Provider "+p.id+" cannot cancel deferred responses for "+model.API, nil)
	}
	return impl.CancelDeferred(model, handle, opts)
}

func filterModelsByProvider(models []*Model, providerID string) []*Model {
	var out []*Model
	for _, m := range models {
		if m.Provider == providerID {
			out = append(out, m)
		}
	}
	return out
}

func noAPIModelStream(providerID, api string) *AssistantMessageEventStream {
	return errorStreamFor(newModelsError(ModelsErrorStream, "Provider "+providerID+" has no API implementation for "+api, nil))
}

func noDeferredStream(providerID, api string) *AssistantMessageEventStream {
	return errorStreamFor(newModelsError(ModelsErrorProvider, "Provider "+providerID+" does not support deferred responses for "+api, nil))
}

// errorStreamFor returns a stream that immediately yields a single error event.
func errorStreamFor(err error) *AssistantMessageEventStream {
	s := NewAssistantMessageEventStream()
	msg := &AssistantMessage{Role: "assistant", Content: []ContentBlock{}, StopReason: StopError, ErrorMessage: err.Error(), Timestamp: nowMs()}
	s.Push(Event{Type: EventError, Reason: StopError, Error: msg})
	s.End(msg)
	return s
}

// CreateProviderOptions are the inputs to createProvider.
type CreateProviderOptions struct {
	ID          string
	Name        string
	BaseURL     string
	Headers     ProviderHeaders
	Auth        ProviderAuth
	Models      []*Model
	FetchModels func(RefreshModelsContext) ([]*Model, error)
	// FilterModels restricts GetModels-per-auth availability.
	FilterModels func(models []*Model, cred *Credential) []*Model
	// API is a single ProviderStreams, or a map keyed by Model.API for
	// mixed-API providers.
	API any
}

// CreateProvider builds a Provider from its parts.
func CreateProvider(opts CreateProviderOptions) Provider {
	p := &providerImpl{
		id:          opts.ID,
		name:        opts.Name,
		baseURL:     opts.BaseURL,
		headers:     opts.Headers,
		auth:        opts.Auth,
		baseline:    opts.Models,
		fetchModels: opts.FetchModels,
		filter:      opts.FilterModels,
	}
	if p.name == "" {
		p.name = p.id
	}
	switch a := opts.API.(type) {
	case ProviderStreams:
		p.single = a
	case map[string]ProviderStreams:
		p.byAPI = a
	}
	return p
}

// CreateModelsOptions configure a Models collection.
type CreateModelsOptions struct {
	Credentials CredentialStore
	ModelsStore ModelsStore
	AuthContext AuthContext
}

// Models is the runtime collection of providers with auth resolution and
// stream convenience methods.
type Models interface {
	GetProviders() []Provider
	GetProvider(id string) Provider
	GetModels(providerID string) []*Model
	GetModel(providerID, id string) *Model
	Refresh(ctx context.Context) (*ModelsRefreshResult, error)
	CheckAuth(providerID string, ctx context.Context) (*AuthCheck, error)
	GetAvailable(providerID string, ctx context.Context) ([]*Model, error)
	GetAuth(providerID string, overrides AuthResolutionOverrides) (*AuthResult, error)
	Logout(providerID string, ctx context.Context) error

	Stream(model *Model, context *Context, opts *StreamOptions) *AssistantMessageEventStream
	Complete(model *Model, context *Context, opts *StreamOptions) (*AssistantMessage, error)
	StreamSimple(model *Model, context *Context, opts *SimpleStreamOptions) *AssistantMessageEventStream
	CompleteSimple(model *Model, context *Context, opts *SimpleStreamOptions) (*AssistantMessage, error)
}

// MutableModels adds provider mutation.
type MutableModels interface {
	Models
	SetProvider(p Provider)
	DeleteProvider(id string)
	ClearProviders()
}

type modelsImpl struct {
	providers   map[string]Provider
	credentials CredentialStore
	modelsStore ModelsStore
	authContext AuthContext
}

// CreateModels builds a new, empty Models collection. Use BuiltinModels to get
// the built-in providers pre-registered.
func CreateModels(opts *CreateModelsOptions) MutableModels {
	m := &modelsImpl{
		providers: map[string]Provider{},
	}
	if opts == nil {
		opts = &CreateModelsOptions{}
	}
	if opts.Credentials != nil {
		m.credentials = opts.Credentials
	} else {
		m.credentials = NewInMemoryCredentialStore()
	}
	if opts.ModelsStore != nil {
		m.modelsStore = opts.ModelsStore
	} else {
		m.modelsStore = NewInMemoryModelsStore()
	}
	if opts.AuthContext != nil {
		m.authContext = opts.AuthContext
	} else {
		m.authContext = EnvAuthContext{}
	}
	return m
}

func (m *modelsImpl) SetProvider(p Provider)   { m.providers[p.ID()] = p }
func (m *modelsImpl) DeleteProvider(id string) { delete(m.providers, id) }
func (m *modelsImpl) ClearProviders()          { m.providers = map[string]Provider{} }

func (m *modelsImpl) GetProviders() []Provider {
	out := make([]Provider, 0, len(m.providers))
	for _, p := range m.providers {
		out = append(out, p)
	}
	return out
}

func (m *modelsImpl) GetProvider(id string) Provider { return m.providers[id] }

func (m *modelsImpl) GetModels(providerID string) []*Model {
	if providerID != "" {
		if p, ok := m.providers[providerID]; ok {
			return p.GetModels()
		}
		return nil
	}
	var out []*Model
	for _, p := range m.providers {
		out = append(out, p.GetModels()...)
	}
	return out
}

func (m *modelsImpl) GetModel(providerID, id string) *Model {
	for _, model := range m.GetModels(providerID) {
		if model.ID == id {
			return model
		}
	}
	return nil
}

func (m *modelsImpl) Logout(providerID string, ctx context.Context) error {
	return m.credentials.Delete(providerID, ctx)
}

func (m *modelsImpl) GetAuth(providerID string, overrides AuthResolutionOverrides) (*AuthResult, error) {
	provider := m.providers[providerID]
	if provider == nil {
		return nil, nil
	}
	return resolveProviderAuth(provider, m.credentials, m.authContext, overrides)
}

func (m *modelsImpl) CheckAuth(providerID string, ctx context.Context) (*AuthCheck, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	provider := m.providers[providerID]
	if provider == nil {
		return nil, nil
	}
	cred, err := m.credentials.Read(providerID, ctx)
	if err != nil {
		return nil, err
	}
	if cred != nil && cred.Type == CredentialTypeOAuth {
		if provider.Auth().OAuth != nil {
			return &AuthCheck{Type: CredentialTypeOAuth, Source: "OAuth"}, nil
		}
		return nil, nil
	}
	result, err := resolveProviderAuth(provider, m.credentials, m.authContext, AuthResolutionOverrides{Ctx: ctx})
	if err != nil {
		return nil, err
	}
	if result == nil {
		return nil, nil
	}
	return &AuthCheck{Type: CredentialTypeAPIKey, Source: result.Source}, nil
}

func (m *modelsImpl) GetAvailable(providerID string, ctx context.Context) ([]*Model, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	var providers []Provider
	if providerID != "" {
		if p, ok := m.providers[providerID]; ok {
			providers = []Provider{p}
		}
	} else {
		providers = m.GetProviders()
	}
	var out []*Model
	for _, p := range providers {
		cred, err := m.credentials.Read(p.ID(), ctx)
		if err != nil {
			return nil, err
		}
		auth, err := m.CheckAuth(p.ID(), ctx)
		if err != nil {
			return nil, err
		}
		if auth == nil {
			continue
		}
		out = append(out, p.FilterModels(p.GetModels(), cred)...)
	}
	return out, nil
}

// Refresh attempts to refresh all dynamic, configured providers.
func (m *modelsImpl) Refresh(ctx context.Context) (*ModelsRefreshResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	result := &ModelsRefreshResult{Errors: map[string]error{}}
	for _, p := range m.providers {
		refresher, ok := p.(providerRefresher)
		if !ok {
			continue
		}
		err := refresher.RefreshModels(RefreshModelsContext{
			AllowNetwork: true,
			Ctx:          ctx,
			Publish: func(pub ModelsPublication) (bool, error) {
				if pub.Delete {
					if err := m.modelsStore.Delete(p.ID(), ctx); err != nil {
						return false, err
					}
				} else if pub.Persist != nil {
					if err := m.modelsStore.Write(p.ID(), pub.Persist, ctx); err != nil {
						return false, err
					}
				}
				if pub.Update != nil {
					pub.Update()
				}
				return true, nil
			},
		})
		if err != nil {
			result.Errors[p.ID()] = err
		}
	}
	return result, nil
}

func (m *modelsImpl) requireProvider(model *Model) (Provider, error) {
	p := m.providers[model.Provider]
	if p == nil {
		return nil, newModelsError(ModelsErrorProvider, "Unknown provider: "+model.Provider, nil)
	}
	return p, nil
}

// applyAuth merges resolved auth (key, headers, baseUrl) into request options.
func (m *modelsImpl) applyAuth(model *Model, opts *StreamOptions) (*Model, *StreamOptions, error) {
	if _, err := m.requireProvider(model); err != nil {
		return nil, nil, err
	}
	resolution, err := resolveProviderAuth(m.providers[model.Provider], m.credentials, m.authContext, AuthResolutionOverrides{
		APIKey: opts.APIKey,
		Env:    opts.Env,
		Ctx:    opts.Ctx,
	})
	if err != nil {
		return nil, nil, err
	}
	if resolution == nil {
		return nil, nil, newModelsError(ModelsErrorAuth, "Provider is not configured: "+model.Provider, nil)
	}
	auth := resolution.Auth

	apiKey := opts.APIKey
	if apiKey == "" {
		apiKey = auth.APIKey
	}
	merged := mergeHeaders(auth.Headers, opts.Headers)

	requestModel := model
	if auth.BaseURL != "" {
		cp := *model
		cp.BaseURL = auth.BaseURL
		requestModel = &cp
	}

	out := *opts
	out.APIKey = apiKey
	out.Headers = merged
	return requestModel, &out, nil
}

func mergeHeaders(base, override ProviderHeaders) ProviderHeaders {
	if base == nil && override == nil {
		return nil
	}
	merged := ProviderHeaders{}
	for k, v := range base {
		merged[k] = v
	}
	for k, v := range override {
		lower := lowerStr(k)
		for existing := range merged {
			if lowerStr(existing) == lower {
				delete(merged, existing)
			}
		}
		merged[k] = v
	}
	return merged
}

func lowerStr(s string) string {
	b := []byte(s)
	for i := range b {
		if b[i] >= 'A' && b[i] <= 'Z' {
			b[i] += 'a' - 'A'
		}
	}
	return string(b)
}

func (m *modelsImpl) Stream(model *Model, context *Context, opts *StreamOptions) *AssistantMessageEventStream {
	if opts == nil {
		opts = &StreamOptions{}
	}
	return lazyStream(model, func() (*AssistantMessageEventStream, error) {
		provider, err := m.requireProvider(model)
		if err != nil {
			return nil, err
		}
		requestModel, requestOpts, err := m.applyAuth(model, opts)
		if err != nil {
			return nil, err
		}
		return provider.Stream(requestModel, context, requestOpts), nil
	})
}

func (m *modelsImpl) Complete(model *Model, context *Context, opts *StreamOptions) (*AssistantMessage, error) {
	return m.Stream(model, context, opts).Result(), nil
}

func (m *modelsImpl) StreamSimple(model *Model, context *Context, opts *SimpleStreamOptions) *AssistantMessageEventStream {
	if opts == nil {
		opts = &SimpleStreamOptions{}
	}
	return lazyStream(model, func() (*AssistantMessageEventStream, error) {
		provider, err := m.requireProvider(model)
		if err != nil {
			return nil, err
		}
		requestModel, requestOpts, err := m.applyAuth(model, &opts.StreamOptions)
		if err != nil {
			return nil, err
		}
		cp := *opts
		cp.StreamOptions = *requestOpts
		return provider.StreamSimple(requestModel, context, &cp), nil
	})
}

func (m *modelsImpl) CompleteSimple(model *Model, context *Context, opts *SimpleStreamOptions) (*AssistantMessage, error) {
	return m.StreamSimple(model, context, opts).Result(), nil
}

// hasApi reports whether a dynamically-looked-up model speaks the given API.
func hasApi(model *Model, api Api) bool { return model.API == api }

// modelsAreEqual compares models by id and provider.
func modelsAreEqual(a, b *Model) bool {
	if a == nil || b == nil {
		return false
	}
	return a.ID == b.ID && a.Provider == b.Provider
}

// calculateCost fills usage.cost from model rates, applying volume tiers and
// the Anthropic 2x 1h-cache-write rule.
func calculateCost(model *Model, usage *Usage) UsageCost {
	inputTokens := usage.Input + usage.CacheRead + usage.CacheWrite
	rates := ModelCostRates{
		Input:      model.Cost.Input,
		Output:     model.Cost.Output,
		CacheRead:  model.Cost.CacheRead,
		CacheWrite: model.Cost.CacheWrite,
	}
	matched := -1
	for _, tier := range model.Cost.Tiers {
		if inputTokens > tier.InputTokensAbove && tier.InputTokensAbove > matched {
			rates = ModelCostRates{
				Input:      tier.Input,
				Output:     tier.Output,
				CacheRead:  tier.CacheRead,
				CacheWrite: tier.CacheWrite,
			}
			matched = tier.InputTokensAbove
		}
	}
	longWrite := usage.CacheWrite1h
	shortWrite := usage.CacheWrite - longWrite
	cost := UsageCost{
		Input:      (rates.Input / 1e6) * float64(usage.Input),
		Output:     (rates.Output / 1e6) * float64(usage.Output),
		CacheRead:  (rates.CacheRead / 1e6) * float64(usage.CacheRead),
		CacheWrite: (rates.CacheWrite*float64(shortWrite) + rates.Input*2*float64(longWrite)) / 1e6,
	}
	cost.Total = cost.Input + cost.Output + cost.CacheRead + cost.CacheWrite
	return cost
}

var extendedThinkingLevels = []ThinkingLevel{
	ThinkingOff, ThinkingMinimal, ThinkingLow, ThinkingMedium, ThinkingHigh, ThinkingXHigh, ThinkingMax,
}

// GetSupportedThinkingLevels returns the thinking levels a reasoning model
// supports.
func GetSupportedThinkingLevels(model *Model) []ThinkingLevel {
	if !model.Reasoning {
		return []ThinkingLevel{ThinkingOff}
	}
	var out []ThinkingLevel
	for _, level := range extendedThinkingLevels {
		mapped, ok := model.ThinkingLevelMap[level]
		if ok && mapped == nil {
			continue
		}
		if level == ThinkingXHigh || level == ThinkingMax {
			if !ok {
				continue
			}
		}
		out = append(out, level)
	}
	return out
}

// ClampThinkingLevel clamps a requested level to the nearest supported level.
func ClampThinkingLevel(model *Model, level ThinkingLevel) ThinkingLevel {
	available := GetSupportedThinkingLevels(model)
	if containsLevel(available, level) {
		return level
	}
	idx := indexOfLevel(extendedThinkingLevels, level)
	if idx == -1 {
		return firstLevel(available)
	}
	for i := idx + 1; i < len(extendedThinkingLevels); i++ {
		if containsLevel(available, extendedThinkingLevels[i]) {
			return extendedThinkingLevels[i]
		}
	}
	for i := idx - 1; i >= 0; i-- {
		if containsLevel(available, extendedThinkingLevels[i]) {
			return extendedThinkingLevels[i]
		}
	}
	return firstLevel(available)
}

func containsLevel(levels []ThinkingLevel, l ThinkingLevel) bool {
	for _, x := range levels {
		if x == l {
			return true
		}
	}
	return false
}

func indexOfLevel(levels []ThinkingLevel, l ThinkingLevel) int {
	for i, x := range levels {
		if x == l {
			return i
		}
	}
	return -1
}

func firstLevel(levels []ThinkingLevel) ThinkingLevel {
	if len(levels) == 0 {
		return ThinkingOff
	}
	return levels[0]
}
