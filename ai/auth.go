package ai

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// CredentialType discriminates the credential union.
type CredentialType = string

const (
	CredentialTypeAPIKey CredentialType = "api_key"
	CredentialTypeOAuth  CredentialType = "oauth"
)

// Credential is a single stored auth credential for a provider. The Type field
// discriminates the two variants; unused fields are simply empty.
type Credential struct {
	Type CredentialType `json:"type"`

	// api_key variant
	Key string      `json:"key,omitempty"`
	Env ProviderEnv `json:"env,omitempty"`

	// oauth variant
	Refresh string `json:"refresh,omitempty"`
	Access  string `json:"access,omitempty"`
	Expires int64  `json:"expires,omitempty"`
}

// CredentialInfo is non-secret credential metadata.
type CredentialInfo struct {
	ProviderID string
	Type       CredentialType
}

// CredentialStore persists at most one credential per provider.
type CredentialStore interface {
	Read(providerID string, ctx context.Context) (*Credential, error)
	List(ctx context.Context) ([]CredentialInfo, error)
	// Modify is the only write path: a serialized read-modify-write whose fn
	// runs under the provider's lock.
	Modify(providerID string, ctx context.Context, fn func(*Credential) (*Credential, error)) (*Credential, error)
	Delete(providerID string, ctx context.Context) error
}

// InMemoryCredentialStore is the non-persistent default store.
type InMemoryCredentialStore struct {
	mu    sync.Mutex
	locks map[string]*sync.Mutex
	creds map[string]*Credential
}

func NewInMemoryCredentialStore() *InMemoryCredentialStore {
	return &InMemoryCredentialStore{locks: map[string]*sync.Mutex{}, creds: map[string]*Credential{}}
}

func (s *InMemoryCredentialStore) lock(id string) *sync.Mutex {
	s.mu.Lock()
	defer s.mu.Unlock()
	l, ok := s.locks[id]
	if !ok {
		l = &sync.Mutex{}
		s.locks[id] = l
	}
	return l
}

func (s *InMemoryCredentialStore) Read(providerID string, ctx context.Context) (*Credential, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	l := s.lock(providerID)
	l.Lock()
	defer l.Unlock()
	return s.creds[providerID], nil
}

func (s *InMemoryCredentialStore) List(ctx context.Context) ([]CredentialInfo, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	out := []CredentialInfo{}
	for id, c := range s.creds {
		out = append(out, CredentialInfo{ProviderID: id, Type: c.Type})
	}
	return out, nil
}

func (s *InMemoryCredentialStore) Modify(providerID string, ctx context.Context, fn func(*Credential) (*Credential, error)) (*Credential, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	l := s.lock(providerID)
	l.Lock()
	defer l.Unlock()
	cur := s.creds[providerID]
	next, err := fn(cur)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if next != nil {
		s.creds[providerID] = next
		return next, nil
	}
	return cur, nil
}

func (s *InMemoryCredentialStore) Delete(providerID string, ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	l := s.lock(providerID)
	l.Lock()
	defer l.Unlock()
	delete(s.creds, providerID)
	return nil
}

// AuthContext supplies ambient environment and file presence checks.
type AuthContext interface {
	Env(name string) string
	FileExists(path string) bool
}

// EnvAuthContext resolves environment from the process environment.
type EnvAuthContext struct{}

func (EnvAuthContext) Env(name string) string { return os.Getenv(name) }

func (EnvAuthContext) FileExists(path string) bool {
	if strings.HasPrefix(path, "~") {
		home, err := os.UserHomeDir()
		if err != nil {
			return false
		}
		path = filepath.Join(home, path[1:])
	}
	_, err := os.Stat(path)
	return err == nil
}

// ModelAuth expresses the per-request auth outcome: an api key, headers, or a
// base URL override. Anything beyond these is provider configuration.
type ModelAuth struct {
	APIKey  string
	Headers ProviderHeaders
	BaseURL string
}

// AuthResult is a resolved auth plus a source label for status UI.
type AuthResult struct {
	Auth   ModelAuth
	Env    ProviderEnv
	Source string
}

// AuthCheck reports whether a provider has complete auth without refreshing.
type AuthCheck struct {
	Source string
	Type   CredentialType
}

// AuthType is the login method discriminator.
type AuthType = string

const (
	AuthTypeAPIKey AuthType = "api_key"
	AuthTypeOAuth  AuthType = "oauth"
)

// ApiKeyAuth is the api-key auth strategy of a provider.
type ApiKeyAuth interface {
	Name() string
	// Resolve returns a best-effort auth result, or (nil, nil) when the
	// provider is unconfigured.
	Resolve(ctx AuthContext, cred *Credential, cancelCtx context.Context) (*AuthResult, error)
}

// OAuthAuth is the oauth auth strategy of a provider.
type OAuthAuth interface {
	Name() string
	// ToAuth derives a ModelAuth from a stored oauth credential.
	ToAuth(cred *Credential) (*ModelAuth, error)
	// Refresh exchanges a refresh token for a fresh credential.
	Refresh(cred *Credential, cancelCtx context.Context) (*Credential, error)
}

// ProviderAuth bundles the (optional) auth strategies. At least one must be
// present.
type ProviderAuth struct {
	APIKey ApiKeyAuth
	OAuth  OAuthAuth
}

// ModelsErrorCode enumerates error categories.
type ModelsErrorCode = string

const (
	ModelsErrorModelSource     ModelsErrorCode = "model_source"
	ModelsErrorModelValidation ModelsErrorCode = "model_validation"
	ModelsErrorProvider        ModelsErrorCode = "provider"
	ModelsErrorStream          ModelsErrorCode = "stream"
	ModelsErrorAuth            ModelsErrorCode = "auth"
	ModelsErrorOAuth           ModelsErrorCode = "oauth"
)

// ModelsError is the package's typed error.
type ModelsError struct {
	Code  ModelsErrorCode
	Msg   string
	Cause error
}

func (e *ModelsError) Error() string {
	if e.Cause != nil && !strings.Contains(e.Msg, e.Cause.Error()) {
		return fmt.Sprintf("%s: %v", e.Msg, e.Cause)
	}
	return e.Msg
}

func (e *ModelsError) Unwrap() error { return e.Cause }

func newModelsError(code ModelsErrorCode, msg string, cause error) error {
	return &ModelsError{Code: code, Msg: msg, Cause: cause}
}

// AuthResolutionOverrides are per-request auth overrides.
type AuthResolutionOverrides struct {
	APIKey string
	Env    ProviderEnv
	Ctx    context.Context
}

// overlayAuthContext lets an explicit env override take precedence over the
// ambient environment.
type overlayAuthContext struct {
	base AuthContext
	env  ProviderEnv
}

func (o overlayAuthContext) Env(name string) string {
	if v, ok := o.env[name]; ok {
		return v
	}
	return o.base.Env(name)
}

func (o overlayAuthContext) FileExists(path string) bool { return o.base.FileExists(path) }

// envApiKeyAuth is the standard ambient api-key auth: stored key first, then
// the first populated env var.
type envApiKeyAuth struct {
	name    string
	envVars []string
}

func (a envApiKeyAuth) Name() string { return a.name }

func (a envApiKeyAuth) Resolve(ctx AuthContext, cred *Credential, cancelCtx context.Context) (*AuthResult, error) {
	if cred != nil && cred.Key != "" {
		return &AuthResult{
			Auth:   ModelAuth{APIKey: cred.Key},
			Env:    cred.Env,
			Source: "stored credential",
		}, nil
	}
	for _, name := range a.envVars {
		if v := ctx.Env(name); v != "" {
			return &AuthResult{Auth: ModelAuth{APIKey: v}, Source: name}, nil
		}
	}
	return nil, nil
}

// NewEnvAPIKeyAuth builds an ambient api-key auth for the given env names.
func NewEnvAPIKeyAuth(name string, envVars ...string) ApiKeyAuth {
	return envApiKeyAuth{name: name, envVars: envVars}
}

// resolveProviderAuth resolves a provider's auth, mirroring the TypeScript
// order: explicit apiKey override, stored credential, then ambient env.
func resolveProviderAuth(provider Provider, credentials CredentialStore, authCtx AuthContext, overrides AuthResolutionOverrides) (*AuthResult, error) {
	ctx := overrides.Ctx
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	pa := provider.Auth()
	effective := authCtx
	if overrides.Env != nil {
		effective = overlayAuthContext{base: authCtx, env: overrides.Env}
	}

	if overrides.APIKey != "" && pa.APIKey != nil {
		result, err := pa.APIKey.Resolve(effective, &Credential{Type: CredentialTypeAPIKey, Key: overrides.APIKey, Env: overrides.Env}, ctx)
		if err != nil {
			return nil, newModelsError(ModelsErrorAuth, "API key auth failed for provider "+provider.ID(), err)
		}
		if result != nil {
			return result, nil
		}
	}

	cred, err := credentials.Read(provider.ID(), ctx)
	if err != nil {
		return nil, newModelsError(ModelsErrorAuth, "Credential store read failed for "+provider.ID(), err)
	}
	if cred != nil {
		switch cred.Type {
		case CredentialTypeOAuth:
			if pa.OAuth == nil {
				return nil, nil
			}
			return resolveStoredOAuth(provider.ID(), pa.OAuth, cred, credentials, ctx)
		case CredentialTypeAPIKey:
			if pa.APIKey == nil {
				return nil, nil
			}
			// merge overrides env into stored env
			merged := cred
			if overrides.Env != nil {
				cp := *cred
				if cp.Env == nil {
					cp.Env = ProviderEnv{}
				}
				for k, v := range overrides.Env {
					cp.Env[k] = v
				}
				merged = &cp
			}
			result, err := pa.APIKey.Resolve(effective, merged, ctx)
			if err != nil {
				return nil, newModelsError(ModelsErrorAuth, "API key auth failed for provider "+provider.ID(), err)
			}
			return result, nil
		default:
			return nil, nil
		}
	}

	if pa.APIKey != nil {
		result, err := pa.APIKey.Resolve(effective, nil, ctx)
		if err != nil {
			return nil, newModelsError(ModelsErrorAuth, "API key auth failed for provider "+provider.ID(), err)
		}
		return result, nil
	}
	return nil, nil
}

func resolveStoredOAuth(providerID string, oauth OAuthAuth, cred *Credential, credentials CredentialStore, ctx context.Context) (*AuthResult, error) {
	const minValidityMs = 5 * 60 * 1000
	if time.Now().UnixMilli()+minValidityMs >= cred.Expires {
		next, err := credentials.Modify(providerID, ctx, func(cur *Credential) (*Credential, error) {
			if cur == nil || cur.Type != CredentialTypeOAuth {
				return nil, nil
			}
			if time.Now().UnixMilli()+minValidityMs < cur.Expires {
				return nil, nil // refreshed concurrently
			}
			return oauth.Refresh(cur, ctx)
		})
		if err != nil {
			return nil, newModelsError(ModelsErrorOAuth, "OAuth refresh failed for provider "+providerID, err)
		}
		if next != nil {
			cred = next
		}
	}
	auth, err := oauth.ToAuth(cred)
	if err != nil {
		return nil, newModelsError(ModelsErrorOAuth, "OAuth auth derivation failed for provider "+providerID, err)
	}
	return &AuthResult{Auth: *auth, Source: "OAuth"}, nil
}

var errNotImplemented = errors.New("not implemented")
