package config

import (
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/BurntSushi/toml"
)

type Config struct {
	Server struct {
		Address string `toml:"address"`
	} `toml:"server"`
	Providers map[string]Provider `toml:"providers"`
	Models    map[string]Model    `toml:"models"`
}

type Provider struct {
	Protocol string `toml:"protocol"`
	BaseURL  string `toml:"base_url"`
	APIKey   string `toml:"api_key"`
	// Some LiteLLM adapters discard plain-text document blocks.
	TextDocumentsAsText bool `toml:"text_documents_as_text"`
}

type Model struct {
	DisplayName  string    `toml:"display_name"`
	OwnedBy      string    `toml:"owned_by"`
	SystemPrompt string    `toml:"system_prompt"`
	CreatedAt    time.Time `toml:"created_at"`
	Routes       []Route   `toml:"routes"`
}

// Route maps a public model to an upstream.
// Lower priority runs first; weight balances routes within the same priority.
type Route struct {
	Provider string `toml:"provider"`
	Model    string `toml:"model"`
	Priority int    `toml:"priority"`
	Weight   int    `toml:"weight"`
}

func Load(path string) (Config, error) {
	var cfg Config
	cfg.Server.Address = ":8080"
	meta, err := toml.DecodeFile(path, &cfg)
	if err != nil {
		return Config{}, fmt.Errorf("load config: %w", err)
	}
	if keys := meta.Undecoded(); len(keys) > 0 {
		return Config{}, fmt.Errorf("unknown config keys: %v", keys)
	}
	if strings.TrimSpace(cfg.Server.Address) == "" {
		return Config{}, fmt.Errorf("server.address must not be empty")
	}
	for name, provider := range cfg.Providers {
		if provider.Protocol == "" {
			provider.Protocol = "openai"
			cfg.Providers[name] = provider
		}
		if err := validateProvider(name, provider); err != nil {
			return Config{}, err
		}
	}
	for id, model := range cfg.Models {
		normalized, err := normalizeModel(id, model, cfg.Providers)
		if err != nil {
			return Config{}, err
		}
		cfg.Models[id] = normalized
	}
	return cfg, nil
}

func validateProvider(name string, provider Provider) error {
	if strings.TrimSpace(name) == "" {
		return fmt.Errorf("provider name must not be empty")
	}
	if provider.Protocol != "openai" {
		return fmt.Errorf("provider %q: protocol must be openai", name)
	}
	parsed, err := url.Parse(provider.BaseURL)
	if err != nil {
		return fmt.Errorf("provider %q: parse base_url: %w", name, err)
	}
	hasValidScheme := parsed.Scheme == "http" || parsed.Scheme == "https"
	hasForbiddenParts := parsed.RawQuery != "" || parsed.Fragment != "" || parsed.User != nil
	if parsed.Hostname() == "" || !hasValidScheme || hasForbiddenParts {
		return fmt.Errorf("provider %q: base_url must be an http or https url without credentials, query or fragment", name)
	}
	return nil
}

func normalizeModel(id string, model Model, providers map[string]Provider) (Model, error) {
	if strings.TrimSpace(id) == "" || len(model.Routes) == 0 {
		return Model{}, fmt.Errorf("model %q: non-empty id and at least one route required", id)
	}
	if model.DisplayName == "" {
		model.DisplayName = id
	}
	if model.OwnedBy == "" {
		model.OwnedBy = "ngon.tech"
	}
	if model.CreatedAt.IsZero() {
		model.CreatedAt = time.Unix(0, 0).UTC()
	}
	for index := range model.Routes {
		if err := normalizeRoute(id, index, &model.Routes[index], providers); err != nil {
			return Model{}, err
		}
	}
	return model, nil
}

func normalizeRoute(id string, index int, route *Route, providers map[string]Provider) error {
	if _, ok := providers[route.Provider]; !ok {
		return fmt.Errorf("model %q: unknown provider %q", id, route.Provider)
	}
	hasInvalidModel := strings.TrimSpace(route.Model) == ""
	hasInvalidRouting := route.Priority < 0 || route.Weight < 0
	if hasInvalidModel || hasInvalidRouting {
		return fmt.Errorf("model %q route %d: model required; priority and weight must not be negative", id, index)
	}
	if route.Weight == 0 {
		route.Weight = 1
	}
	return nil
}
