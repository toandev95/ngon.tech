package httpapi

import (
	"errors"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5/middleware"

	"ngon.tech/internal/config"
)

const (
	defaultModelLimit = 20
	maxModelLimit     = 1000
)

type modelsHandler struct {
	ids             []string
	openAIModels    []openAIModel
	anthropicModels []anthropicModel
}

type openAIModel struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type anthropicModel struct {
	ID             string `json:"id"`
	Type           string `json:"type"`
	DisplayName    string `json:"display_name"`
	CreatedAt      string `json:"created_at"`
	Capabilities   any    `json:"capabilities"`
	MaxInputTokens any    `json:"max_input_tokens"`
	MaxTokens      any    `json:"max_tokens"`
}

type modelPage struct {
	start   int
	end     int
	hasMore bool
}

func newModelsHandler(cfg config.Config) *modelsHandler {
	ids := make([]string, 0, len(cfg.Models))
	for id := range cfg.Models {
		ids = append(ids, id)
	}
	slices.SortFunc(ids, func(a, b string) int {
		if order := cfg.Models[b].CreatedAt.Compare(cfg.Models[a].CreatedAt); order != 0 {
			return order
		}
		return strings.Compare(a, b)
	})

	handler := &modelsHandler{
		ids:             ids,
		openAIModels:    make([]openAIModel, 0, len(ids)),
		anthropicModels: make([]anthropicModel, 0, len(ids)),
	}
	for _, id := range ids {
		handler.appendModel(id, cfg.Models[id])
	}
	return handler
}

func (h *modelsHandler) appendModel(id string, model config.Model) {
	h.openAIModels = append(h.openAIModels, openAIModel{
		ID:      id,
		Object:  "model",
		Created: model.CreatedAt.Unix(),
		OwnedBy: model.OwnedBy,
	})
	h.anthropicModels = append(h.anthropicModels, anthropicModel{
		ID:          id,
		Type:        "model",
		DisplayName: model.DisplayName,
		CreatedAt:   model.CreatedAt.UTC().Format(time.RFC3339),
	})
}

func (h *modelsHandler) ServeModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Add("Vary", "User-Agent")
	if !usesAnthropicFormat(r.UserAgent()) {
		writeJSON(w, http.StatusOK, map[string]any{
			"object": "list",
			"data":   h.openAIModels,
		})
		return
	}

	page, err := paginateModels(h.ids, r.URL.Query())
	if err != nil {
		writeAnthropicError(
			w,
			http.StatusBadRequest,
			"invalid_request_error",
			err.Error(),
			middleware.GetReqID(r.Context()),
		)
		return
	}

	var firstID, lastID *string
	if page.start < page.end {
		firstID = &h.ids[page.start]
		lastID = &h.ids[page.end-1]
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"data":     h.anthropicModels[page.start:page.end],
		"has_more": page.hasMore,
		"first_id": firstID,
		"last_id":  lastID,
	})
}

func usesAnthropicFormat(userAgent string) bool {
	userAgent = strings.ToLower(userAgent)
	return strings.Contains(userAgent, "anthropic") || strings.Contains(userAgent, "claude")
}

func paginateModels(ids []string, query url.Values) (modelPage, error) {
	limit := defaultModelLimit
	if query.Has("limit") {
		value, err := strconv.Atoi(query.Get("limit"))
		if err != nil || value < 1 || value > maxModelLimit {
			return modelPage{}, errors.New("limit must be between 1 and 1000")
		}
		limit = value
	}

	after, before := query.Get("after_id"), query.Get("before_id")
	if after != "" && before != "" {
		return modelPage{}, errors.New("use either after_id or before_id")
	}
	start, end := 0, len(ids)
	if after != "" || before != "" {
		cursor := after
		if before != "" {
			cursor = before
		}
		index := slices.Index(ids, cursor)
		if index < 0 {
			return modelPage{}, errors.New("unknown model cursor")
		}
		if before != "" {
			end = index
		} else {
			start = index + 1
		}
	}

	hasMore := end-start > limit
	if before != "" {
		start = max(start, end-limit)
	} else {
		end = min(end, start+limit)
	}
	return modelPage{start: start, end: end, hasMore: hasMore}, nil
}
