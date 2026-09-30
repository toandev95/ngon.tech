package httpapi

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"

	"ngon.tech/internal/config"
)

func NewRouter(cfg config.Config) http.Handler {
	router := chi.NewRouter()
	router.Use(middleware.RequestID, requestIDResponseHeader, middleware.Logger, middleware.Recoverer)

	models := newModelsHandler(cfg)
	inference := newInferenceHandler(cfg)
	router.Get("/v1/models", models.ServeModels)
	router.Post("/v1/messages", inference.ServeMessages)
	router.Post("/v1/messages/count_tokens", inference.ServeMessageTokenCount)
	router.Post("/v1/chat/completions", inference.ServeChatCompletions)
	router.Post("/v1/responses", inference.ServeResponses)
	return router
}

func requestIDResponseHeader(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		setRequestID(w.Header(), middleware.GetReqID(r.Context()))
		next.ServeHTTP(w, r)
	})
}

func setRequestID(header http.Header, requestID string) {
	if requestID != "" {
		header.Set("request-id", requestID)
	}
}

func writeOpenAIError(w http.ResponseWriter, status int, kind, message string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"type":    kind,
			"message": message,
			"code":    kind,
			"param":   nil,
		},
	})
}

func writeAnthropicError(w http.ResponseWriter, status int, kind, message, requestID string) {
	body := map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    kind,
			"message": message,
		},
		"request_id": nil,
	}
	if requestID != "" {
		w.Header().Set("request-id", requestID)
		body["request_id"] = requestID
	}
	writeJSON(w, status, body)
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(value); err != nil {
		log.Printf("write json response: status=%d error=%v", status, err)
	}
}
