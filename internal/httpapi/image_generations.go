package httpapi

import (
	"encoding/json"
	"errors"
	"log"
	"maps"
	"net/http"

	"github.com/go-chi/chi/v5/middleware"
	"github.com/openai/openai-go/v3"
)

func (h *inferenceHandler) ServeImageGenerations(w http.ResponseWriter, r *http.Request) {
	var request map[string]any
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "request body must be valid JSON")
		return
	}

	publicModel, _ := request["model"].(string)
	model, ok := h.models[publicModel]
	if !ok {
		log.Printf("image generation model not found: request_id=%s model=%q", middleware.GetReqID(r.Context()), publicModel)
		writeOpenAIError(w, http.StatusNotFound, "model_not_found", "model not found")
		return
	}

	var lastErr error
	for _, route := range h.routes(model) {
		body := maps.Clone(request)
		body["model"] = route.model

		var upstream openai.ImagesResponse
		if err := route.client.Post(r.Context(), "images/generations", body, &upstream); err != nil {
			status := 0
			if apiErr, ok := errors.AsType[*openai.Error](err); ok {
				status = apiErr.StatusCode
			}
			log.Printf("image generation upstream route failed: request_id=%s provider=%q model=%q status=%d error=%T", middleware.GetReqID(r.Context()), route.provider, route.model, status, err)
			lastErr = err
			continue
		}

		images := make([]any, 0, len(upstream.Data))
		for _, image := range upstream.Data {
			item := map[string]any{}
			if image.JSON.B64JSON.Valid() {
				item["b64_json"] = image.B64JSON
			}
			if image.JSON.RevisedPrompt.Valid() {
				item["revised_prompt"] = image.RevisedPrompt
			}
			if image.JSON.URL.Valid() {
				item["url"] = image.URL
			}
			images = append(images, item)
		}

		response := map[string]any{"created": upstream.Created, "data": images}
		if upstream.JSON.Background.Valid() {
			response["background"] = upstream.Background
		}
		if upstream.JSON.OutputFormat.Valid() {
			response["output_format"] = upstream.OutputFormat
		}
		if upstream.JSON.Quality.Valid() {
			response["quality"] = upstream.Quality
		}
		if upstream.JSON.Size.Valid() {
			response["size"] = upstream.Size
		}
		if upstream.JSON.Usage.Valid() {
			usage := map[string]any{}
			if upstream.Usage.JSON.InputTokens.Valid() {
				usage["input_tokens"] = upstream.Usage.InputTokens
			}
			if upstream.Usage.JSON.OutputTokens.Valid() {
				usage["output_tokens"] = upstream.Usage.OutputTokens
			}
			if upstream.Usage.JSON.TotalTokens.Valid() {
				usage["total_tokens"] = upstream.Usage.TotalTokens
			}
			if upstream.Usage.JSON.InputTokensDetails.Valid() {
				usage["input_tokens_details"] = map[string]any{
					"image_tokens": upstream.Usage.InputTokensDetails.ImageTokens,
					"text_tokens":  upstream.Usage.InputTokensDetails.TextTokens,
				}
			}
			if upstream.Usage.JSON.OutputTokensDetails.Valid() {
				usage["output_tokens_details"] = map[string]any{
					"image_tokens": upstream.Usage.OutputTokensDetails.ImageTokens,
					"text_tokens":  upstream.Usage.OutputTokensDetails.TextTokens,
				}
			}
			response["usage"] = usage
		}
		writeJSON(w, http.StatusOK, response)
		return
	}
	h.writeOpenAIUpstreamError(w, r, lastErr)
}
