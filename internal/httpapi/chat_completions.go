package httpapi

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"maps"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/go-chi/chi/v5/middleware"
	"github.com/openai/openai-go/v3"

	"ngon.tech/internal/config"
)

type chatStreamToolCall struct {
	id        string
	name      string
	arguments strings.Builder
}

func (h *inferenceHandler) ServeChatCompletions(w http.ResponseWriter, r *http.Request) {
	var request map[string]any
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "request body must be valid JSON")
		return
	}

	publicModel, _ := request["model"].(string)
	model, ok := h.models[publicModel]
	if !ok {
		log.Printf("chat completions model not found: request_id=%s model=%q", middleware.GetReqID(r.Context()), publicModel)
		writeOpenAIError(w, http.StatusNotFound, "model_not_found", "model not found")
		return
	}

	body, allowedTools := prepareChatCompletionRequest(request, publicModel, model)
	stream, _ := request["stream"].(bool)
	if stream {
		h.serveChatCompletionStream(w, r, publicModel, h.routes(model), body, allowedTools)
		return
	}
	h.serveChatCompletion(w, r, publicModel, h.routes(model), body, allowedTools)
}

func prepareChatCompletionRequest(request map[string]any, publicModel string, model config.Model) (map[string]any, map[string]struct{}) {
	body := maps.Clone(request)
	delete(body, "model")
	delete(body, "stream")

	if systemPrompt := strings.TrimSpace(model.SystemPrompt); systemPrompt != "" {
		systemPrompt = strings.NewReplacer("{{model}}", publicModel, "{{owner}}", model.OwnedBy).Replace(systemPrompt)
		messages, _ := request["messages"].([]any)
		messages = slices.Clone(messages)
		position := 0
		for position < len(messages) {
			message, ok := messages[position].(map[string]any)
			role, _ := message["role"].(string)
			if !ok || (role != "system" && role != "developer") {
				break
			}
			position++
		}
		messages = slices.Insert(messages, position, any(map[string]any{"role": "system", "content": systemPrompt}))
		body["messages"] = messages
	}

	allowedTools := make(map[string]struct{})
	if tools, ok := request["tools"].([]any); ok {
		for _, rawTool := range tools {
			tool, ok := rawTool.(map[string]any)
			if !ok {
				continue
			}
			kind, _ := tool["type"].(string)
			definition, _ := tool[kind].(map[string]any)
			name, _ := definition["name"].(string)
			if name != "" && (kind == "function" || kind == "custom") {
				allowedTools[name] = struct{}{}
			}
		}
	}
	if functions, ok := request["functions"].([]any); ok {
		for _, rawFunction := range functions {
			if function, ok := rawFunction.(map[string]any); ok {
				if name, _ := function["name"].(string); name != "" {
					allowedTools[name] = struct{}{}
				}
			}
		}
	}
	return body, allowedTools
}

func (h *inferenceHandler) serveChatCompletion(w http.ResponseWriter, r *http.Request, publicModel string, routes []upstreamRoute, body map[string]any, allowedTools map[string]struct{}) {
	var lastErr error
	for _, route := range routes {
		upstreamBody := maps.Clone(body)
		delete(upstreamBody, "stream_options")
		params := openai.ChatCompletionNewParams{Model: route.model, Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("")}}
		params.SetExtraFields(upstreamBody)
		completion, err := route.client.Chat.Completions.New(r.Context(), params)
		if err != nil {
			status := 0
			if apiErr, ok := errors.AsType[*openai.Error](err); ok {
				status = apiErr.StatusCode
			}
			log.Printf("chat completions upstream route failed: request_id=%s provider=%q model=%q status=%d error=%T", middleware.GetReqID(r.Context()), route.provider, route.model, status, err)
			lastErr = err
			continue
		}
		if len(completion.Choices) == 0 {
			lastErr = errors.New("upstream returned no choices")
			continue
		}

		choices := make([]any, 0, len(completion.Choices))
		filteredToolCalls := 0
		for _, choice := range completion.Choices {
			message := map[string]any{"role": "assistant", "content": choice.Message.Content}
			if choice.Message.Refusal != "" {
				message["refusal"] = choice.Message.Refusal
			}
			toolCalls := make([]any, 0, len(choice.Message.ToolCalls))
			for _, call := range choice.Message.ToolCalls {
				name, arguments, kind := call.Function.Name, call.Function.Arguments, "function"
				if call.Type == "custom" {
					name, arguments, kind = call.Custom.Name, call.Custom.Input, "custom"
				}
				if _, allowed := allowedTools[name]; !allowed {
					filteredToolCalls++
					continue
				}
				if kind == "custom" {
					toolCalls = append(toolCalls, map[string]any{"id": call.ID, "type": kind, "custom": map[string]any{"name": name, "input": arguments}})
				} else {
					toolCalls = append(toolCalls, map[string]any{"id": call.ID, "type": kind, "function": map[string]any{"name": name, "arguments": arguments}})
				}
			}
			if len(toolCalls) > 0 {
				message["tool_calls"] = toolCalls
				if choice.Message.Content == "" {
					message["content"] = nil
				}
			}
			finishReason := choice.FinishReason
			if len(toolCalls) == 0 && (finishReason == "tool_calls" || finishReason == "function_call") {
				finishReason = "stop"
			}
			result := map[string]any{"index": choice.Index, "message": message, "finish_reason": finishReason}
			if choice.Logprobs.JSON.Content.Valid() || choice.Logprobs.JSON.Refusal.Valid() {
				result["logprobs"] = standardJSON(choice.Logprobs)
			}
			choices = append(choices, result)
		}
		if filteredToolCalls > 0 {
			log.Printf("filtered undeclared upstream chat tool calls: request_id=%s count=%d", middleware.GetReqID(r.Context()), filteredToolCalls)
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"id": completion.ID, "object": "chat.completion", "created": completion.Created,
			"model": publicModel, "choices": choices, "usage": openAIUsage(completion.Usage),
		})
		return
	}
	h.writeOpenAIUpstreamError(w, r, lastErr)
}

func (h *inferenceHandler) serveChatCompletionStream(w http.ResponseWriter, r *http.Request, publicModel string, routes []upstreamRoute, body map[string]any, allowedTools map[string]struct{}) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeOpenAIError(w, http.StatusInternalServerError, "api_error", "streaming is unavailable")
		return
	}
	includeUsage := false
	if options, ok := body["stream_options"].(map[string]any); ok {
		includeUsage, _ = options["include_usage"].(bool)
	}
	streamBody := maps.Clone(body)
	streamBody["stream_options"] = map[string]any{"include_usage": true}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	_, _ = fmt.Fprint(w, ": keep-alive\n\n")
	flusher.Flush()

	start := make(chan streamStart, 1)
	go func() {
		var lastErr error
		for _, route := range routes {
			params := openai.ChatCompletionNewParams{Model: route.model, Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("")}}
			params.SetExtraFields(streamBody)
			sdkStream := route.client.Chat.Completions.NewStreaming(r.Context(), params)
			if !sdkStream.Next() {
				lastErr = sdkStream.Err()
				_ = sdkStream.Close()
				continue
			}
			start <- streamStart{first: sdkStream.Current(), stream: &openAIStream{next: sdkStream.Next, current: sdkStream.Current, err: sdkStream.Err, close: sdkStream.Close}}
			return
		}
		start <- streamStart{err: lastErr}
	}()

	ticker := time.NewTicker(streamPingInterval)
	defer ticker.Stop()
	var started streamStart
	for {
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
			_, _ = fmt.Fprint(w, ": keep-alive\n\n")
			flusher.Flush()
		case started = <-start:
			if started.err != nil || started.stream == nil {
				writeOpenAIStreamData(w, flusher, map[string]any{"error": openAIErrorBody(started.err)})
				writeOpenAIStreamData(w, flusher, "[DONE]")
				return
			}
			goto streamReady
		}
	}

streamReady:
	defer func() { _ = started.stream.close() }()
	chunks := make(chan openai.ChatCompletionChunk, 1)
	streamDone := make(chan error, 1)
	go func() {
		select {
		case chunks <- started.first:
		case <-r.Context().Done():
			return
		}
		for started.stream.next() {
			select {
			case chunks <- started.stream.current():
			case <-r.Context().Done():
				return
			}
		}
		streamDone <- started.stream.err()
		close(chunks)
	}()

	type choiceState struct {
		finishReason string
		tools        map[int64]*chatStreamToolCall
		toolOrder    []int64
	}
	states := make(map[int64]*choiceState)
	var usage openai.CompletionUsage
	responseID := ""
	created := int64(0)
	for {
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
			_, _ = fmt.Fprint(w, ": keep-alive\n\n")
			flusher.Flush()
		case chunk, ok := <-chunks:
			if !ok {
				if streamErr := <-streamDone; streamErr != nil {
					writeOpenAIStreamData(w, flusher, map[string]any{"error": openAIErrorBody(streamErr)})
					writeOpenAIStreamData(w, flusher, "[DONE]")
					return
				}
				goto streamFinished
			}
			if responseID == "" {
				responseID, created = chunk.ID, chunk.Created
			}
			if chunk.Usage.JSON.PromptTokens.Valid() || chunk.Usage.PromptTokens > 0 {
				usage = chunk.Usage
			}
			for _, choice := range chunk.Choices {
				state := states[choice.Index]
				if state == nil {
					state = &choiceState{tools: make(map[int64]*chatStreamToolCall)}
					states[choice.Index] = state
				}
				if choice.FinishReason != "" {
					state.finishReason = choice.FinishReason
				}
				for _, delta := range choice.Delta.ToolCalls {
					tool := state.tools[delta.Index]
					if tool == nil {
						tool = &chatStreamToolCall{}
						state.tools[delta.Index] = tool
						state.toolOrder = append(state.toolOrder, delta.Index)
					}
					if delta.ID != "" {
						tool.id = delta.ID
					}
					if delta.Function.Name != "" {
						tool.name = delta.Function.Name
					}
					tool.arguments.WriteString(delta.Function.Arguments)
				}
				delta := map[string]any{}
				if choice.Delta.Role != "" {
					delta["role"] = choice.Delta.Role
				}
				if choice.Delta.JSON.Content.Valid() || choice.Delta.Content != "" {
					delta["content"] = choice.Delta.Content
				}
				if choice.Delta.JSON.Refusal.Valid() || choice.Delta.Refusal != "" {
					delta["refusal"] = choice.Delta.Refusal
				}
				if len(delta) > 0 {
					streamChoice := map[string]any{"index": choice.Index, "delta": delta, "finish_reason": nil}
					if choice.Logprobs.JSON.Content.Valid() || choice.Logprobs.JSON.Refusal.Valid() {
						streamChoice["logprobs"] = standardJSON(choice.Logprobs)
					}
					writeOpenAIStreamData(w, flusher, map[string]any{
						"id": chunk.ID, "object": "chat.completion.chunk", "created": chunk.Created,
						"model": publicModel, "choices": []any{streamChoice},
					})
				}
			}
		}
	}

streamFinished:
	filteredToolCalls := 0
	choiceIndexes := make([]int64, 0, len(states))
	for index := range states {
		choiceIndexes = append(choiceIndexes, index)
	}
	slices.Sort(choiceIndexes)
	for _, choiceIndex := range choiceIndexes {
		state := states[choiceIndex]
		slices.Sort(state.toolOrder)
		toolCalls := make([]any, 0, len(state.toolOrder))
		for _, toolIndex := range state.toolOrder {
			tool := state.tools[toolIndex]
			if _, allowed := allowedTools[tool.name]; !allowed {
				filteredToolCalls++
				continue
			}
			toolCalls = append(toolCalls, map[string]any{
				"index": toolIndex, "id": tool.id, "type": "function",
				"function": map[string]any{"name": tool.name, "arguments": tool.arguments.String()},
			})
		}
		if len(toolCalls) > 0 {
			writeOpenAIStreamData(w, flusher, map[string]any{
				"id": responseID, "object": "chat.completion.chunk", "created": created, "model": publicModel,
				"choices": []any{map[string]any{"index": choiceIndex, "delta": map[string]any{"tool_calls": toolCalls}, "finish_reason": nil}},
			})
		}
		finishReason := state.finishReason
		if finishReason == "" {
			finishReason = "stop"
		}
		if len(toolCalls) == 0 && (finishReason == "tool_calls" || finishReason == "function_call") {
			finishReason = "stop"
		}
		writeOpenAIStreamData(w, flusher, map[string]any{
			"id": responseID, "object": "chat.completion.chunk", "created": created, "model": publicModel,
			"choices": []any{map[string]any{"index": choiceIndex, "delta": map[string]any{}, "finish_reason": finishReason}},
		})
	}
	if filteredToolCalls > 0 {
		log.Printf("filtered undeclared upstream streaming chat tool calls: request_id=%s count=%d", middleware.GetReqID(r.Context()), filteredToolCalls)
	}
	if includeUsage {
		writeOpenAIStreamData(w, flusher, map[string]any{
			"id": responseID, "object": "chat.completion.chunk", "created": created,
			"model": publicModel, "choices": []any{}, "usage": openAIUsage(usage),
		})
	}
	writeOpenAIStreamData(w, flusher, "[DONE]")
}

func openAIUsage(usage openai.CompletionUsage) map[string]any {
	result := map[string]any{
		"prompt_tokens": usage.PromptTokens, "completion_tokens": usage.CompletionTokens, "total_tokens": usage.TotalTokens,
	}
	promptDetails := map[string]any{}
	if usage.PromptTokensDetails.CachedTokens > 0 {
		promptDetails["cached_tokens"] = usage.PromptTokensDetails.CachedTokens
	}
	if usage.PromptTokensDetails.AudioTokens > 0 {
		promptDetails["audio_tokens"] = usage.PromptTokensDetails.AudioTokens
	}
	if len(promptDetails) > 0 {
		result["prompt_tokens_details"] = promptDetails
	}
	completionDetails := map[string]any{}
	if usage.CompletionTokensDetails.ReasoningTokens > 0 {
		completionDetails["reasoning_tokens"] = usage.CompletionTokensDetails.ReasoningTokens
	}
	if usage.CompletionTokensDetails.AudioTokens > 0 {
		completionDetails["audio_tokens"] = usage.CompletionTokensDetails.AudioTokens
	}
	if usage.CompletionTokensDetails.AcceptedPredictionTokens > 0 {
		completionDetails["accepted_prediction_tokens"] = usage.CompletionTokensDetails.AcceptedPredictionTokens
	}
	if usage.CompletionTokensDetails.RejectedPredictionTokens > 0 {
		completionDetails["rejected_prediction_tokens"] = usage.CompletionTokensDetails.RejectedPredictionTokens
	}
	if len(completionDetails) > 0 {
		result["completion_tokens_details"] = completionDetails
	}
	return result
}

func standardJSON(value any) any {
	data, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	var result any
	if json.Unmarshal(data, &result) != nil {
		return nil
	}
	return result
}

func (h *inferenceHandler) writeOpenAIUpstreamError(w http.ResponseWriter, r *http.Request, err error) {
	status, kind, message := http.StatusBadGateway, "api_error", "upstream request failed"
	if apiErr, ok := errors.AsType[*openai.Error](err); ok {
		status = apiErr.StatusCode
		message = apiErr.Message
		if message == "" {
			message = http.StatusText(status)
		}
		kind = apiErr.Type
		if kind == "" {
			kind = "api_error"
		}
		if status < 400 || status > 599 {
			status = http.StatusBadGateway
		}
	}
	if err != nil {
		log.Printf("openai upstream failed: request_id=%s error=%T", middleware.GetReqID(r.Context()), err)
	}
	writeOpenAIError(w, status, kind, message)
}

func openAIErrorBody(err error) map[string]any {
	kind, message := "api_error", "upstream stream failed"
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		if apiErr.Type != "" {
			kind = apiErr.Type
		}
		if apiErr.Message != "" {
			message = apiErr.Message
		}
	}
	return map[string]any{"type": kind, "message": message, "code": kind, "param": nil}
}

func writeOpenAIStreamData(w http.ResponseWriter, flusher http.Flusher, value any) {
	if value == "[DONE]" {
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}
	data, err := json.Marshal(value)
	if err != nil {
		return
	}
	_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}
