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

type responseToolCall struct {
	id        string
	name      string
	arguments strings.Builder
}

func (h *inferenceHandler) ServeResponses(w http.ResponseWriter, r *http.Request) {
	var request map[string]any
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "request body must be valid JSON")
		return
	}
	if previousID, _ := request["previous_response_id"].(string); previousID != "" || request["conversation"] != nil {
		writeOpenAIError(w, http.StatusBadRequest, "unsupported_parameter", "this gateway supports stateless Responses input, not previous_response_id or conversation")
		return
	}
	if background, _ := request["background"].(bool); background {
		writeOpenAIError(w, http.StatusBadRequest, "unsupported_parameter", "background responses are not supported")
		return
	}

	publicModel, _ := request["model"].(string)
	model, ok := h.models[publicModel]
	if !ok {
		log.Printf("responses model not found: request_id=%s model=%q", middleware.GetReqID(r.Context()), publicModel)
		writeOpenAIError(w, http.StatusNotFound, "model_not_found", "model not found")
		return
	}

	body, allowedTools := translateResponsesRequest(request, publicModel, model)
	stream, _ := request["stream"].(bool)
	if stream {
		h.serveResponsesStream(w, r, request, publicModel, h.routes(model), body, allowedTools)
		return
	}
	h.serveResponse(w, r, request, publicModel, h.routes(model), body, allowedTools)
}

func translateResponsesRequest(request map[string]any, publicModel string, model config.Model) (map[string]any, map[string]struct{}) {
	chatRequest := map[string]any{"model": publicModel}
	for _, field := range []string{"temperature", "top_p", "parallel_tool_calls", "service_tier", "prompt_cache_key", "safety_identifier", "user"} {
		if value, ok := request[field]; ok {
			chatRequest[field] = value
		}
	}
	if value, ok := request["max_output_tokens"]; ok {
		chatRequest["max_completion_tokens"] = value
	}
	if reasoning, ok := request["reasoning"].(map[string]any); ok {
		if effort, ok := reasoning["effort"]; ok {
			chatRequest["reasoning_effort"] = effort
		}
	}
	if text, ok := request["text"].(map[string]any); ok {
		if verbosity, ok := text["verbosity"]; ok {
			chatRequest["verbosity"] = verbosity
		}
		if format, ok := text["format"].(map[string]any); ok {
			switch format["type"] {
			case "json_object":
				chatRequest["response_format"] = map[string]any{"type": "json_object"}
			case "json_schema":
				chatRequest["response_format"] = map[string]any{"type": "json_schema", "json_schema": map[string]any{
					"name": format["name"], "description": format["description"], "schema": format["schema"], "strict": format["strict"],
				}}
			}
		}
	}

	messages := make([]any, 0)
	if instructions, ok := request["instructions"].(string); ok && instructions != "" {
		messages = append(messages, map[string]any{"role": "developer", "content": instructions})
	}
	appendInput := func(rawInput any) {
		inputs, ok := rawInput.([]any)
		if !ok {
			if text, ok := rawInput.(string); ok {
				messages = append(messages, map[string]any{"role": "user", "content": text})
			}
			return
		}
		pendingCalls := make([]any, 0)
		flushCalls := func() {
			if len(pendingCalls) > 0 {
				messages = append(messages, map[string]any{"role": "assistant", "content": nil, "tool_calls": pendingCalls})
				pendingCalls = nil
			}
		}
		for _, rawItem := range inputs {
			item, ok := rawItem.(map[string]any)
			if !ok {
				continue
			}
			switch item["type"] {
			case "function_call":
				pendingCalls = append(pendingCalls, map[string]any{
					"id": item["call_id"], "type": "function",
					"function": map[string]any{"name": item["name"], "arguments": item["arguments"]},
				})
			case "function_call_output":
				flushCalls()
				messages = append(messages, map[string]any{"role": "tool", "tool_call_id": item["call_id"], "content": responseInputContent(item["output"])})
			case "message", nil:
				flushCalls()
				role, _ := item["role"].(string)
				if role != "" {
					messages = append(messages, map[string]any{"role": role, "content": responseInputContent(item["content"])})
				}
			}
		}
		flushCalls()
	}
	appendInput(request["input"])
	chatRequest["messages"] = messages

	if sourceTools, ok := request["tools"].([]any); ok {
		tools := make([]any, 0, len(sourceTools))
		for _, rawTool := range sourceTools {
			tool, ok := rawTool.(map[string]any)
			if !ok || tool["type"] != "function" {
				continue
			}
			name, _ := tool["name"].(string)
			if name == "" {
				continue
			}
			function := map[string]any{"name": name, "parameters": tool["parameters"]}
			for _, field := range []string{"description", "strict"} {
				if value, ok := tool[field]; ok {
					function[field] = value
				}
			}
			tools = append(tools, map[string]any{"type": "function", "function": function})
		}
		if len(tools) > 0 {
			chatRequest["tools"] = tools
		}
	}
	if choice, ok := request["tool_choice"]; ok {
		if object, ok := choice.(map[string]any); ok && object["type"] == "function" {
			chatRequest["tool_choice"] = map[string]any{"type": "function", "function": map[string]any{"name": object["name"]}}
		} else if choice == "none" || choice == "auto" || choice == "required" {
			chatRequest["tool_choice"] = choice
		}
	}
	return prepareChatCompletionRequest(chatRequest, publicModel, model)
}

func responseInputContent(value any) any {
	if text, ok := value.(string); ok {
		return text
	}
	parts, ok := value.([]any)
	if !ok {
		return value
	}
	result := make([]any, 0, len(parts))
	for _, rawPart := range parts {
		part, ok := rawPart.(map[string]any)
		if !ok {
			continue
		}
		switch part["type"] {
		case "input_text", "output_text":
			result = append(result, map[string]any{"type": "text", "text": part["text"]})
		case "input_image":
			if imageURL, ok := part["image_url"].(string); ok && imageURL != "" {
				image := map[string]any{"url": imageURL}
				if detail, ok := part["detail"]; ok {
					image["detail"] = detail
				}
				result = append(result, map[string]any{"type": "image_url", "image_url": image})
			}
		case "input_file":
			file := map[string]any{}
			for _, field := range []string{"file_id", "file_data", "filename"} {
				if item, ok := part[field]; ok {
					file[field] = item
				}
			}
			if len(file) > 0 {
				result = append(result, map[string]any{"type": "file", "file": file})
			}
		}
	}
	return result
}

func (h *inferenceHandler) serveResponse(w http.ResponseWriter, r *http.Request, request map[string]any, publicModel string, routes []upstreamRoute, body map[string]any, allowedTools map[string]struct{}) {
	var lastErr error
	for _, route := range routes {
		params := openai.ChatCompletionNewParams{Model: route.model, Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("")}}
		params.SetExtraFields(body)
		completion, err := route.client.Chat.Completions.New(r.Context(), params)
		if err != nil {
			lastErr = err
			continue
		}
		if len(completion.Choices) == 0 {
			lastErr = errors.New("upstream returned no choices")
			continue
		}
		output, filtered := responseOutput(completion.Choices[0].Message.Content, completion.Choices[0].Message.ToolCalls, allowedTools)
		if filtered > 0 {
			log.Printf("filtered undeclared upstream response tool calls: request_id=%s count=%d", middleware.GetReqID(r.Context()), filtered)
		}
		writeJSON(w, http.StatusOK, completedResponse(responseID(completion.ID), completion.Created, publicModel, request, output, completion.Usage))
		return
	}
	h.writeOpenAIUpstreamError(w, r, lastErr)
}

func responseOutput(text string, calls []openai.ChatCompletionMessageToolCallUnion, allowedTools map[string]struct{}) ([]any, int) {
	output := make([]any, 0, len(calls)+1)
	if text != "" {
		output = append(output, map[string]any{
			"id": responseItemID("msg"), "type": "message", "status": "completed", "role": "assistant",
			"content": []any{map[string]any{"type": "output_text", "text": text, "annotations": []any{}, "logprobs": []any{}}},
		})
	}
	filtered := 0
	for _, call := range calls {
		name, arguments := call.Function.Name, call.Function.Arguments
		if call.Type == "custom" {
			name, arguments = call.Custom.Name, call.Custom.Input
		}
		if _, allowed := allowedTools[name]; !allowed {
			filtered++
			continue
		}
		output = append(output, map[string]any{
			"id": responseItemID("fc"), "type": "function_call", "status": "completed",
			"call_id": call.ID, "name": name, "arguments": arguments,
		})
	}
	return output, filtered
}

func (h *inferenceHandler) serveResponsesStream(w http.ResponseWriter, r *http.Request, request map[string]any, publicModel string, routes []upstreamRoute, body map[string]any, allowedTools map[string]struct{}) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeOpenAIError(w, http.StatusInternalServerError, "api_error", "streaming is unavailable")
		return
	}
	streamBody := maps.Clone(body)
	streamBody["stream_options"] = map[string]any{"include_usage": true}
	responseID := responseItemID("resp")
	created := time.Now().Unix()
	sequence := int64(0)
	base := responseEnvelope(responseID, created, publicModel, request, "in_progress", []any{}, openai.CompletionUsage{})

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	writeResponseEvent(w, flusher, "response.created", map[string]any{"type": "response.created", "sequence_number": sequence, "response": base})
	sequence++
	writeResponseEvent(w, flusher, "response.in_progress", map[string]any{"type": "response.in_progress", "sequence_number": sequence, "response": base})
	sequence++

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
				failed := responseEnvelope(responseID, created, publicModel, request, "failed", []any{}, openai.CompletionUsage{})
				failed["error"] = openAIErrorBody(started.err)
				writeResponseEvent(w, flusher, "response.failed", map[string]any{"type": "response.failed", "sequence_number": sequence, "response": failed})
				return
			}
			goto streamReady
		}
	}

streamReady:
	defer func() { _ = started.stream.close() }()
	chunks := make(chan openai.ChatCompletionChunk, 1)
	done := make(chan error, 1)
	go func() {
		chunks <- started.first
		for started.stream.next() {
			select {
			case chunks <- started.stream.current():
			case <-r.Context().Done():
				return
			}
		}
		done <- started.stream.err()
		close(chunks)
	}()

	messageID := responseItemID("msg")
	textStarted := false
	text := strings.Builder{}
	tools := map[int64]*responseToolCall{}
	toolOrder := make([]int64, 0)
	var usage openai.CompletionUsage
	for {
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
			_, _ = fmt.Fprint(w, ": keep-alive\n\n")
			flusher.Flush()
		case chunk, ok := <-chunks:
			if !ok {
				if streamErr := <-done; streamErr != nil {
					failed := responseEnvelope(responseID, created, publicModel, request, "failed", []any{}, usage)
					failed["error"] = openAIErrorBody(streamErr)
					writeResponseEvent(w, flusher, "response.failed", map[string]any{"type": "response.failed", "sequence_number": sequence, "response": failed})
					return
				}
				goto streamFinished
			}
			if chunk.Usage.JSON.PromptTokens.Valid() || chunk.Usage.PromptTokens > 0 {
				usage = chunk.Usage
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			choice := chunk.Choices[0]
			if choice.Delta.Content != "" {
				if !textStarted {
					item := map[string]any{"id": messageID, "type": "message", "status": "in_progress", "role": "assistant", "content": []any{}}
					writeResponseEvent(w, flusher, "response.output_item.added", map[string]any{"type": "response.output_item.added", "sequence_number": sequence, "output_index": 0, "item": item})
					sequence++
					writeResponseEvent(w, flusher, "response.content_part.added", map[string]any{"type": "response.content_part.added", "sequence_number": sequence, "item_id": messageID, "output_index": 0, "content_index": 0, "part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}, "logprobs": []any{}}})
					sequence++
					textStarted = true
				}
				text.WriteString(choice.Delta.Content)
				writeResponseEvent(w, flusher, "response.output_text.delta", map[string]any{"type": "response.output_text.delta", "sequence_number": sequence, "item_id": messageID, "output_index": 0, "content_index": 0, "delta": choice.Delta.Content, "logprobs": []any{}})
				sequence++
			}
			for _, delta := range choice.Delta.ToolCalls {
				tool := tools[delta.Index]
				if tool == nil {
					tool = &responseToolCall{}
					tools[delta.Index] = tool
					toolOrder = append(toolOrder, delta.Index)
				}
				if delta.ID != "" {
					tool.id = delta.ID
				}
				if delta.Function.Name != "" {
					tool.name = delta.Function.Name
				}
				tool.arguments.WriteString(delta.Function.Arguments)
			}
		}
	}

streamFinished:
	output := make([]any, 0, len(tools)+1)
	outputIndex := int64(0)
	if textStarted {
		part := map[string]any{"type": "output_text", "text": text.String(), "annotations": []any{}, "logprobs": []any{}}
		item := map[string]any{"id": messageID, "type": "message", "status": "completed", "role": "assistant", "content": []any{part}}
		writeResponseEvent(w, flusher, "response.output_text.done", map[string]any{"type": "response.output_text.done", "sequence_number": sequence, "item_id": messageID, "output_index": outputIndex, "content_index": 0, "text": text.String(), "logprobs": []any{}})
		sequence++
		writeResponseEvent(w, flusher, "response.content_part.done", map[string]any{"type": "response.content_part.done", "sequence_number": sequence, "item_id": messageID, "output_index": outputIndex, "content_index": 0, "part": part})
		sequence++
		writeResponseEvent(w, flusher, "response.output_item.done", map[string]any{"type": "response.output_item.done", "sequence_number": sequence, "output_index": outputIndex, "item": item})
		sequence++
		output = append(output, item)
		outputIndex++
	}
	slices.Sort(toolOrder)
	filtered := 0
	for _, index := range toolOrder {
		tool := tools[index]
		if _, allowed := allowedTools[tool.name]; !allowed {
			filtered++
			continue
		}
		itemID := responseItemID("fc")
		added := map[string]any{"id": itemID, "type": "function_call", "status": "in_progress", "call_id": tool.id, "name": tool.name, "arguments": ""}
		writeResponseEvent(w, flusher, "response.output_item.added", map[string]any{"type": "response.output_item.added", "sequence_number": sequence, "output_index": outputIndex, "item": added})
		sequence++
		arguments := tool.arguments.String()
		if arguments != "" {
			writeResponseEvent(w, flusher, "response.function_call_arguments.delta", map[string]any{"type": "response.function_call_arguments.delta", "sequence_number": sequence, "item_id": itemID, "output_index": outputIndex, "delta": arguments})
			sequence++
		}
		writeResponseEvent(w, flusher, "response.function_call_arguments.done", map[string]any{"type": "response.function_call_arguments.done", "sequence_number": sequence, "item_id": itemID, "output_index": outputIndex, "arguments": arguments})
		sequence++
		item := map[string]any{"id": itemID, "type": "function_call", "status": "completed", "call_id": tool.id, "name": tool.name, "arguments": arguments}
		writeResponseEvent(w, flusher, "response.output_item.done", map[string]any{"type": "response.output_item.done", "sequence_number": sequence, "output_index": outputIndex, "item": item})
		sequence++
		output = append(output, item)
		outputIndex++
	}
	if filtered > 0 {
		log.Printf("filtered undeclared upstream streaming response tool calls: request_id=%s count=%d", middleware.GetReqID(r.Context()), filtered)
	}
	completed := completedResponse(responseID, created, publicModel, request, output, usage)
	writeResponseEvent(w, flusher, "response.completed", map[string]any{"type": "response.completed", "sequence_number": sequence, "response": completed})
}

func completedResponse(id string, created int64, publicModel string, request map[string]any, output []any, usage openai.CompletionUsage) map[string]any {
	return responseEnvelope(id, created, publicModel, request, "completed", output, usage)
}

func responseEnvelope(id string, created int64, publicModel string, request map[string]any, status string, output []any, usage openai.CompletionUsage) map[string]any {
	result := map[string]any{
		"id": id, "object": "response", "created_at": created, "status": status, "background": false,
		"error": nil, "incomplete_details": nil, "model": publicModel, "output": output,
		"parallel_tool_calls": true, "previous_response_id": nil,
		"usage": responsesUsage(usage),
	}
	// Do not reflect instructions or the often very large client tool schemas back in
	// every lifecycle event. Consumers need generated output and usage; request policy
	// remains private and upstream/provider details remain hidden.
	if value, ok := request["max_output_tokens"]; ok {
		result["max_output_tokens"] = value
	}
	if value, ok := request["parallel_tool_calls"]; ok {
		result["parallel_tool_calls"] = value
	}
	return result
}

func responsesUsage(usage openai.CompletionUsage) map[string]any {
	return map[string]any{
		"input_tokens": usage.PromptTokens, "input_tokens_details": map[string]any{"cached_tokens": usage.PromptTokensDetails.CachedTokens},
		"output_tokens": usage.CompletionTokens, "output_tokens_details": map[string]any{"reasoning_tokens": usage.CompletionTokensDetails.ReasoningTokens},
		"total_tokens": usage.TotalTokens,
	}
}

func responseID(upstream string) string {
	if strings.HasPrefix(upstream, "resp_") {
		return upstream
	}
	return responseItemID("resp")
}

func responseItemID(prefix string) string {
	return prefix + strings.TrimPrefix(newMessageID(), "msg")
}

func writeResponseEvent(w http.ResponseWriter, flusher http.Flusher, event string, value any) {
	data, err := json.Marshal(value)
	if err != nil {
		return
	}
	_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
	flusher.Flush()
}
