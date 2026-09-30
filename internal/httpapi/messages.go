package httpapi

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"maps"
	mathrand "math/rand/v2"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/go-chi/chi/v5/middleware"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/tiktoken-go/tokenizer"

	"ngon.tech/internal/config"
)

const streamPingInterval = 10 * time.Second

type inferenceHandler struct {
	models     map[string]config.Model
	config     config.Config
	tokenCodec tokenizer.Codec
}

type upstreamRoute struct {
	client   openai.Client
	provider string
	model    string
}

type streamStart struct {
	stream *openAIStream
	first  openai.ChatCompletionChunk
	err    error
}

type openAIStream struct {
	next    func() bool
	current func() openai.ChatCompletionChunk
	err     func() error
	close   func() error
}

type streamToolCall struct {
	id        string
	name      string
	arguments strings.Builder
}

func newInferenceHandler(cfg config.Config) *inferenceHandler {
	codec, err := tokenizer.Get(tokenizer.O200kBase)
	if err != nil {
		panic(fmt.Sprintf("initialize message token counter: %v", err))
	}
	return &inferenceHandler{models: cfg.Models, config: cfg, tokenCodec: codec}
}

func (h *inferenceHandler) ServeMessageTokenCount(w http.ResponseWriter, r *http.Request) {
	var request map[string]any
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "request body must be valid JSON", middleware.GetReqID(r.Context()))
		return
	}

	publicModel, _ := request["model"].(string)
	model, ok := h.models[publicModel]
	if !ok {
		log.Printf("count tokens model not found: request_id=%s model=%q", middleware.GetReqID(r.Context()), publicModel)
		writeAnthropicError(w, http.StatusNotFound, "not_found_error", "model not found", middleware.GetReqID(r.Context()))
		return
	}

	translated, _ := translateAnthropicRequest(request, publicModel, model)
	prompt := map[string]any{"messages": translated["messages"]}
	for _, field := range []string{"tools", "tool_choice", "response_format"} {
		if value, exists := translated[field]; exists {
			prompt[field] = value
		}
	}
	for _, field := range []string{"thinking", "output_config"} {
		if value, exists := request[field]; exists {
			prompt[field] = value
		}
	}

	mediaTokens := 0
	var scrubMedia func(any) any
	scrubMedia = func(value any) any {
		switch typed := value.(type) {
		case []any:
			result := make([]any, len(typed))
			for index, item := range typed {
				result[index] = scrubMedia(item)
			}
			return result
		case map[string]any:
			result := make(map[string]any, len(typed))
			for key, item := range typed {
				result[key] = scrubMedia(item)
			}
			return result
		case string:
			comma := strings.IndexByte(typed, ',')
			if comma < 0 || !strings.Contains(typed[:comma], ";base64") {
				return typed
			}
			decoded, err := base64.StdEncoding.DecodeString(typed[comma+1:])
			if err != nil {
				return "[media]"
			}
			switch {
			case strings.HasPrefix(typed, "data:image/"):
				visualTokens := 256
				if dimensions, _, err := image.DecodeConfig(bytes.NewReader(decoded)); err == nil {
					visualTokens = max(1, ((dimensions.Width+27)/28)*((dimensions.Height+27)/28))
					visualTokens = min(visualTokens, 1568)
				}
				mediaTokens += visualTokens
				return "[image]"
			case strings.HasPrefix(typed, "data:application/pdf"):
				pages := max(1, bytes.Count(decoded, []byte("/Type /Page"))-bytes.Count(decoded, []byte("/Type /Pages")))
				mediaTokens += pages * 2000
				return "[pdf]"
			default:
				return typed
			}
		default:
			return value
		}
	}

	canonical, err := json.Marshal(scrubMedia(prompt))
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "request cannot be counted", middleware.GetReqID(r.Context()))
		return
	}
	textTokens, err := h.tokenCodec.Count(string(canonical))
	if err != nil {
		writeAnthropicError(w, http.StatusInternalServerError, "api_error", "token counting failed", middleware.GetReqID(r.Context()))
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"input_tokens": max(1, textTokens+mediaTokens+8)})
}

func (h *inferenceHandler) ServeMessages(w http.ResponseWriter, r *http.Request) {
	var request map[string]any
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "request body must be valid JSON", middleware.GetReqID(r.Context()))
		return
	}

	publicModel, _ := request["model"].(string)
	model, ok := h.models[publicModel]
	if !ok {
		log.Printf("messages model not found: request_id=%s model=%q", middleware.GetReqID(r.Context()), publicModel)
		writeAnthropicError(w, http.StatusNotFound, "not_found_error", "model not found", middleware.GetReqID(r.Context()))
		return
	}

	upstreamBody, allowedTools := translateAnthropicRequest(request, publicModel, model)
	routes := h.routes(model)
	stream, _ := request["stream"].(bool)
	if stream {
		h.serveMessagesStream(w, r, publicModel, routes, upstreamBody, allowedTools)
		return
	}
	h.serveMessagesResponse(w, r, publicModel, routes, upstreamBody, allowedTools)
}

func (h *inferenceHandler) routes(model config.Model) []upstreamRoute {
	routes := slices.Clone(model.Routes)
	slices.SortStableFunc(routes, func(a, b config.Route) int { return a.Priority - b.Priority })
	for start := 0; start < len(routes); {
		end := start + 1
		for end < len(routes) && routes[end].Priority == routes[start].Priority {
			end++
		}
		for position := start; position < end; position++ {
			totalWeight := 0
			for index := position; index < end; index++ {
				totalWeight += routes[index].Weight
			}
			pick := mathrand.IntN(totalWeight)
			selected := position
			for index := position; index < end; index++ {
				pick -= routes[index].Weight
				if pick < 0 {
					selected = index
					break
				}
			}
			routes[position], routes[selected] = routes[selected], routes[position]
		}
		start = end
	}
	result := make([]upstreamRoute, 0, len(routes))
	for _, route := range routes {
		provider := h.config.Providers[route.Provider]
		// The official SDK supports OpenAI-compatible gateways through a custom base URL.
		// Source: https://github.com/openai/openai-go#requestoptions
		result = append(result, upstreamRoute{
			client: openai.NewClient(
				option.WithAPIKey(provider.APIKey),
				option.WithBaseURL(strings.TrimRight(provider.BaseURL, "/")+"/"),
				option.WithMaxRetries(0),
			),
			provider: route.Provider,
			model:    route.Model,
		})
	}
	return result
}

func translateAnthropicRequest(request map[string]any, publicModel string, model config.Model) (map[string]any, map[string]struct{}) {
	result := make(map[string]any)
	allowedTools := make(map[string]struct{})
	for _, field := range []string{"temperature", "top_p"} {
		if value, ok := request[field]; ok {
			result[field] = value
		}
	}
	if value, ok := request["max_tokens"]; ok {
		result["max_tokens"] = value
	}
	if value, ok := request["stop_sequences"]; ok {
		result["stop"] = value
	}
	if metadata, ok := request["metadata"].(map[string]any); ok {
		if userID, ok := metadata["user_id"].(string); ok {
			result["user"] = userID
		}
	}

	messages := make([]any, 0)
	if system, ok := request["system"]; ok {
		messages = append(messages, map[string]any{"role": "system", "content": translateSystemContent(system)})
	}
	if systemPrompt := strings.TrimSpace(model.SystemPrompt); systemPrompt != "" {
		systemPrompt = strings.NewReplacer("{{model}}", publicModel, "{{owner}}", model.OwnedBy).Replace(systemPrompt)
		// Keep the gateway-controlled policy after the client's system content so it
		// has the final system-message position before the conversation starts.
		messages = append(messages, map[string]any{"role": "system", "content": systemPrompt})
	}
	if sourceMessages, ok := request["messages"].([]any); ok {
		for _, rawMessage := range sourceMessages {
			message, ok := rawMessage.(map[string]any)
			if !ok {
				continue
			}
			messages = append(messages, translateMessage(message)...)
		}
	}
	result["messages"] = messages

	if sourceTools, ok := request["tools"].([]any); ok {
		tools := make([]any, 0, len(sourceTools))
		for _, rawTool := range sourceTools {
			tool, ok := rawTool.(map[string]any)
			if !ok {
				continue
			}
			name, _ := tool["name"].(string)
			if name != "" {
				allowedTools[name] = struct{}{}
			}
			function := map[string]any{"name": tool["name"], "parameters": tool["input_schema"]}
			if description, ok := tool["description"]; ok {
				function["description"] = description
			}
			if strict, ok := tool["strict"]; ok {
				function["strict"] = strict
			}
			tools = append(tools, map[string]any{"type": "function", "function": function})
		}
		result["tools"] = tools
	}
	if choice, ok := request["tool_choice"].(map[string]any); ok {
		switch choice["type"] {
		case "none", "auto":
			result["tool_choice"] = choice["type"]
		case "any":
			result["tool_choice"] = "required"
		case "tool":
			result["tool_choice"] = map[string]any{"type": "function", "function": map[string]any{"name": choice["name"]}}
		}
		if disabled, ok := choice["disable_parallel_tool_use"].(bool); ok && disabled {
			result["parallel_tool_calls"] = false
		}
	}
	if output, ok := request["output_config"].(map[string]any); ok {
		if format, ok := output["format"].(map[string]any); ok && format["type"] == "json_schema" {
			result["response_format"] = map[string]any{
				"type":        "json_schema",
				"json_schema": map[string]any{"name": "response", "schema": format["schema"], "strict": true},
			}
		}
	}
	return result, allowedTools
}

func translateSystemContent(value any) any {
	if text, ok := value.(string); ok {
		return text
	}
	blocks, ok := value.([]any)
	if !ok {
		return value
	}
	parts := make([]any, 0, len(blocks))
	for _, rawBlock := range blocks {
		block, ok := rawBlock.(map[string]any)
		if ok && block["type"] == "text" {
			parts = append(parts, map[string]any{"type": "text", "text": block["text"]})
		}
	}
	return parts
}

func translateMessage(message map[string]any) []any {
	role, _ := message["role"].(string)
	content := message["content"]
	if text, ok := content.(string); ok {
		return []any{map[string]any{"role": role, "content": text}}
	}
	blocks, ok := content.([]any)
	if !ok {
		return []any{map[string]any{"role": role, "content": content}}
	}

	if role == "assistant" {
		parts := make([]any, 0, len(blocks))
		toolCalls := make([]any, 0)
		for _, rawBlock := range blocks {
			block, ok := rawBlock.(map[string]any)
			if !ok {
				continue
			}
			switch block["type"] {
			case "text":
				parts = append(parts, map[string]any{"type": "text", "text": block["text"]})
			case "tool_use":
				arguments, _ := json.Marshal(block["input"])
				toolCalls = append(toolCalls, map[string]any{
					"id": block["id"], "type": "function",
					"function": map[string]any{"name": block["name"], "arguments": string(arguments)},
				})
			}
		}
		translated := map[string]any{"role": "assistant", "content": parts}
		if len(parts) == 0 {
			translated["content"] = nil
		}
		if len(toolCalls) > 0 {
			translated["tool_calls"] = toolCalls
		}
		return []any{translated}
	}

	result := make([]any, 0, len(blocks)+1)
	userParts := make([]any, 0, len(blocks))
	flushUserParts := func() {
		if len(userParts) > 0 {
			result = append(result, map[string]any{"role": "user", "content": userParts})
			userParts = nil
		}
	}
	for _, rawBlock := range blocks {
		block, ok := rawBlock.(map[string]any)
		if !ok {
			continue
		}
		if block["type"] == "tool_result" {
			flushUserParts()
			result = append(result, map[string]any{
				"role": "tool", "tool_call_id": block["tool_use_id"],
				"content": translateToolResult(block["content"]),
			})
			continue
		}
		if part := translateContentBlock(block); part != nil {
			userParts = append(userParts, part)
		}
	}
	flushUserParts()
	if len(result) == 0 {
		return []any{map[string]any{"role": "user", "content": ""}}
	}
	return result
}

func translateToolResult(value any) any {
	if text, ok := value.(string); ok {
		return text
	}
	blocks, ok := value.([]any)
	if !ok {
		return value
	}
	parts := make([]any, 0, len(blocks))
	for _, rawBlock := range blocks {
		if block, ok := rawBlock.(map[string]any); ok {
			if part := translateContentBlock(block); part != nil {
				parts = append(parts, part)
			}
		}
	}
	return parts
}

func translateContentBlock(block map[string]any) any {
	switch block["type"] {
	case "text":
		return map[string]any{"type": "text", "text": block["text"]}
	case "image":
		source, _ := block["source"].(map[string]any)
		url := ""
		switch source["type"] {
		case "base64":
			url = fmt.Sprintf("data:%v;base64,%v", source["media_type"], source["data"])
		case "url":
			url, _ = source["url"].(string)
		case "file":
			return map[string]any{"type": "file", "file": map[string]any{"file_id": source["file_id"]}}
		}
		if url != "" {
			return map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}}
		}
	case "document":
		source, _ := block["source"].(map[string]any)
		switch source["type"] {
		case "base64":
			mediaType, _ := source["media_type"].(string)
			if strings.HasPrefix(mediaType, "text/") {
				data, _ := source["data"].(string)
				if decoded, err := base64.StdEncoding.DecodeString(data); err == nil {
					data = string(decoded)
				}
				return map[string]any{"type": "text", "text": data}
			}
			filename, _ := block["title"].(string)
			if filename == "" {
				filename = "document.pdf"
			}
			return map[string]any{"type": "file", "file": map[string]any{
				"filename":  filename,
				"file_data": fmt.Sprintf("data:%s;base64,%v", mediaType, source["data"]),
			}}
		case "file":
			return map[string]any{"type": "file", "file": map[string]any{"file_id": source["file_id"]}}
		case "text":
			return map[string]any{"type": "text", "text": source["data"]}
		case "content":
			return map[string]any{"type": "text", "text": source["content"]}
		}
	}
	return nil
}

func (h *inferenceHandler) serveMessagesResponse(w http.ResponseWriter, r *http.Request, publicModel string, routes []upstreamRoute, body map[string]any, allowedTools map[string]struct{}) {
	var lastErr error
	for _, route := range routes {
		params := openai.ChatCompletionNewParams{Model: route.model, Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("")}}
		params.SetExtraFields(body)
		completion, err := route.client.Chat.Completions.New(r.Context(), params)
		if err != nil {
			status := 0
			if apiErr, ok := errors.AsType[*openai.Error](err); ok {
				status = apiErr.StatusCode
			}
			log.Printf("messages upstream route failed: request_id=%s provider=%q model=%q status=%d error=%T", middleware.GetReqID(r.Context()), route.provider, route.model, status, err)
			lastErr = err
			continue
		}
		if len(completion.Choices) == 0 {
			lastErr = errors.New("upstream returned no choices")
			continue
		}
		choice := completion.Choices[0]
		content := make([]any, 0, len(choice.Message.ToolCalls)+1)
		if choice.Message.Content != "" {
			content = append(content, map[string]any{"type": "text", "text": choice.Message.Content})
		}
		acceptedToolCalls := 0
		filteredToolCalls := 0
		for _, call := range choice.Message.ToolCalls {
			name, arguments := call.Function.Name, call.Function.Arguments
			if call.Type == "custom" {
				name, arguments = call.Custom.Name, call.Custom.Input
			}
			if _, allowed := allowedTools[name]; !allowed {
				filteredToolCalls++
				continue
			}
			var input any = map[string]any{}
			_ = json.Unmarshal([]byte(arguments), &input)
			content = append(content, map[string]any{"type": "tool_use", "id": call.ID, "name": name, "input": input})
			acceptedToolCalls++
		}
		if filteredToolCalls > 0 {
			log.Printf("filtered undeclared upstream tool calls: request_id=%s count=%d", middleware.GetReqID(r.Context()), filteredToolCalls)
		}
		finishReason := choice.FinishReason
		if acceptedToolCalls == 0 && (finishReason == "tool_calls" || finishReason == "function_call") {
			finishReason = "stop"
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"id": newMessageID(), "type": "message", "role": "assistant", "content": content,
			"model": publicModel, "stop_reason": anthropicStopReason(finishReason), "stop_sequence": nil,
			"usage": anthropicUsage(completion.Usage),
		})
		return
	}
	h.writeMessagesUpstreamError(w, r, lastErr)
}

func (h *inferenceHandler) serveMessagesStream(w http.ResponseWriter, r *http.Request, publicModel string, routes []upstreamRoute, body map[string]any, allowedTools map[string]struct{}) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeAnthropicError(w, http.StatusInternalServerError, "api_error", "streaming is unavailable", middleware.GetReqID(r.Context()))
		return
	}
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
			streamBody := make(map[string]any, len(body)+1)
			maps.Copy(streamBody, body)
			streamBody["stream_options"] = map[string]any{"include_usage": true}
			params.SetExtraFields(streamBody)
			sdkStream := route.client.Chat.Completions.NewStreaming(r.Context(), params)
			if !sdkStream.Next() {
				lastErr = sdkStream.Err()
				_ = sdkStream.Close()
				continue
			}
			start <- streamStart{
				first:  sdkStream.Current(),
				stream: &openAIStream{next: sdkStream.Next, current: sdkStream.Current, err: sdkStream.Err, close: sdkStream.Close},
			}
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
			writeSSE(w, flusher, "ping", map[string]any{"type": "ping"})
		case started = <-start:
			if started.err != nil || started.stream == nil {
				writeSSE(w, flusher, "error", anthropicStreamError(started.err))
				return
			}
			goto streamReady
		}
	}

streamReady:
	defer func() { _ = started.stream.close() }()
	messageID := newMessageID()
	// Anthropic streams have a strict message/block/delta/stop event order.
	// Source: https://platform.claude.com/docs/en/build-with-claude/streaming#full-http-stream-response
	writeSSE(w, flusher, "message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": messageID, "type": "message", "role": "assistant", "content": []any{}, "model": publicModel,
			"stop_reason": nil, "stop_sequence": nil, "usage": map[string]any{"input_tokens": int64(0), "output_tokens": int64(0)},
		},
	})

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

	blockIndex := 0
	textOpen := false
	finishReason := "stop"
	inputTokens := int64(0)
	outputTokens := int64(0)
	cacheReadTokens := int64(0)
	cacheWriteTokens := int64(0)
	tools := map[int64]*streamToolCall{}
	toolOrder := make([]int64, 0)
	for {
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
			writeSSE(w, flusher, "ping", map[string]any{"type": "ping"})
		case chunk, ok := <-chunks:
			if !ok {
				streamErr := <-streamDone
				if streamErr != nil {
					writeSSE(w, flusher, "error", anthropicStreamError(streamErr))
					return
				}
				goto streamFinished
			}
			if chunk.Usage.JSON.PromptTokens.Valid() || chunk.Usage.PromptTokens > 0 {
				inputTokens = chunk.Usage.PromptTokens
				outputTokens = chunk.Usage.CompletionTokens
				cacheReadTokens = chunk.Usage.PromptTokensDetails.CachedTokens
				cacheWriteTokens = chunk.Usage.PromptTokensDetails.CacheWriteTokens
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			choice := chunk.Choices[0]
			if choice.FinishReason != "" {
				finishReason = choice.FinishReason
			}
			if choice.Delta.Content != "" {
				if !textOpen {
					writeSSE(w, flusher, "content_block_start", map[string]any{"type": "content_block_start", "index": blockIndex, "content_block": map[string]any{"type": "text", "text": ""}})
					textOpen = true
				}
				writeSSE(w, flusher, "content_block_delta", map[string]any{"type": "content_block_delta", "index": blockIndex, "delta": map[string]any{"type": "text_delta", "text": choice.Delta.Content}})
			}
			for _, delta := range choice.Delta.ToolCalls {
				tool := tools[delta.Index]
				if tool == nil {
					tool = &streamToolCall{}
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
	if textOpen {
		writeSSE(w, flusher, "content_block_stop", map[string]any{"type": "content_block_stop", "index": blockIndex})
		blockIndex++
	}
	slices.Sort(toolOrder)
	acceptedToolCalls := 0
	filteredToolCalls := 0
	for _, index := range toolOrder {
		tool := tools[index]
		if _, allowed := allowedTools[tool.name]; !allowed {
			filteredToolCalls++
			continue
		}
		writeSSE(w, flusher, "content_block_start", map[string]any{
			"type": "content_block_start", "index": blockIndex,
			"content_block": map[string]any{"type": "tool_use", "id": tool.id, "name": tool.name, "input": map[string]any{}},
		})
		writeSSE(w, flusher, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": blockIndex,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": tool.arguments.String()},
		})
		writeSSE(w, flusher, "content_block_stop", map[string]any{"type": "content_block_stop", "index": blockIndex})
		blockIndex++
		acceptedToolCalls++
	}
	if filteredToolCalls > 0 {
		log.Printf("filtered undeclared upstream streaming tool calls: request_id=%s count=%d", middleware.GetReqID(r.Context()), filteredToolCalls)
	}
	if acceptedToolCalls == 0 && (finishReason == "tool_calls" || finishReason == "function_call") {
		finishReason = "stop"
	}
	usage := map[string]any{"input_tokens": max(0, inputTokens-cacheReadTokens-cacheWriteTokens), "output_tokens": outputTokens}
	if cacheReadTokens > 0 {
		usage["cache_read_input_tokens"] = cacheReadTokens
	}
	if cacheWriteTokens > 0 {
		usage["cache_creation_input_tokens"] = cacheWriteTokens
	}
	writeSSE(w, flusher, "message_delta", map[string]any{
		"type": "message_delta", "delta": map[string]any{"stop_reason": anthropicStopReason(finishReason), "stop_sequence": nil},
		"usage": usage,
	})
	writeSSE(w, flusher, "message_stop", map[string]any{"type": "message_stop"})
}

func (h *inferenceHandler) writeMessagesUpstreamError(w http.ResponseWriter, r *http.Request, err error) {
	status, kind, message := http.StatusBadGateway, "api_error", "upstream request failed"
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		status = apiErr.StatusCode
		message = apiErr.Message
		if message == "" {
			message = http.StatusText(status)
		}
		switch status {
		case http.StatusBadRequest, http.StatusUnprocessableEntity:
			kind = "invalid_request_error"
		case http.StatusUnauthorized:
			kind = "authentication_error"
		case http.StatusForbidden:
			kind = "permission_error"
		case http.StatusNotFound:
			kind = "not_found_error"
		case http.StatusRequestEntityTooLarge:
			kind = "request_too_large"
		case http.StatusTooManyRequests:
			kind = "rate_limit_error"
		case http.StatusServiceUnavailable, 529:
			kind = "overloaded_error"
		default:
			if status < 400 || status > 599 {
				status = http.StatusBadGateway
			}
		}
	}
	if err != nil {
		log.Printf("messages upstream failed: request_id=%s error=%T", middleware.GetReqID(r.Context()), err)
	}
	writeAnthropicError(w, status, kind, message, middleware.GetReqID(r.Context()))
}

func anthropicStopReason(reason string) string {
	switch reason {
	case "length":
		return "max_tokens"
	case "tool_calls", "function_call":
		return "tool_use"
	case "content_filter":
		return "refusal"
	default:
		return "end_turn"
	}
}

func anthropicUsage(usage openai.CompletionUsage) map[string]any {
	cacheRead := usage.PromptTokensDetails.CachedTokens
	cacheWrite := usage.PromptTokensDetails.CacheWriteTokens
	result := map[string]any{
		"input_tokens":  max(0, usage.PromptTokens-cacheRead-cacheWrite),
		"output_tokens": usage.CompletionTokens,
	}
	if cacheRead > 0 {
		result["cache_read_input_tokens"] = cacheRead
	}
	if cacheWrite > 0 {
		result["cache_creation_input_tokens"] = cacheWrite
	}
	return result
}

func anthropicStreamError(err error) map[string]any {
	message := "upstream stream failed"
	if err != nil {
		var apiErr *openai.Error
		if errors.As(err, &apiErr) && apiErr.Message != "" {
			message = apiErr.Message
		}
	}
	return map[string]any{"type": "error", "error": map[string]any{"type": "api_error", "message": message}}
}

func writeSSE(w http.ResponseWriter, flusher http.Flusher, event string, value any) {
	data, err := json.Marshal(value)
	if err != nil {
		return
	}
	_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
	flusher.Flush()
}

func newMessageID() string {
	bytes := make([]byte, 12)
	if _, err := rand.Read(bytes); err != nil {
		return fmt.Sprintf("msg_%d", time.Now().UnixNano())
	}
	return "msg_" + hex.EncodeToString(bytes)
}
