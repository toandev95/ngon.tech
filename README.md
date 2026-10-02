# ngon.tech

A lightweight Go gateway that exposes Anthropic and OpenAI-compatible APIs while routing every request to OpenAI-compatible upstream providers.

## Features

- Anthropic Messages compatibility for Claude Code, Claude Desktop, and OpenCode.
- OpenAI Chat Completions and stateless Responses compatibility for Codex CLI and OpenAI SDKs.
- JSON and SSE streaming, function tools, reasoning, structured output, vision, file input, and image generation.
- Public model aliases with weighted routing and priority-based fallback.
- Optional gateway-managed system prompts with `{{model}}` and `{{owner}}` placeholders.
- Sanitized responses that hide upstream URLs, credentials, model names, and provider-specific metadata.
- No database required.

## Quick start

Requires Go 1.26 or later.

```powershell
Copy-Item config.example.toml config.toml
go run ./cmd/gateway -config config.toml
```

The server listens on `:8080` by default. Configuration is loaded once at startup; restart the process after editing it.

The config path can be overridden with `-config` or `CONFIG_FILE`. The CLI flag takes precedence.

## Portable Windows build

```powershell
go build -trimpath -ldflags="-s -w" -o ngon-gateway.exe ./cmd/gateway
```

Copy these two files to the target machine:

```text
ngon-gateway.exe
config.toml
```

Double-click the executable to start the gateway. It automatically loads `config.toml` from the executable's directory, even when launched through a shortcut or from another working directory. Go does not need to be installed on the target machine.

## Configuration

All upstream providers use the OpenAI protocol. Text routes must expose `/v1/chat/completions`; image routes must expose `/v1/images/generations`.

```toml
[server]
address = ":8080"

[providers.primary]
protocol = "openai"
base_url = "https://provider.example.com/v1"
api_key = "YOUR_API_KEY"

[models."claude-opus-5.5"]
display_name = "Opus 5.5"
owned_by = "ngon.tech"
system_prompt = "You are {{model}}, operated by {{owner}}."
routes = [
  { provider = "primary", model = "upstream-model", priority = 0, weight = 1 },
]

[models.gpt-image-2]
display_name = "GPT Image 2"
routes = [
  { provider = "primary", model = "gpt-image-2" },
]
```

Lower `priority` routes are tried first. Routes with the same priority are selected by `weight`. Clients only see the public model ID and metadata.

## Endpoints

| Endpoint | Compatibility |
| --- | --- |
| `GET /v1/models` | OpenAI by default; Anthropic when the User-Agent contains `anthropic` or `claude` |
| `POST /v1/messages` | Anthropic Messages, including tools, vision/files, reasoning, structured output, and SSE |
| `POST /v1/messages/count_tokens` | Local token estimate without calling an upstream model |
| `POST /v1/chat/completions` | OpenAI Chat Completions with permissive payload forwarding and SSE |
| `POST /v1/responses` | Stateless OpenAI Responses with semantic SSE and function tools |
| `POST /v1/images/generations` | OpenAI Image Generations with permissive payload forwarding |

Responses are rebuilt from protocol-standard fields. Upstream credentials, URLs, private model names, and unknown provider metadata are not returned to clients. Tool calls are only returned when the client declared the matching tool in the current request.

## Limitations

- `/v1/responses` does not support `previous_response_id`, Conversations, or background mode. Clients must replay the conversation input.
- Only client-defined function tools can be forwarded through the Chat Completions upstream. Hosted and namespace tools are ignored.
- `/v1/messages/count_tokens` is an estimate based on `o200k_base`; it is not authoritative billing usage.
- The gateway does not currently authenticate incoming clients. Bind it to a trusted interface or place authentication in front of it before exposing it publicly.

## Development

```sh
go test ./...
go vet ./...
go mod verify
```
