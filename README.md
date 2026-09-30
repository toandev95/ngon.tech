# ngon.tech

API gateway Go: **chi** lo routing/middleware, **BurntSushi/toml** load cấu hình và **OpenAI Go SDK** gọi các provider OpenAI-compatible.

## Chạy

Yêu cầu Go 1.26 trở lên. Từ thư mục project:

```powershell
Copy-Item config.example.toml config.toml
go run ./cmd/gateway
```

Hoặc chạy trực tiếp config mẫu:

```sh
go run ./cmd/gateway -config config.example.toml
```

Cấu hình đọc một lần lúc khởi động, hoàn toàn từ TOML (kể cả API key). Đổi config thì restart. Có thể chọn file bằng `CONFIG_FILE` hoặc flag `-config`; flag có ưu tiên cao hơn.

## Cấu trúc

```text
cmd/gateway/main.go          # Khởi động HTTP server, graceful shutdown
internal/config/config.go   # Schema, load và validate TOML
internal/httpapi/router.go  # Khai báo routes và middleware chung
internal/httpapi/models.go  # Model catalog và pagination
internal/httpapi/messages.go # Anthropic Messages ↔ OpenAI Chat Completions
internal/httpapi/chat_completions.go # OpenAI Chat Completions facade
internal/httpapi/responses.go # Stateless OpenAI Responses facade
config.example.toml         # Mẫu provider và model mapping
```

## Endpoints

| Endpoint | Hành vi |
| --- | --- |
| `GET /v1/models` | User-Agent chứa `anthropic`/`claude` → Anthropic; còn lại (gồm OpenAI hoặc thiếu UA) → OpenAI. Không phân biệt hoa/thường. |
| `POST /v1/messages` | Anthropic Messages-compatible facade; translates requests, responses, tools, images/files, and SSE to/from OpenAI Chat Completions upstreams. |
| `POST /v1/chat/completions` | OpenAI Chat Completions-compatible facade; hỗ trợ JSON, tools, reasoning, vision/files và SSE. |
| `POST /v1/responses` | Stateless OpenAI Responses-compatible facade cho Codex CLI; hỗ trợ JSON, semantic SSE, function tools, vision/files và reasoning. |

Hai format cùng trả toàn bộ model công khai, không lọc theo GPT/Claude hoặc protocol upstream. Catalog được sắp theo `created_at` giảm dần rồi ID. Anthropic hỗ trợ `limit` (mặc định 20, tối đa 1000), `after_id`, `before_id`. Anthropic model items có các field chuẩn `id`, `type`, `display_name`, `created_at`, `capabilities`, `max_input_tokens`, `max_tokens`; các capability/limit chưa được cấu hình trả `null`.

```sh
curl -H "User-Agent: OpenAI/Python" http://localhost:8080/v1/models
curl -H "User-Agent: Anthropic/Python" http://localhost:8080/v1/models
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-opus-5.5","max_tokens":128,"messages":[{"role":"user","content":"Hello"}]}'
```

## Mapping model và provider

`models.<id>` là tên client gọi, ví dụ `claude-opus-5`. Mỗi route ánh xạ sang **provider + tên model upstream**. Client chỉ thấy ID/metadata công khai; URL, key và tên model upstream không có trong catalog.

- Mọi provider phải hỗ trợ OpenAI-compatible `POST /v1/chat/completions`; `protocol` có thể bỏ qua hoặc đặt là `openai`.
- `base_url`: API base gồm prefix, ví dụ `https://host/v1`.
- `routes`: một hoặc nhiều upstream cho cùng model công khai.
- `priority`: tier nhỏ hơn được ưu tiên; tier lớn hơn dành cho fallback.
- `weight`: tỷ trọng giữa các route cùng tier; mặc định 1. Priority mặc định 0.
- `display_name` mặc định bằng ID; `owned_by` mặc định `ngon.tech`; `created_at` mặc định Unix epoch khi không khai báo.
- `system_prompt`: policy do gateway quản lý, được chèn sau system prompt của client và trước conversation. Hỗ trợ `{{model}}` (public model ID) và `{{owner}}`.

`/v1/messages` giữ model ID công khai ở response nhưng thay bằng model route khi gọi upstream. System prompt, content blocks, tool definitions/results, tool choice, stop sequences, structured output, ảnh base64/URL, document text/base64/file ID và SSE được chuyển đổi giữa hai protocol. Response chỉ lộ Anthropic message/error fields cần thiết; thông tin provider, URL, key và payload thừa từ upstream không được trả ra client.

`/v1/chat/completions` nhận payload OpenAI-compatible theo kiểu permissive và chuyển tiếp các field client gửi, chỉ thay public model bằng route upstream và chèn `system_prompt`. Response JSON/SSE được dựng lại từ các field Chat Completions chuẩn, trả public model ID và loại metadata riêng của provider. `reasoning_effort` được chuyển tiếp; số reasoning token (nếu upstream cung cấp) nằm trong `usage.completion_tokens_details.reasoning_tokens`.

`/v1/responses` chuyển Responses input/items và semantic SSE sang cùng upstream Chat Completions. Endpoint này cố ý stateless: Codex CLI dùng `store: false` và tự phát lại input items nên không cần database. `previous_response_id`, Conversations API và background mode chưa được hỗ trợ; gateway trả lỗi rõ nếu client yêu cầu các chế độ cần state đó. Built-in/namespace tools không thể chạy ở upstream Chat Completions nên chỉ function tools được chuyển tiếp và allowlist.

Tool call từ upstream chỉ được trả về nếu tên tool có trong `tools` của request hiện tại. Tool do provider tự chèn hoặc bịa ra bị loại ở gateway; nếu không còn tool hợp lệ, `stop_reason` được chuẩn hóa thành `end_turn`. `system_prompt` giúp định hướng identity/policy nhưng không thay thế enforcement bằng code và không thể ghi đè một hidden prompt mà provider tự thêm sau request.

`POST /v1/messages/count_tokens` trả về `{ "input_tokens": number }` mà không gọi model. Vì upstream chỉ hỗ trợ OpenAI Chat Completions, đây là ước lượng local dùng tokenizer `o200k_base`, cộng token ảnh theo kích thước và ước lượng PDF theo số trang; kết quả phù hợp để client quản lý context nhưng không phải số billing chính xác của provider.

Streaming gửi SSE comment keep-alive ngay khi nhận request và mỗi 10 giây trong lúc chờ upstream, tránh client/proxy kết luận gateway mất kết nối trước token đầu tiên. Server không đặt hard timeout cho inference; HTTP keep-alive giữa các request được giữ tối đa 10 phút.

Config kiểm tra protocol, URL, provider được tham chiếu, route và key TOML không hợp lệ ngay khi khởi động. Cho phép catalog rỗng.

## Build / kiểm tra

```sh
go build ./...
go vet ./...
```
