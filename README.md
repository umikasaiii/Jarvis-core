# JARVIS Core

JARVIS Core is the local AI orchestration server for the JARVIS assistant.
It runs on a PC/laptop and, when reachable, lets the JARVIS Android app
delegate chat, complex reasoning, and (in future phases) memory/RAG/voice
to a locally hosted LLM instead of the phone's own on-device model.

Android keeps working stand-alone if JARVIS Core is offline or
unreachable — Core is a delegate, not a hard dependency.

```
Android
  |
  v
JARVIS Core API (/v1)
  |
  v
Request Orchestrator
  |
  v
CoreAiRouter --- FastModel
             \-- BrainModel
  |
  v
Response Streaming (SSE)
```

This phase implements the server, API, LLM adapter layer, streaming,
health/capabilities, FAST/BRAIN routing, config, logging, and tests. It
does **not** implement long-term memory, RAG, a Context Engine, an Inner
Loop, proactivity, STT/TTS, Home Assistant integration, or vision — those
are documented placeholders under `future/`.

## Requirements

- Python 3.11+ (3.12 recommended)
- Optional, for real inference: [Ollama](https://ollama.com) or a
  [llama.cpp](https://github.com/ggml-org/llama.cpp) server (`llama-server`)
  running locally. Without either, the server runs happily on the built-in
  `fake` backend (deterministic echo provider), which is also what the
  test suite uses.

## Installation

```bash
git clone <this-repo>
cd jarvis-core
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
```

## Configuration

All configuration lives in `.env` (see `.env.example` for the full list
and defaults). Key groups:

- **Server**: `SERVER_HOST`, `SERVER_PORT`, `ALLOW_REMOTE_CONNECTIONS`,
  `DEVELOPMENT_MODE`.
- **FAST model**: `FAST_MODEL_BACKEND` (`fake` | `ollama` | `llamacpp`),
  `FAST_MODEL_NAME`, `FAST_MODEL_PATH`, `FAST_CONTEXT_SIZE`,
  `FAST_THREADS`, `FAST_GPU_LAYERS`, `FAST_BASE_URL`.
- **BRAIN model**: same fields, `BRAIN_*`. If your machine can't yet run
  two distinct models, set `BRAIN_MODEL_NAME`/`BRAIN_MODEL_BACKEND` equal
  to the FAST ones — nothing else changes.
- **Routing**: `ROUTER_FAST_MAX_CHARS`, `ROUTER_BRAIN_KEYWORDS`,
  `ROUTER_FAST_KEYWORDS`.
- **Requests/backpressure**: `REQUEST_TIMEOUT`, `MAX_CONCURRENT_REQUESTS`,
  `MAX_QUEUE_SIZE`.
- **Conversation/context**: `MAX_CONVERSATION_MESSAGES`, `MAX_CONTEXT_CHARS`.
- **Security**: `API_TOKEN`, `MAX_REQUEST_BODY_BYTES`, `RATE_LIMIT_PER_MINUTE`.
- **Logging**: `LOG_LEVEL`, `LOG_JSON`.

The system prompt lives in `config/prompts/jarvis_system.txt` — edit it
directly, no rebuild/recompile needed.

## Choosing a model backend

**Ollama** (simplest to set up):
```bash
ollama pull qwen2.5:3b-instruct        # example FAST model
ollama pull qwen2.5:14b-instruct       # example BRAIN model
```
```
FAST_MODEL_BACKEND=ollama
FAST_MODEL_NAME=qwen2.5:3b-instruct
FAST_BASE_URL=http://127.0.0.1:11434

BRAIN_MODEL_BACKEND=ollama
BRAIN_MODEL_NAME=qwen2.5:14b-instruct
BRAIN_BASE_URL=http://127.0.0.1:11434
```

**llama.cpp server**: start `llama-server` yourself (with your model,
thread count, GPU layers, etc.), then point Core at it:
```
FAST_MODEL_BACKEND=llamacpp
FAST_MODEL_NAME=my-fast-model
FAST_BASE_URL=http://127.0.0.1:8080
```

Switching backends or models never touches the API, routing, or the
Android contract — see `providers/base.py` (`LlmProvider` interface) and
`ai/model_manager.py`.

## Running

```bash
scripts/start-dev.sh      # auto-reload, for development   (Windows: scripts/start-dev.ps1)
scripts/start-server.sh   # no reload, closer to production (Windows: scripts/start-server.ps1)
```
or directly:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Testing

```bash
scripts/test.sh            # Windows: scripts/test.ps1
# or
pytest
```

All 49+ tests run against the built-in `FakeLlmProvider` — no multi-GB
model download required. Covered: health, capabilities, request
validation, FAST/BRAIN routing (including explicit target and keyword
heuristics), model-unavailable/fallback, timeouts, streaming (start/token/
done/error, cancellation), inference queue/backpressure, conversation
isolation and context trimming, invalid requests, protocol-version
mismatch, and startup/shutdown (including with an unreachable backend).

### Real-model integration check

Once you've configured a real Ollama/llama.cpp model, sanity-check it:
```bash
scripts/start-server.sh &
curl http://127.0.0.1:8000/v1/health
curl http://127.0.0.1:8000/v1/models
scripts/benchmark.sh --base-url http://127.0.0.1:8000 --target FAST
scripts/benchmark.sh --base-url http://127.0.0.1:8000 --target BRAIN
```
`benchmark.py` reports prompt/generated tokens, TTFT, tokens/sec, and
total latency per prompt (RAM usage is included only if `psutil` is
installed).

## API (v1)

| Method | Path              | Purpose                                          |
|--------|-------------------|---------------------------------------------------|
| GET    | `/v1/health`      | Lightweight liveness probe (no inference)          |
| GET    | `/v1/capabilities`| What this Core build supports                      |
| GET    | `/v1/models`      | FAST/BRAIN backend, model name, live availability  |
| POST   | `/v1/chat`        | Conversational request/response (persists history) |
| POST   | `/v1/ai/request`  | Generic non-streaming AI request                   |
| POST   | `/v1/ai/stream`   | Same as above, streamed via SSE (start/token/done/error) |

Request/response bodies follow `JarvisRequest`/`JarvisResponse`
(`api/schemas/request.py`, `api/schemas/response.py`), matching the
Android `JarvisCoreRequest`/`JarvisCoreResponse`/`AiExecutionTarget`/
`AiRequestType` contract. Every request must declare `protocolVersion`
(`"1"` today); a mismatched version returns `400` explicitly rather than
failing silently.

Example:
```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"protocolVersion":"1","text":"Ciao JARVIS, come stai?"}'
```

Streaming (SSE, one JSON object per `data:` line):
```bash
curl -N -X POST http://127.0.0.1:8000/v1/ai/stream \
  -H "Content-Type: application/json" \
  -d '{"protocolVersion":"1","text":"Raccontami qualcosa"}'
```

## Connecting Android

Point `JarvisCoreClient`'s base URL at this machine's LAN address and
port (e.g. `http://192.168.1.50:8000`). Recommended flow:
1. `GET /v1/health` to detect Core is reachable, and `GET /v1/capabilities`
   to adapt UI/feature availability to this Core build.
2. `POST /v1/ai/stream` for normal conversation, falling back to the
   phone's local model if Core is unreachable or returns an error.
3. Keep `conversationId` stable per on-device conversation so Core can
   retain short-term context; `context`/`metadata` are optional — don't
   assume Core needs them.

If Core is shut down, Android must continue operating on its own local
model; Core never needs to be "informed" of this, it's simply offline.

## Security (LAN)

- Plain HTTP is acceptable for local development on a trusted LAN only.
- Set `API_TOKEN` to require `Authorization: Bearer <token>` on every
  endpoint except `/v1/health` and the docs routes.
- Request bodies are capped (`MAX_REQUEST_BODY_BYTES`) and a light
  per-IP rate limiter is always active (`RATE_LIMIT_PER_MINUTE`).
- Binding to anything other than `127.0.0.1`/`localhost` logs a loud
  warning at startup — this is not a statement that remote/Internet
  access is safe. True remote access will be added later via VPN/TLS,
  not by exposing this server directly.
- No secrets are ever logged; request text, documents, and the system
  prompt are excluded from structured logs by design (see
  `core/logging.py`).

## Troubleshooting

- **`/v1/models` shows `available: false`**: the configured backend
  (Ollama/llama.cpp) isn't reachable at `FAST_BASE_URL`/`BRAIN_BASE_URL`.
  Start it first, or check the URL/port.
- **Server starts but chat/stream returns an error**: check
  `LlmProviderError` details in the JSON response's `error` field and the
  server logs (`requestId` correlates them).
- **429 `rate_limited`**: raise `RATE_LIMIT_PER_MINUTE` or check for a
  client retry loop.
- **413 `request_too_large`**: raise `MAX_REQUEST_BODY_BYTES` if you
  intentionally send large `context`/`metadata` payloads.
- **Streaming stalls in a browser/proxy**: some proxies buffer SSE;
  `/v1/ai/stream` already sets `X-Accel-Buffering: no` and
  `Cache-Control: no-cache`, but a proxy in front of Core may need
  equivalent config.

## Project structure

```
app/            FastAPI app factory, lifespan (startup/shutdown)
api/            routes/ + schemas/ (versioned /v1 API contract)
core/           config, structured logging, security middleware
ai/             CoreAiRouter, ModelManager, InferenceQueue, RequestOrchestrator
providers/      LlmProvider interface + fake/ollama/llamacpp adapters
conversations/  ConversationStore interface + in-memory implementation
future/         documented, unwired extension points for later phases
config/prompts/ system prompt (editable without a rebuild)
tests/          pytest suite (FakeLlmProvider-based, no real model needed)
scripts/        start-dev, start-server, test, benchmark (sh + ps1)
```

## What's next (out of scope for this phase)

Long-term/personal memory, vector DB + RAG, Context Engine, Inner Loop,
proactivity, STT/TTS, Home Assistant integration, and advanced vision —
see `future/*/README.md` for the interface sketches each will plug into.
