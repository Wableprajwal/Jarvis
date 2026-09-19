# Jarvis

Multi-agent developer tooling platform. An orchestrator routes engineering requests to specialist agents
(Code, Research, Ops) over a **swappable local model backend**, isolates credentials, sandboxes code execution, traces
everything, and grades itself with a two-layer eval harness. See `JARVIS_BUILD.md` for the full spec and `DECISIONS.md`
for the reasoning behind each design choice.

**Headline property:** every core behaviour (loop, routing, tool validation, parser, memory, eval scoring) is testable
with `MockProvider` - the test suite needs no live model.

## Setup

### 1. Ollama (local model)
```bash
brew install ollama          # or https://ollama.com/download
ollama serve &               # if not already running
ollama pull qwen2.5:7b       # default model; llama3.1:8b also supports tool calling
```

### 2. Postgres (+ pgvector)
```bash
make up                      # docker compose up -d postgres  (published on host port 5433)
cp .env.example .env
```
Already running your own Postgres? Point `DATABASE_URL` in `.env` at it; the schema (`backend/app/schema.sql`) is
applied automatically at startup and is idempotent.

### 3. Backend
```bash
make setup                   # venv + pinned requirements
make dev                     # FastAPI on :8000
```

### 4. Tests
```bash
make test                    # whole suite; MockProvider only, live-model tests auto-skip when Ollama is absent
cd backend && .venv/bin/python -m pytest tests/test_live_ollama.py   # exercise a real model
```

## Switching models
`MODEL_PROVIDER=ollama|mock` selects the backend; `*_MODEL` env vars pick a model per role (orchestrator, code,
research, ops, judge, summarizer). Models without native function-calling automatically use the prompted JSON-block
parser (`app/providers/tool_call_parser.py`). Nothing above `app/providers/` knows or cares which is active.
