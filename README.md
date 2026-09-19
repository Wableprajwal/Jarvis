# Jarvis

A **multi-agent platform for developer workflows**. An orchestrator receives an engineering request, decides which
specialist agent(s) it needs, routes the work, and assembles the answer. It runs on a **swappable local model backend**,
isolates credentials, sandboxes code execution, records a structured trace of everything it does, and **grades its own
output** with a two-layer evaluation harness.

`JARVIS_BUILD.md` is the full spec; `DECISIONS.md` records why each design choice was made (and what was rejected).

**Headline property:** every core behaviour — the tool-calling loop, routing, tool validation, the prompted-tool-call
parser, memory summarization, eval scoring — is testable through `MockProvider`. The test suite needs **no live model**
and no network.

---

## Architecture

```
┌──────────────┐      ┌──────────────────────────────────────────────────────────┐
│  React UI    │◄────►│                    FastAPI backend                        │
│ Chat / Trace │ /api │  ┌───────────────┐                                        │
│ / Eval views │      │  │ Orchestrator  │── routes via route_to_<agent> ──┐      │
└──────────────┘      │  └───────┬───────┘                                 ▼      │
                      │          │                        ┌────────────────────┐  │
                      │          │                        │ Code    agent      │  │
                      │          │                        │ Research agent     │  │
                      │          │                        │ Ops      agent     │  │
                      │          │                        └─────────┬──────────┘  │
                      │  ┌───────▼────────┐   ┌─────────────────────▼───────────┐ │
                      │  │ ModelProvider  │   │ ToolRegistry                    │ │
                      │  │ Ollama / Mock  │   │  per-agent allow-lists          │ │
                      │  └────────────────┘   │  schema validation              │ │
                      │  ┌────────────────┐   │  confirmation gate              │ │
                      │  │ Session +      │   │  code_exec (sandboxed)          │ │
                      │  │ MemoryStore    │   │  rag_search, files, git         │ │
                      │  └────────────────┘   │  calendar / email (gated)       │ │
                      │  ┌────────────────┐   └─────────────────────────────────┘ │
                      │  │ Eval harness   │   ┌─────────────────────────────────┐ │
                      │  │ routing + task │   │ Observability: TraceEvent       │ │
                      │  └────────────────┘   └─────────────────────────────────┘ │
                      └──────────────────────────────┬───────────────────────────┘
                                                     │
                     ┌───────────────────────────────▼────────────────────────────┐
                     │ Postgres: sessions, messages, facts, trace_events,         │
                     │           eval_runs, rag_chunks (pgvector or float8[])     │
                     │ Sandbox:  Docker / Seatbelt / bwrap / unshare (fail-closed) │
                     └────────────────────────────────────────────────────────────┘
```

**Core rule:** every box is an interface first. The orchestrator and agents never import a concrete model SDK, vector
client or tool. `app/bootstrap.py` is the single composition root that knows the concrete classes.

Two design choices worth calling out:

- **Routing is itself a tool call.** Each sub-agent is exposed to the orchestrator as a `route_to_<agent>` tool, so one
  loop serves both function calls and agent handoffs — uniform tracing, one iteration cap, chaining for free.
- **Capability isolation.** An agent can only reach a tool through `registry.get(agent, tool)`, which raises unless that
  agent was explicitly granted it. A routing bug or a hallucinated tool name cannot hand `email_send` to the Code agent.

---

## Setup

### 1. Ollama (local model)
```bash
brew install ollama          # or https://ollama.com/download
ollama serve &               # if not already running
ollama pull qwen2.5:7b       # default; llama3.1:8b also supports native tool-calling
```

### 2. Postgres (+ pgvector)
```bash
make up                      # docker compose up -d postgres   (published on host port 5433)
cp .env.example .env
```
Already running your own Postgres? Point `DATABASE_URL` at it — the schema (`backend/app/schema.sql`) is applied at
startup and is idempotent. Without pgvector the RAG store automatically falls back to a `float8[]` + numpy cosine
implementation, so nothing breaks. Without any database at all, the app degrades to in-memory traces/memory/eval runs.

### 3. Backend
```bash
make setup                   # venv + pinned requirements
make ingest                  # embed the sample notes in docs/ for the Research agent (idempotent)
make dev                     # FastAPI on :8000
```

### 4. Frontend
```bash
make frontend                # npm install + vite dev server on :5173, proxying /api -> :8000
```
Backend not running? `cd frontend && npm run mock` serves realistic fixture data on :8000 instead.

---

## Everyday commands

| Command | What it does |
|---|---|
| `make check` | everything CI runs: lint + full test suite + offline eval gate |
| `make test` | whole pytest suite (no live model needed; Ollama-backed tests self-skip) |
| `make lint` | `ruff check` over `app/` and `tests/` |
| `make eval` | the **offline eval gate** — deterministic, no model, non-zero exit on regression |
| `make eval-live` | the same scenarios against the real configured model |
| `make eval-regression-demo` | deliberately breaks a prompt to show the gate catching it |
| `make ingest` | (re-)embed `docs/` into the vector store |
| `make setup-db` | create the separate database the tests use (only if your role can't `CREATE DATABASE`) |

**Tests never touch your dev database.** They use `<your db>_test` (override with `TEST_DATABASE_URL`). Sharing one
database meant a test run left hundreds of mock-latency rows behind, which then skewed the `/metrics` dashboard.

---

## Running the eval suite

Two layers, because "did it pick the right agent" and "was the answer any good" fail for different reasons:

- **Routing (deterministic, no judge).** The real orchestrator with real agent descriptions, but stub sub-agents, so only
  the routing decision is measured against `expected_agent`. Scenarios: `backend/app/eval/routing_scenarios/`.
- **Task (checks + LLM judge).** The full pipeline — real registry, allow-lists, confirmation gate, tools, RAG, memory —
  scored by deterministic checks (tool sequence, required/forbidden text, was the answer actually grounded in retrieval,
  did anything irreversible happen without confirmation) **and** an LLM judge on task success / grounding / tool
  correctness / safety. Both must pass. Scenarios: `backend/app/eval/task_scenarios/`.

```bash
make eval          # offline: MockProvider + a simulated router. This is the CI gate.
make eval-live     # real model + real judge. Slower, and the number that actually means something.
curl -X POST localhost:8000/eval/run    # same thing through the API; the dashboard renders it
```

How the offline gate stays meaningful with no model: routing scenarios run against a deterministic policy whose decision
is a **function of the agent-description text it is shown**, so editing a description really does move routing accuracy;
task scenarios carry a compact replay script that drives the real tools, gate and retrieval. Each run is scored per
category, diffed against the previous run (regressions + scenarios that flipped pass→fail), and gated on per-category
thresholds in `backend/app/eval/thresholds.yaml`. `config_hash` fingerprints the provider, every model, every prompt,
every agent's allow-list and every tool schema, so any of those changing shows up as a new configuration.

The judge is always a **separate provider instance** (its own `JUDGE_MODEL`) from the agents under test — point it at a
stronger model where you can.

---

## Switching models

`MODEL_PROVIDER=ollama|mock` selects the backend; `ORCHESTRATOR_MODEL`, `CODE_MODEL`, `RESEARCH_MODEL`, `OPS_MODEL`,
`JUDGE_MODEL` and `SUMMARIZER_MODEL` pick a model **per role** — routing can run on something cheap while code
generation runs on something stronger. A model without native function-calling automatically switches to the prompted
JSON-block parser (`app/providers/tool_call_parser.py`), which validates and repairs malformed tool calls instead of
crashing. Nothing above `app/providers/` knows which is active.

---

## Security posture

- `code_exec` runs LLM-written code with **no network, a read-only scoped filesystem, a hard timeout, and memory/process
  caps**, in Docker / macOS Seatbelt / bwrap / unshare. If no isolating backend can be proven available, it **refuses to
  run** rather than falling back to a bare subprocess.
- Secrets live only in `CredentialStore` and reach tools as constructor-time closures. Three independent layers
  (provider, tracer, registry) scrub any that leak into a message, trace or tool result, and fail closed.
- Irreversible tools (`email_send`, `calendar_create`) never execute inline — they return a pending action that a human
  must approve via `POST /confirm/{id}`.
- `user_id` is a **namespacing key, not authentication** — this build has no authN, so the API enforces session
  ownership consistently but a caller can still claim any id. Real deployment adds auth middleware in front.

---

## Repo layout

```
backend/app/
  providers/    ModelProvider abstraction: Mock, Ollama, prompted tool-call parser
  orchestrator/ routing-as-tool-calling
  agents/       Code / Research / Ops, each with an explicit tool allow-list
  harness/      the shared tool-calling loop, sessions, memory
  tools/        registry (validation + allow-lists + confirmation gate), sandboxed code_exec,
                credentials, files/git, RAG search, calendar, email
  rag/          chunking, embeddings, hybrid retrieval, reranking, stores
  eval/         scenarios, judge, runner, CI gate
  observability/ TraceEvent + Postgres store + metrics
frontend/       React + TS: Chat, TraceView, EvalDashboard
k8s/            Deployment/Service/ConfigMap — design artifacts, not a live cluster
```

## Honest scope

This runs locally. `k8s/` manifests describe how it would be deployed and have **not** been applied to a live cluster.
The sandbox is verified on macOS Seatbelt here; the Docker/bwrap/unshare backends have the same tests but skip on a host
that cannot run them. See the end of `DECISIONS.md` for what the live eval found — including two scenarios still judged
as failures by a 7B judge that are arguably correct behaviour, left documented rather than tuned away.
