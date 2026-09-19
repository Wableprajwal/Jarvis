# DECISIONS

Format: *Chose X over Y because Z; accepted tradeoff W.* One section per build phase.

## Phase 1 — Provider, loop, first agent

### Routing is a tool call
Chose to present each sub-agent to the orchestrator as a tool (`route_to_<agent>`, schema `{task, reason?}`) over a
separate classifier / hard-coded router, because one mechanism (the tool-calling loop) then serves function calls *and*
agent handoffs: one loop to test, uniform tracing, uniform iteration caps, and chaining/parallel dispatch fall out for
free (a second `route_to_*` call after the first result is a chain; several calls in one model turn run concurrently
via `asyncio.gather`). Accepted tradeoff: routing quality is only as good as the model's tool selection, and a weak
local model can mis-route — which is exactly why routing has its own deterministic eval layer (Phase 7).
The optional `reason` field is what gets recorded in the `route_decision` trace event.

### Route tools are built per request, in a private registry
Chose per-request `RouteTool` instances closing over `(tracer, session_id, turn_id)` over a global route tool with
ambient context, because `Tool.execute(arguments)` should stay a pure `arguments -> result` contract (no hidden
globals, no context-vars), and so the orchestrator's allow-list contains routing tools and nothing else.
Accepted tradeoff: a few small objects allocated per turn.

### Per-agent tool allow-lists, enforced at dispatch
Chose an explicit `registry.grant(agent, tools)` + `registry.get(agent, tool)` that raises `ToolNotPermitted` over
handing agents the global registry, because a routing bug or a model hallucinating `email_send` inside the Code agent
must not be able to reach the tool. Enforcement is at *dispatch* (not just in the schemas offered to the model),
since a model can emit a tool name it was never shown. The violation is converted to a structured error result
(including the tools that *are* available) and fed back to the model instead of crashing the loop.
Accepted tradeoff: an agent's allow-list is declared on the agent class, so it is only as trustworthy as code review
of that class; a central policy file is the next step at larger scale.

### Validation and failures never crash the loop
Chose validate-then-execute (`jsonschema`, Draft 2020-12) in the registry over trusting each tool to validate, and a
catch-all at the tool boundary (`safe_execute`) over letting exceptions propagate, because a 7B local model produces
malformed calls routinely; the useful response is a precise error message the model can correct against.
Accepted tradeoff: a catch-all can hide programmer errors — mitigated by emitting them as `tool_result` trace events
with the exception type and message.

### Provider-neutral tool schema; explicit registry param on the loop
Chose `{"name","description","input_schema"}` as the tool definition handed to providers (each provider translates to
its wire format) over the OpenAI `{"type":"function",...}` shape, so agents/tests never encode one vendor's format.
`run_agent_loop` takes the `registry` explicitly (the spec sketched a module global) so tests and the per-request
orchestrator registry need no monkeypatching. Accepted tradeoff: one extra parameter.

### Iteration cap as a first-class failure mode
Hard cap (default 8) returning `<max iterations reached>` with `truncated=True` and an `error` trace event, rather than
raising, because non-convergence of a weak model is expected behaviour to observe and grade, not an exception.
The orchestrator also tells its caller when a sub-agent was truncated.

### Confirmation gating lives in the registry
Tools with `requires_confirmation=True` never execute in `dispatch`; the registry stores a `PendingAction` and returns a
"pending_confirmation" result the model is told means *not performed*. Only `registry.confirm(id)` (called by the API
after a human approves) executes it. Chose gating in the registry over per-tool logic so no tool author can forget it.
Accepted tradeoff: pending actions are held in process memory in this build (lost on restart); a durable table is the
obvious production change.

## Phase 2 — Tracing

### One trace table, ordered by a DB-assigned sequence
Chose a single `trace_events` table (JSONB payload) with a `BIGSERIAL seq` as the ordering key over separate per-type
tables or ordering by timestamp, because the same rows feed the UI trace view, the metrics dashboard *and* the eval
harness, and timestamps collide when concurrent tool calls finish in the same millisecond. `seq` gives a total order
even across processes. Accepted tradeoff: JSONB payloads are schemaless, so payload shape per event type is a
convention enforced by the loop rather than by the database.

### The loop logs before it continues
`tool_call` is written *before* execution and `tool_result` after, so a hung or crashing tool still leaves a record of
what was attempted. The orchestrator's `route_decision` (agent, task, stated reason) is written by the route tool
between the two, which is why the trace reads model_call -> tool_call -> route_decision -> <sub-agent events> ->
tool_result -> model_call -> final_response. All events of a request share a `turn_id`, sub-agents included.

### Tracer is an interface; Postgres is one implementation
`Tracer` (abstract) with `NullTracer`, `InMemoryTracer` (unit tests, eval replay) and `PostgresTracer`. The read side
(`TraceStore`: turn/session lookups, per-agent p50/p95 model latency via `percentile_cont`, tool success rate,
tokens/session) is separate from the write side so the hot path stays a single INSERT.

### A fresh database instead of the pre-existing `jarvis` one
The local Postgres already contained a `jarvis` database from an earlier, differently-shaped schema (text payloads, no
`agent_name`) holding real rows. Chose to leave it untouched and create `jarvis_platform` over migrating/dropping it,
because destroying someone's data to satisfy a new schema is not a call the build should make. Accepted tradeoff: two
databases on the dev machine; compose users are unaffected (they get a clean `jarvis` DB in the container).

## Phase 3 — Ollama and real tool-calling

### OpenAI-compatible endpoint, native tools when the model supports them
Chose Ollama's `/v1/chat/completions` over its native `/api/chat` because the OpenAI shape is the de-facto wire format:
swapping in vLLM, llama.cpp-server or a hosted API later is a base-URL change, not a rewrite. Native function-calling is
auto-selected by model family (`qwen2.5`, `llama3.1`, ...) and can be forced per instance; everything else goes through
`PromptedToolCallParser`. Accepted tradeoff: a hard-coded family allow-list must be maintained (overridable via
`native_tools=`).

### Malformed model output is data, not an exception
Chose a first-class `ModelResponse.error` (set by the provider/parser) that the loop turns into a corrective user message
and a retry, over raising. A 7B model emits broken JSON regularly; each retry consumes an iteration so it is bounded by the
same hard cap. The parser is deliberately lenient about things that are harmless (code fences, prose around the block,
trailing commas, `name`/`parameters` aliases, stringified arguments) and strict about things that are not (unknown tool,
schema violation): one bad call rejects the whole batch so the model never half-executes a plan.

### Verified against a real model, not just mocks
The Phase 1 flow (orchestrator -> code agent -> `read_file` -> answer) is exercised against local `qwen2.5:7b` in both native
and prompted modes (`tests/test_live_ollama.py`; auto-skipped when Ollama is not running so CI stays model-free).

## Phase 4 — Multi-agent: RAG, Research, Ops

### Embedding model decoupled from the chat model
Chose a local `bge-small-en-v1.5` embedder behind an `Embedder` interface, independent of the chat `ModelProvider`, over reusing the chat model's embeddings, because RAG then works fully offline and swapping (or upgrading) the chat model never invalidates stored vectors. Every chunk is tagged with `Embedder.name` and searches are scoped to it, so chunks from a different embedder are invisible rather than silently wrong. Accepted tradeoff: a second model to ship; vectors are only comparable within one embedder.

### Hybrid retrieval over pure-vector
Chose dense top-20 + BM25 top-20, fused with reciprocal-rank fusion, then a cross-encoder rerank to the top 5 with citations, over dense-only, because BM25 catches exact identifiers, alert names and config keys (`HalcyonQueueDepth`, `hnsw.ef_search`) that embeddings blur, while the reranker fixes ordering. Accepted tradeoff: more moving parts and a rerank latency cost per query.

### RAG is an agent-chosen tool, not always-injected
Chose `rag_search` as a Tool the Research agent decides to call over force-injecting retrieved context into every prompt, because always-RAG wastes context and degrades tasks that need no retrieval, and tool calls are traceable and gradable (the eval checks retrieval happened and grounded the answer). Accepted tradeoff: a weak model may skip retrieval or under-search; the Research prompt requires search-first and an explicit "I could not find this in the notes" when unsupported.

### pgvector with an array-store fallback (and an in-memory store)
Chose pgvector (HNSW) as the primary store, a `float8[]` + numpy cosine fallback picked automatically by a factory, and an `InMemoryVectorStore` for CI/eval, over requiring the extension everywhere, because dev Postgres often lacks it (this machine's Postgres 14 does; the compose image `pgvector/pgvector:pg16` has it). Accepted tradeoff: the fallback is a linear scan (fine to tens of thousands of chunks); `PgVectorStore` is written but was not exercised against a live pgvector in this environment.

### Confirmation gating for calendar_create / email_send
Chose gating in the registry (tools declare `requires_confirmation=True`; nothing executes until `registry.confirm(id)`) over trusting the prompt to ask first, because irreversible actions need a human gate and a hallucinated or injected call must not act. Accepted tradeoff: an extra user step, and semantic errors JSON Schema cannot catch (bad dates, missing drafts) only surface after approval.
