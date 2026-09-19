# JARVIS — Multi-Agent Developer Tooling Platform
## Complete Build Specification for Claude Code

> **How to use this document:** This is the single source of truth for building Jarvis. Read it fully before writing code. Follow the phases in Section 12 in order. After each phase, run its acceptance checks and append a note to `DECISIONS.md`. Do not skip ahead — later phases depend on the interfaces established earlier. Do not pause to ask which phase to do next; proceed through the build order automatically until a phase's acceptance checks pass, then move to the next.

---

## 1. Objective

Jarvis is a **multi-agent platform for developer workflows**. An orchestrator agent receives an engineering request, decides which specialized sub-agent(s) it needs, routes the work, and assembles a final answer. It runs on a **swappable local model backend**, isolates credentials and sandboxes code execution, records structured traces of everything it does, and **grades its own outputs** with a two-layer evaluation harness (routing correctness + task correctness).

The design goal is that the **model backend is a replaceable component**: the orchestrator, sub-agents, memory, tools, and eval layers never depend on which model is running underneath. A local Ollama model today; a different one tomorrow; a mock model in tests — with no changes to the rest of the system.

### What it does for the user (real utility — the "customer" is an engineer)
- **Code/Repo agent**: answers questions about a local codebase, generates unit tests for a given function/file, and summarizes a git diff into a PR description.
- **Research agent**: hybrid RAG over the user's own technical notes/docs, returning cited answers.
- **Ops agent**: calendar/email actions (secondary — demonstrates breadth of tool integration and confirmation-gating on irreversible actions).
- **Orchestrator**: routes requests to the right agent(s), chains them when a task needs more than one (e.g. Research retrieves context → Code uses it to write a test), and exposes a trace of *why* it routed the way it did.

### Non-goals (keep scope honest)
- Not a production Kubernetes deployment. We produce container/K8s manifests as design artifacts, not a live cluster.
- Not a hosted API product. Runs locally.
- Not dependent on any paid API key. Primary backend is local Ollama; a `MockProvider` covers all automated tests.

---

## 2. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Backend | Python 3.11+, FastAPI | async throughout |
| Model runtime | Ollama (local) | default model: `qwen2.5:7b` or `llama3.1:8b` (both support tool-calling) |
| Relational store | PostgreSQL | sessions, memory, traces, eval runs |
| Vector store | pgvector (preferred) or Chroma | embeddings for RAG |
| Embeddings | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) | local, offline, independent of the chat model |
| Keyword search | `rank_bm25` | for hybrid retrieval |
| Reranker | `sentence-transformers` CrossEncoder (`ms-marco-MiniLM-L-6-v2`) | optional but implement |
| Frontend | React + TypeScript + Vite | chat, trace view, eval dashboard |
| Containerization | Docker + docker-compose | plus `k8s/` manifests as design artifacts |
| Testing | pytest, pytest-asyncio | all core logic testable via `MockProvider` |
| Config | pydantic-settings, `.env` | never hardcode secrets |

Use `uv` or `pip` with a `requirements.txt`/`pyproject.toml`. Pin versions.

---

## 3. Architecture Overview

```
┌──────────────┐     ┌────────────────────────────────────────────────┐
│  React UI    │◄───►│                 FastAPI Backend                  │
│ Chat / Trace │     │  ┌──────────────┐                                │
│ / Eval views │     │  │ Orchestrator  │──routes to──┐                 │
└──────────────┘     │  │   Agent       │             │                 │
                      │  └──────┬───────┘             ▼                 │
                      │         │            ┌─────────────────────┐    │
                      │         │            │  Code/Repo Agent     │    │
                      │         │            │  Research/RAG Agent  │    │
                      │         │            │  Ops/Scheduler Agent │    │
                      │         │            └─────────┬───────────┘    │
                      │  ┌──────▼───────┐   ┌───────────▼────────────┐  │
                      │  │ ModelProvider │   │  ToolRegistry (per-    │  │
                      │  │ Ollama / Mock │   │  agent allow-lists)    │  │
                      │  └──────────────┘   │  - code_exec (sandbox) │  │
                      │  ┌──────────────┐   │  - rag_search          │  │
                      │  │ Session +     │   │  - git_diff/summarize  │  │
                      │  │ MemoryStore   │   │  - calendar/email      │  │
                      │  └──────────────┘   └────────────────────────┘  │
                      │  ┌──────────────┐   ┌────────────────────────┐  │
                      │  │ Eval Harness  │   │ Observability          │  │
                      │  │ routing+task  │   │ TraceEvent + dashboard │  │
                      │  └──────────────┘   └────────────────────────┘  │
                      └────────────────────────┬──────────────────────────┘
                                                │
                          ┌─────────────────────▼─────────────────────┐
                          │ Postgres (sessions/memory/traces/evals)    │
                          │ pgvector (embeddings)                       │
                          │ Sandboxed exec (isolated subprocess/container)│
                          └─────────────────────────────────────────────┘
```

**Core principle:** every box is an interface first, implementation second. The orchestrator and agents never import a concrete model SDK, a concrete vector DB client, or a concrete tool. They depend only on the abstract contracts in Sections 4–9. This is the single most important architectural rule.

---

## 4. ModelProvider — model abstraction

```python
# app/providers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str | None = None

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

@dataclass
class ModelResponse:
    text: str | None
    tool_calls: list[ToolCall]
    stop_reason: str          # "end_turn" | "tool_use" | "max_tokens"
    usage: Usage

class ModelProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[dict],       # normalized message dicts
        tools: list[dict],          # JSON-schema tool definitions
        max_tokens: int = 2048,
    ) -> ModelResponse: ...

    @abstractmethod
    def supports_native_tool_calling(self) -> bool: ...
```

Implementations:
- **`MockProvider`** — deterministic. Takes a scripted list of `ModelResponse`s (or a rule: "if the last user message contains X, return tool_call Y"). Enables full harness/orchestrator/eval testing with **no live model**. Build this FIRST.
- **`OllamaProvider`** — calls Ollama's OpenAI-compatible endpoint (`POST {OLLAMA_HOST}/v1/chat/completions`). If the model supports native function calling, use the `tools` param and parse `tool_calls` from the response. If `supports_native_tool_calling()` is False, use `PromptedToolCallParser`.
- **`PromptedToolCallParser`** (`app/providers/tool_call_parser.py`) — a separate, unit-tested component. Injects tool schemas into the system prompt, instructs the model to emit tool calls as a fenced ```json block, extracts and validates that block against each tool's JSON schema, and returns normalized `ToolCall`s. Handle malformed output gracefully (return an error the harness feeds back to the model, don't crash).

Provider selection: a factory `get_provider(name: str) -> ModelProvider` reading `MODEL_PROVIDER` from config. No other code branches on provider type. Agents may each hold their own provider instance (e.g. a cheaper model for routing, a stronger one for code gen) — support this even if the same model is used everywhere today.

---

## 5. Agent Harness — the tool-calling loop

```python
# app/harness/loop.py — shared by orchestrator and every sub-agent
async def run_agent_loop(
    provider: ModelProvider,
    messages: list[dict],
    tools: list[Tool],
    tracer: Tracer,
    session_id: str,
    agent_name: str,
    max_iterations: int = 8,
) -> AgentResult:
    for i in range(max_iterations):
        resp = await provider.generate(messages, [t.schema for t in tools])
        tracer.log(session_id, agent_name, "model_call", ...)   # latency, tokens
        if resp.tool_calls:
            for call in resp.tool_calls:
                tool = registry.get(agent_name, call.name)      # allow-list enforced
                result = await safe_execute(tool, call.arguments)  # never raises
                tracer.log(session_id, agent_name, "tool_result", ...)
                messages.append(tool_result_message(call, result))
            continue
        tracer.log(session_id, agent_name, "final_response", ...)
        return AgentResult(text=resp.text, messages=messages)
    return AgentResult(text="<max iterations reached>", messages=messages, truncated=True)
```

Rules:
- Hard cap iterations (default 8) — a weaker local model failing to converge is a real failure mode, design for it.
- `safe_execute` wraps tool execution: any exception becomes a structured error result fed back to the model, never a harness crash.
- Every iteration logs a `TraceEvent` **before** continuing.
- Tool argument validation against JSON schema happens **before** `execute` is called; invalid args return a structured error.

---

## 6. Orchestrator & Sub-Agents

```python
# app/agents/base.py
class SubAgent(ABC):
    name: str
    description: str            # the orchestrator reads this to decide routing
    system_prompt: str
    tool_names: list[str]       # explicit allow-list, NOT the global registry

    async def run(self, task: str, context: list[dict], tracer, session_id) -> AgentResult:
        tools = [registry.get(self.name, n) for n in self.tool_names]
        messages = [{"role": "system", "content": self.system_prompt}, *context,
                    {"role": "user", "content": task}]
        return await run_agent_loop(self.provider, messages, tools, tracer, session_id, self.name)
```

**Orchestrator** (`app/orchestrator/orchestrator.py`):
- Routing is itself a tool-calling decision: present each sub-agent as a "tool" (`route_to_<agent>`) whose schema is `{task: str}`, built from the agent's `name` + `description`. The orchestrator calls `route_to_code`, the harness dispatches to `CodeAgent.run(task)`, the result comes back as a tool result, and the orchestrator decides whether it's done or needs another agent. **This reuses the same tool-calling loop for both function calls and agent handoffs** — one mechanism, two uses. State this as a deliberate design choice.
- Support sequential chaining (agent A's output feeds agent B) and, where independent, parallel dispatch with result merging.
- Log a `route_decision` trace event capturing which agent was chosen and the orchestrator's stated reason.

**Sub-agents** (each in `app/agents/`):
- `CodeAgent` — tools: `code_exec` (sandboxed), `read_file`, `git_diff_summarize`, `generate_tests`.
- `ResearchAgent` — tools: `rag_search`.
- `OpsAgent` — tools: `calendar_list`, `calendar_create` (confirmation-gated), `email_search`, `email_draft`, `email_send` (confirmation-gated).

**Capability isolation:** each agent is constructed with an explicit tool allow-list. `registry.get(agent_name, tool_name)` raises if that agent isn't permitted that tool. A routing bug therefore cannot expose `email_send` to the Code agent. This is a security-adjacent design choice — document it.

---

## 7. Memory — MemoryStore interface

```python
# app/harness/memory.py
class MemoryStore(ABC):
    async def get_context(self, session_id: str) -> list[dict]: ...
    async def append(self, session_id: str, message: dict) -> None: ...
    async def summarize_if_needed(self, session_id: str, token_budget: int) -> None: ...
    async def get_long_term_facts(self, user_id: str) -> list[Fact]: ...
    async def upsert_long_term_fact(self, user_id: str, fact: Fact) -> None: ...
```

- **Short-term**: rolling window in Postgres per session. When the active context exceeds `token_budget`, `summarize_if_needed` calls the model to compress the *older* turns into a single summary block, replacing them (do not merely truncate). Trigger is token-budget based, not turn-count based — document why (turns vary wildly in size; budget is what actually matters for the context window).
- **Long-term**: a `facts` table keyed by `user_id`. Populated explicitly ("remember that…") or by a periodic extraction pass over closed sessions. Injected at the start of a new session so the orchestrator has durable context.
- Both behind one interface so swapping storage backends never touches the harness or agents.

---

## 8. RAG Pipeline (Research agent's tool)

- **Ingestion** (`app/rag/ingest.py`): recursive character splitter, ~500-token chunks, ~15% overlap, metadata `{source, date, doc_type}`. Embed with the local sentence-transformers model. Store in pgvector.
- **Embedding model is independent of the chat ModelProvider** — RAG works fully offline regardless of which chat model is active. Document this decoupling.
- **Retrieval** (`app/rag/retriever.py`): hybrid — dense vector search (top-k=20) + BM25 keyword (top-k=20) → merge/dedupe → CrossEncoder rerank → top 5 with citations.
- **Exposure**: `rag_search(query: str) -> list[Citation]` is a Tool the Research agent calls. It is NOT force-injected into every prompt — the agent decides when to retrieve. Document why (always-RAG wastes context and degrades on tasks that don't need retrieval).

---

## 9. Tools, Credential Isolation & Sandboxing

```python
# app/tools/registry.py
class Tool(ABC):
    name: str
    description: str
    input_schema: dict           # JSON Schema
    requires_confirmation: bool   # True for send/write/irreversible actions

    async def execute(self, arguments: dict) -> ToolResult: ...
```

**Registry**: validates `arguments` against `input_schema` before `execute`. Enforces per-agent allow-lists (Section 6). Malformed calls return a structured error, never crash or silently no-op.

**Credential isolation** (`app/tools/credentials.py`): OAuth tokens / API secrets live here and are injected into tool instances at construction. The model context NEVER contains a secret — the model sees only tool names, schemas, and results. Load secrets from env/`.env`.

**Sandboxing** — `code_exec` (`app/tools/code_exec.py`) must run untrusted generated code with real isolation:
- Separate subprocess (or a short-lived Docker container if Docker is available).
- **No network access** (e.g. run under a network namespace / `unshare -n`, or a container with `--network none`).
- **Read-only, scoped filesystem** — mount only the specific file(s) the task needs, not the whole repo or home dir.
- **Hard timeout** (e.g. 10s) and **memory cap** (e.g. 256MB, via `resource.setrlimit` or container limits).
- Runs as a restricted OS user where possible.
- Return `{stdout, stderr, exit_code, timed_out}` as a structured `ToolResult`.

Document the rejected alternative in `DECISIONS.md`: running `code_exec` in-process would let a single malformed generation touch the harness's own filesystem, network, and credentials — unacceptable blast radius.

**Confirmation gating**: tools with `requires_confirmation=True` do not execute directly. They return a "pending confirmation" result; the API surfaces it to the UI; execution proceeds only after the user confirms. This mirrors real agent products and is a good tradeoff to defend (irreversible actions need a human gate; read actions don't).

---

## 10. Observability

```python
# app/observability/trace.py — TraceEvent, written to Postgres `trace_events`
# {id, session_id, agent_name, turn_id, event_type, payload(json), latency_ms, tokens, ts}
```
Event types: `route_decision`, `model_call`, `tool_call`, `tool_result`, `memory_summarize`, `final_response`, `error`.

This table is the single source of truth for both the frontend **trace view** and the eval harness's ability to inspect *how* an answer was produced, not just what it was. Build a minimal dashboard (React) over aggregates: routing accuracy over time, per-agent latency (p50/p95), tool success rate, tokens/session. This is the "know an agent is broken before users do" story at demo scale.

---

## 11. Evaluation Harness (two-layer)

- **Scenario sets** (YAML under `app/eval/`):
  - `routing_scenarios/`: `{input, expected_agent, category}` — the correct sub-agent is known.
  - `task_scenarios/`: `{input, expected_tool_calls?, grounding_source?, category, rubric_notes}`.
  - Categories: routing, tool-selection, RAG-grounding, memory-recall, refusal/edge-case. Hand-write 25–50 total.
- **Judge** (`app/eval/judge.py`): a separate `ModelProvider` instance (may point at a stronger model than the agent under test — matters more with a weak local agent). Scores each transcript on: task success (0–1), grounding fidelity (only asserted what tools/retrieval supported), tool correctness, safety. Returns structured scores + rationale.
- **Routing eval**: compares the orchestrator's chosen agent to `expected_agent` — deterministic, no judge needed.
- **Runner** (`app/eval/runner.py`): `POST /eval/run` executes all scenarios against the current config, writes a row to `eval_runs` `{run_id, config_hash, ts, scores_by_category}`, and **diffs against the previous run to flag regressions**. This is the "gates every prompt, skill, harness change, and model upgrade" story.
- **CI gating**: provide a `make eval` / pytest entry that runs the eval suite against `MockProvider` deterministically and fails the build if any category's pass-rate drops below a threshold. This makes "offline replay + CI gating" (JD language) a real, demonstrable thing.

---

## 12. Build Order (do these in sequence; each has acceptance checks)

**Phase 1 — Provider + loop + one agent (Mock-driven)**
- Build `ModelProvider` base, `MockProvider`, `Tool`/`ToolRegistry`, `run_agent_loop`, a trivial `get_time` tool.
- Build `CodeAgent` with a mock `read_file` tool. Build the `Orchestrator` routing to it.
- ✅ Acceptance: a pytest drives a scripted `MockProvider` end-to-end — user request → orchestrator routes to CodeAgent → tool call executes → final answer. Loop iteration cap tested. All green with no live model.

**Phase 2 — Tracing**
- `TraceEvent` + `Tracer`, Postgres `trace_events`, wired into the loop and orchestrator (`route_decision` included).
- ✅ Acceptance: after a run, `trace_events` contains an ordered, inspectable record of the full turn.

**Phase 3 — Ollama + real tool-calling**
- `OllamaProvider` + `PromptedToolCallParser` (with unit tests for malformed output).
- `docker-compose.yml` bringing up Postgres + pgvector. `README` with Ollama setup (`ollama pull qwen2.5:7b`).
- ✅ Acceptance: with Ollama running locally, the Phase 1 flow works against a real model. Parser tests pass.

**Phase 4 — Multi-agent**
- Add `ResearchAgent` + RAG pipeline (ingest, hybrid retrieve, rerank, `rag_search` tool). Add `OpsAgent` with calendar/email tools (mock the external APIs behind the `Tool` interface; real OAuth optional/last).
- Per-agent allow-lists enforced in the registry.
- ✅ Acceptance: orchestrator correctly routes 3 distinct requests to the 3 agents; a chained request (Research → Code) works; an allow-list violation is rejected in a test.

**Phase 5 — Sandboxing + credentials**
- Real `code_exec` sandbox (no network, scoped FS, timeout, mem cap). `credentials.py` isolation.
- ✅ Acceptance: a test proves generated code cannot open a socket / read outside its scoped path / exceed the timeout, and that no secret ever appears in a message/trace payload.

**Phase 6 — Memory**
- Short-term rolling window + token-budget summarization; long-term facts table + session-start injection.
- ✅ Acceptance: a multi-turn test shows older turns get summarized past the budget; a fact stored in one session is recalled in a new one.

**Phase 7 — Eval harness**
- Routing + task scenario sets, judge, runner, `eval_runs` + regression diff, `make eval` CI gate against MockProvider.
- ✅ Acceptance: `POST /eval/run` produces a scorecard; deliberately breaking a prompt makes the diff flag a regression; `make eval` fails on a sub-threshold category.

**Phase 8 — Frontend + infra artifacts**
- React: Chat, TraceView (per-turn tool calls + retrieved context + route decisions), EvalDashboard (scorecards + trend).
- `k8s/` manifests (Deployment, Service, ConfigMap) as design artifacts.
- ✅ Acceptance: end-to-end from the UI: send a message, watch the trace populate, view a routing decision, open the eval dashboard.

---

## 13. Repo Structure

```
jarvis/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app + routes (/chat, /eval/run, /traces, /confirm)
│   │   ├── config.py               # pydantic-settings; MODEL_PROVIDER, DB, OLLAMA_HOST, model names
│   │   ├── providers/
│   │   │   ├── base.py
│   │   │   ├── mock_provider.py
│   │   │   ├── ollama_provider.py
│   │   │   └── tool_call_parser.py
│   │   ├── orchestrator/
│   │   │   ├── orchestrator.py
│   │   │   └── routing.py
│   │   ├── agents/
│   │   │   ├── base.py
│   │   │   ├── code_agent.py
│   │   │   ├── research_agent.py
│   │   │   └── ops_agent.py
│   │   ├── harness/
│   │   │   ├── loop.py
│   │   │   ├── session.py
│   │   │   └── memory.py
│   │   ├── rag/
│   │   │   ├── ingest.py
│   │   │   ├── retriever.py
│   │   │   └── reranker.py
│   │   ├── tools/
│   │   │   ├── registry.py
│   │   │   ├── code_exec.py         # sandboxed
│   │   │   ├── credentials.py       # isolated
│   │   │   ├── files.py             # read_file, git_diff_summarize, generate_tests
│   │   │   ├── gmail.py
│   │   │   └── calendar.py
│   │   ├── eval/
│   │   │   ├── judge.py
│   │   │   ├── runner.py
│   │   │   ├── routing_scenarios/
│   │   │   └── task_scenarios/
│   │   └── observability/
│   │       └── trace.py
│   ├── tests/                       # pytest, all core logic via MockProvider
│   ├── requirements.txt / pyproject.toml
│   └── Dockerfile
├── frontend/
│   └── src/components/{Chat,TraceView,EvalDashboard}/
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
├── docker-compose.yml               # postgres+pgvector, backend, (frontend)
├── Makefile                         # make dev, make test, make eval, make ingest
├── .env.example
├── DECISIONS.md
└── README.md
```

---

## 14. Cross-Cutting Requirements

- **Testing:** every phase ships tests. Core logic (loop, orchestrator routing, memory summarization, tool validation, parser, eval scoring) MUST be testable via `MockProvider` with no live model. This is a headline design property — call it out in the README.
- **Config/secrets:** all via `config.py` + `.env`. Never hardcode. Ship `.env.example`. `.gitignore` must exclude `.env`.
- **Async:** FastAPI + async DB access throughout. Providers, tools, agents are `async`.
- **SOLID / clean interfaces:** favor composition over inheritance; each module depends on abstractions (Sections 4–9), not concretions. This is explicitly graded in the interview.
- **DECISIONS.md:** after each phase, append entries in the form *"Chose X over Y because Z; accepted tradeoff W."* Cover at minimum: routing-as-tool-calling, per-agent allow-lists, sandbox isolation model, embedding model decoupled from chat model, hybrid vs. pure-vector retrieval, token-budget vs. turn-count summarization, judge decoupled from agent model, confirmation gating.
- **README.md:** setup (Ollama install + model pull, docker-compose up, ingest sample docs, run backend/frontend), architecture diagram, and a short "how to run the eval suite" section.
- **Git hygiene:** small, clearly-messaged commits per logical unit. The commit history is itself an interview artifact ("here's how I built the eval gate incrementally").

---

## 15. Definition of Done

1. `docker-compose up` + local Ollama → working end-to-end multi-agent chat via the UI.
2. All three sub-agents reachable; orchestrator routes correctly and can chain agents; trace view shows route decisions.
3. `code_exec` provably sandboxed; no secret ever reaches a model/trace payload.
4. Shared memory: summarization past budget + cross-session fact recall both demonstrated.
5. `POST /eval/run` produces a scorecard; `make eval` gates CI against MockProvider and catches an injected regression.
6. `DECISIONS.md` populated; `README.md` complete; `k8s/` manifests present as design artifacts.
7. Test suite green with no live model dependency.

---

## 16. Interview Talking Points (for the human, not for building)

Keep these defensible with specifics from what you actually built:
- **Why an orchestrator over one mega-agent?** Tool-selection accuracy degrades as tool count grows; per-agent system prompts perform better per-domain; capability isolation shrinks blast radius.
- **Why is routing a tool call?** One mechanism (tool-calling) serves both function calls and agent handoffs — less code, uniform tracing, easier to eval.
- **How is code execution sandboxed?** No network, read-only scoped mount, timeout + memory cap, isolated from harness credentials; in-process was rejected for blast radius.
- **How do you know the eval harness works?** Two layers (routing + task), scorecard diffing catches regressions, and it runs deterministically against MockProvider in CI with no live model.
- **What changes at Tesla's scale?** Durable session storage → managed embeddings store + a real message queue for async agent handoff; subprocess sandbox → container/VM boundary; single-node → K8s with the manifests already sketched.
- **Honesty on scope:** "I containerized it and wrote K8s manifests for how I'd run it; I haven't operated a live cluster with it." Overclaiming loses the deep-dive.
