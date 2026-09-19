# Demo walkthrough (≈10 minutes)

Every prompt below works against the shipped sample notes (`docs/`) and sample workspace
(`backend/app/eval/fixtures/workspace/`). Each one is chosen to show a specific design property; the
**Traces** tab shows *why* the system did what it did after every step.

## 0. Bring it up

```bash
ollama pull qwen2.5:7b && ollama serve &        # local model
docker compose up -d                            # Postgres+pgvector, backend, UI
open http://localhost:8080
```
(or without Docker: `make up`, `make setup`, `make ingest`, `make dev`, `make frontend` → :5173)

## 1. Routing + grounded RAG with citations — *Research agent*

> According to our notes, how many attempts does the retry policy allow, and what is the deadline?

Expect: routed to **research**, answer "5 attempts … 30 seconds", cited `[adr-0007-retry-backoff.md#…]`.
In Traces: the `route_decision` card shows the orchestrator's stated **reason**; the `rag_search` step
expands into the retrieved passages with scores. Point out that retrieval is a *tool the agent chose to
call*, not context force-fed into every prompt.

## 2. A chained request — *Research → Code, in order*

> Look up our retry policy in the notes, then write pytest tests for utils/retry.py based on it.

Expect: two route decisions, `research` then `code`; the tests contain the policy's real numbers
(5 attempts, 10 s cap). In Traces: research fully completes before code starts, and code's context
carries research's findings. This is the fix described in DECISIONS.md ("Agent handoffs run in order").

## 3. Capability isolation + the confirmation gate — *Ops agent*

> Email dana@example.com that the release slipped to Monday.

Expect: routed to **ops**; the reply says the email is **awaiting your approval** — nothing was sent. An
approval card appears with the exact arguments; click **Approve** and the send executes (against the
fake mailbox). Then try:

> Read utils/retry.py and also mail me a summary.

The Code agent has no `email_send` — the registry refuses the call at dispatch (allow-list), the model
is told so, and the answer is honest that it could not send. Show the `tool_result` with
`ok: false, "not permitted"` in the trace.

## 4. Sandboxed code execution — *Code agent*

> Use code_exec to run: print(sum(range(1, 101)))

Expect: `5050`; the trace's `tool_result` shows `"sandbox": "seatbelt"` (or `docker` on Linux). Then:

> Use code_exec to fetch https://example.com and print the status.

Expect: it fails — no network inside the sandbox. `backend/tests/test_phase5_sandbox.py` proves the
rest (no reads outside scope, timeout, memory cap, fork bomb contained, scrubbed environment).

## 5. Memory across sessions

> Remember that my team is called Platform Reliability.

Click **New** session, then:

> What is my team called?

Expect: answered without any agent — the fact was injected at session start from long-term memory.

## 6. The eval harness — *Evals tab*

Click **Run eval** (≈15 min live - it runs as a background job and the page polls; or show a stored run). Talk through the two layers: routing is scored
deterministically against `expected_agent`; task scenarios are scored by deterministic checks *and* a
judge that is a **separate provider instance**. Regressions vs the previous run are flagged loudly.

Then in a terminal:
```bash
make eval                    # offline gate, no model, ~20 s — this is what CI runs
make eval-regression-demo    # breaks the Research description on purpose: routing drops, gate fails
```

## 7. Honest scope (say this before being asked)

- The k8s manifests are design artifacts; nothing has been applied to a live cluster (`k8s/README.md`).
- `user_id` is namespacing, not authentication.
- Two live eval scenarios still fail with the 7B model and are documented as findings, not tuned
  away (end of `DECISIONS.md`).

## Talking points → evidence

| Claim | Where it is proven |
|---|---|
| model backend is swappable | `MockProvider` runs 370+ tests with no model; `make eval` runs the whole pipeline offline |
| routing is a tool call | `orchestrator/routing.py`; one loop in `harness/loop.py` serves tools and handoffs |
| per-agent allow-lists | `tools/registry.py::get` raises `ToolNotPermitted`; test `edge-cross-agent-tool-blocked` |
| sandbox is real | 32 unmocked tests: sockets fail, `/etc/passwd` unreadable, fork bomb contained, 1 GB alloc killed |
| secrets never reach the model | `tools/credentials.py`: 3 independent redaction layers, fail-closed, 58 tests |
| eval catches regressions | `test_phase7_eval.py`: injected prompt break, disabled gate, emptied index all trip it |
| it was verified live, not just mocked | DECISIONS.md "Live verification" and "Audit pass" sections |
