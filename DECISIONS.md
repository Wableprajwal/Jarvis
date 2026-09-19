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
