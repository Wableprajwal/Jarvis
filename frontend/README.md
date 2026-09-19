# Jarvis frontend

React + TypeScript + Vite UI for the Jarvis multi-agent platform. Three views:

- **Chat** — talk to the orchestrator. Each assistant reply carries chips for the
  route decisions (agent + why) and a link into the trace for that turn.
  Confirmation-gated tools render an approval card (Approve / Reject).
- **Traces** — turn list on the left, an indented per-agent timeline on the right:
  route decisions highlighted, `tool_call`/`tool_result` joined into one step with
  args, ok/fail badge, latency and collapsible output, RAG passages rendered as
  cited quotes, errors in red, plus a turn summary header and a metrics strip
  (p50/p95 per agent, tool success rate, tokens).
- **Evals** — run the suite, read the latest scorecard as per-category cards,
  see regressions flagged loudly, a hand-drawn SVG trend chart of pass rate per
  category over recent runs, and a filterable scenario details table.

Navigation is hash-based (`#/chat`, `#/traces/<turn_id>`, `#/evals/<run_id>`), so
every view is linkable.

## Run it

```bash
npm install

# 1. Against the real backend (FastAPI on :8000)
npm run dev                  # http://localhost:5173

# 2. Against fixture data, no backend needed
npm run mock                 # zero-dep node server on :8000
npm run dev                  # in a second terminal

npm test                     # vitest unit tests
npm run build                # tsc -b && vite build (must be error-free)
npm run preview              # serve the production build (also proxies /api)
```

## API wiring

The UI calls `/api/*`. In dev and preview, Vite proxies `/api/*` to
`http://localhost:8000/*` with the prefix stripped (see `vite.config.ts`), so
there is no CORS configuration anywhere. Point the proxy elsewhere with
`JARVIS_BACKEND=http://host:port npm run dev`, or bypass it entirely with
`VITE_API_BASE` (see `.env.example`) when serving the built assets from another
origin.

- `src/api/types.ts` — wire types for every endpoint.
- `src/api/client.ts` — typed client; all failures become a single `ApiError`
  that distinguishes "backend unreachable" from an HTTP error carrying `detail`.
- `src/lib/trace.ts` — pure transforms: event sorting, tool call/result joining,
  per-agent lanes, turn summary. Unit tested.
- `src/lib/evalSelectors.ts` — pure scorecard selectors: category rows with
  deltas, regressions, trend series, detail filters. Unit tested.

Every field is read defensively: missing arrays, null latency/tokens and unknown
`event_type` values render rather than crash.

## Mock backend

`mock-server.mjs` (plain node, no dependencies) serves realistic fixtures for
every endpoint, including a multi-agent chained turn with RAG citations and a
sandbox-blocked `code_exec` failure, a pending `email_send` confirmation that
flips when approved, and five eval runs where run 4 regresses two categories.
`POST /chat` and `POST /eval/run` mutate the in-memory state, so the UI's write
paths are exercisable offline. As in the real backend, `POST /eval/run` answers `202`
with a job and `GET /eval/jobs/{id}` reports `running` on the first poll and `done` (with
the scorecard) after, so the dashboard's polling loop is exercised against the mock too.

## Design notes

- Design tokens live in `src/styles/tokens.css`; light is the base palette and
  dark redefines only the token values under `prefers-color-scheme`.
- The trend chart is hand-drawn SVG (no chart library). Categorical colours are
  assigned in fixed slot order and never cycled, a missing category is drawn as a
  gap rather than a zero, and identity is never colour-alone: there is always a
  legend, and the cards and details table carry every number in text.
- Responsive to ~400px; panes stack below 900px.
