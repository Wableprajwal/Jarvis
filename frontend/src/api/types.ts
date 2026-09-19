/**
 * Wire types for the Jarvis FastAPI backend.
 *
 * The backend is still evolving, so everything here is written defensively:
 * arrays may arrive missing, numbers may arrive null, and unknown event types
 * are tolerated rather than crashing the view.
 */

export interface Health {
  status: string;
  provider: string;
  model: string;
}

export interface RouteDecision {
  agent: string;
  task: string;
  reason: string | null;
}

export type ConfirmationStatus = 'pending' | 'confirmed' | 'rejected' | (string & {});

export interface PendingConfirmation {
  id: string;
  session_id: string;
  agent: string;
  tool: string;
  arguments: Record<string, unknown>;
  status: ConfirmationStatus;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  user_id?: string;
}

export interface ChatResponse {
  session_id: string;
  turn_id: string;
  text: string | null;
  truncated: boolean;
  route_decisions: RouteDecision[];
  pending_confirmations: PendingConfirmation[];
}

export interface SessionSummary {
  id: string;
  user_id: string | null;
  created_at: string;
  preview: string | null;
}

export type MessageRole = 'user' | 'assistant' | 'system' | (string & {});

export interface StoredMessage {
  role: MessageRole;
  content: string;
  created_at: string;
}

export interface TurnSummary {
  turn_id: string;
  session_id: string;
  started: string;
  events: number;
  tokens: number;
  routed: string[];
}

/** Known event types. Unknown strings are rendered with a neutral fallback. */
export type TraceEventType =
  | 'route_decision'
  | 'model_call'
  | 'tool_call'
  | 'tool_result'
  | 'memory_summarize'
  | 'final_response'
  | 'error'
  | (string & {});

export interface TraceEvent {
  seq: number;
  id: string;
  session_id: string;
  agent_name: string;
  turn_id: string;
  event_type: TraceEventType;
  payload: Record<string, unknown>;
  latency_ms: number | null;
  tokens: number | null;
  ts: string;
}

export interface TurnTrace {
  turn_id: string;
  events: TraceEvent[];
}

// ---- payload shapes (all fields optional at runtime; narrowed via lib/trace.ts)

export interface RouteDecisionPayload {
  agent?: string;
  task?: string;
  reason?: string | null;
}

export interface ModelCallPayload {
  iteration?: number;
  stop_reason?: string;
  text?: string;
  tool_calls?: string[];
  parse_error?: string | null;
  prompt_tokens?: number;
  completion_tokens?: number;
}

export interface ToolCallPayload {
  tool?: string;
  arguments?: Record<string, unknown>;
  call_id?: string;
}

export interface Citation {
  source: string;
  chunk_id: string;
  text: string;
  score: number | null;
}

export interface ToolResultPayload {
  tool?: string;
  call_id?: string;
  ok?: boolean;
  pending_confirmation?: boolean;
  error?: string | null;
  output?: unknown;
}

export interface FinalResponsePayload {
  text?: string;
  iterations?: number;
}

export interface ErrorPayload {
  stage?: string;
  error?: string;
}

export interface MemorySummarizePayload {
  tokens_before?: number;
  tokens_after?: number;
}

// ---- metrics

export interface AgentLatency {
  agent: string;
  calls: number;
  p50_ms: number | null;
  p95_ms: number | null;
}

export interface ToolSuccess {
  tool: string;
  calls: number;
  ok: number;
  success_rate: number | null;
}

export interface TokenMetrics {
  sessions: number;
  avg_tokens_per_session: number | null;
}

export interface Metrics {
  latency_by_agent: AgentLatency[];
  tool_success: ToolSuccess[];
  tokens: TokenMetrics | null;
}

// ---- eval

export interface CategoryScore {
  pass_rate: number;
  passed: number;
  total: number;
}

export interface Regression {
  category: string;
  previous: number;
  current: number;
}

export type EvalLayer = 'routing' | 'task' | (string & {});

export interface ScenarioDetail {
  scenario_id: string;
  category: string;
  layer: EvalLayer;
  passed: boolean;
  expected?: string;
  actual?: string;
  rationale?: string;
}

/** POST /eval/run returns one of these; poll GET /eval/jobs/{job_id} until status leaves "running". */
export interface EvalJob {
  job_id: string;
  status: 'running' | 'done' | 'failed' | (string & {});
  mode: 'mock' | 'live' | (string & {});
  started_at: string;
  finished_at: string | null;
  run_id: string | null;
  error: string | null;
  scorecard?: Scorecard;
}

export interface Scorecard {
  run_id: string;
  ts: string;
  config_hash: string;
  scores_by_category: Record<string, CategoryScore>;
  regressions: Regression[];
  previous_run_id: string | null;
  details: ScenarioDetail[];
}

export interface ConfirmResponse {
  id: string;
  status: 'confirmed' | 'rejected' | (string & {});
  result: Record<string, unknown> | null;
}
