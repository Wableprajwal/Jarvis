/**
 * Pure transforms over the raw `trace_events` stream.
 *
 * The backend emits a flat, ordered list of events. The TraceView needs three
 * things the flat list does not give it:
 *   1. tool_call and tool_result joined into a single reviewable step,
 *   2. per-agent lanes (orchestrator at depth 0, delegated agents indented),
 *   3. a turn-level summary (latency, tokens, agents, errors).
 *
 * Everything here is defensive: payload fields may be missing, latency/tokens
 * may be null, and event types outside the known set must not crash the view.
 */

import type {
  Citation,
  ErrorPayload,
  FinalResponsePayload,
  MemorySummarizePayload,
  ModelCallPayload,
  RouteDecisionPayload,
  ToolCallPayload,
  ToolResultPayload,
  TraceEvent,
} from '../api/types';

export function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function str(v: unknown): string | null {
  return typeof v === 'string' ? v : null;
}

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

export function payloadOf<T>(event: TraceEvent): Partial<T> {
  return (isRecord(event.payload) ? event.payload : {}) as Partial<T>;
}

/** Sort by `seq`, falling back to timestamp, without mutating the input. */
export function sortEvents(events: readonly TraceEvent[]): TraceEvent[] {
  return [...(events ?? [])].sort((a, b) => {
    const sa = num(a?.seq);
    const sb = num(b?.seq);
    if (sa !== null && sb !== null && sa !== sb) return sa - sb;
    const ta = Date.parse(a?.ts ?? '');
    const tb = Date.parse(b?.ts ?? '');
    if (Number.isFinite(ta) && Number.isFinite(tb) && ta !== tb) return ta - tb;
    return 0;
  });
}

export interface ToolStep {
  kind: 'tool';
  /** Stable key for React. */
  key: string;
  agent: string;
  callId: string | null;
  tool: string;
  args: Record<string, unknown>;
  /** null while a call has no matching result (still running, or dropped). */
  ok: boolean | null;
  pendingConfirmation: boolean;
  error: string | null;
  output: unknown;
  citations: Citation[];
  latencyMs: number | null;
  tokens: number | null;
  call: TraceEvent | null;
  result: TraceEvent | null;
  seq: number;
  ts: string;
}

export interface EventStep {
  kind: 'event';
  key: string;
  agent: string;
  event: TraceEvent;
  seq: number;
  ts: string;
}

export type TimelineStep = ToolStep | EventStep;

/** A contiguous run of steps produced by one agent — rendered as an indented lane. */
export interface AgentLane {
  agent: string;
  /** 0 for the first agent seen (the orchestrator); 1 for delegated agents. */
  depth: number;
  steps: TimelineStep[];
}

const UNKNOWN_AGENT = 'unknown';

function agentOf(e: TraceEvent): string {
  return str(e?.agent_name) ?? UNKNOWN_AGENT;
}

/** Pull citations out of a rag_search-shaped tool_result output. */
export function extractCitations(output: unknown): Citation[] {
  const list = Array.isArray(output)
    ? output
    : isRecord(output) && Array.isArray(output['citations'])
      ? (output['citations'] as unknown[])
      : isRecord(output) && Array.isArray(output['results'])
        ? (output['results'] as unknown[])
        : null;
  if (!list) return [];
  const cites: Citation[] = [];
  for (const item of list) {
    if (!isRecord(item)) continue;
    const source = str(item['source']);
    const text = str(item['text']);
    // Require the two fields that make a passage citable; ignore other rows.
    if (source === null && text === null) continue;
    cites.push({
      source: source ?? 'unknown source',
      chunk_id: str(item['chunk_id']) ?? '',
      text: text ?? '',
      score: num(item['score']),
    });
  }
  return cites;
}

/**
 * Join tool_call events with their tool_result.
 *
 * Primary key is `call_id`. When the backend omits it we fall back to matching
 * the earliest still-open call with the same tool name on the same agent, which
 * is correct for the sequential loop the harness actually runs.
 */
export function buildSteps(events: readonly TraceEvent[]): TimelineStep[] {
  const ordered = sortEvents(events);
  const steps: TimelineStep[] = [];
  const byCallId = new Map<string, ToolStep>();
  const openByName = new Map<string, ToolStep[]>();

  const nameKey = (agent: string, tool: string) => `${agent}\u0000${tool}`;

  for (const e of ordered) {
    if (!e || typeof e !== 'object') continue;
    const agent = agentOf(e);

    if (e.event_type === 'tool_call') {
      const p = payloadOf<ToolCallPayload>(e);
      const tool = str(p.tool) ?? 'unknown_tool';
      const callId = str(p.call_id);
      const step: ToolStep = {
        kind: 'tool',
        key: `tool-${callId ?? `${e.id ?? e.seq}`}`,
        agent,
        callId,
        tool,
        args: isRecord(p.arguments) ? p.arguments : {},
        ok: null,
        pendingConfirmation: false,
        error: null,
        output: undefined,
        citations: [],
        latencyMs: num(e.latency_ms),
        tokens: num(e.tokens),
        call: e,
        result: null,
        seq: num(e.seq) ?? steps.length,
        ts: e.ts,
      };
      steps.push(step);
      if (callId) byCallId.set(callId, step);
      const k = nameKey(agent, tool);
      const bucket = openByName.get(k);
      if (bucket) bucket.push(step);
      else openByName.set(k, [step]);
      continue;
    }

    if (e.event_type === 'tool_result') {
      const p = payloadOf<ToolResultPayload>(e);
      const tool = str(p.tool) ?? 'unknown_tool';
      const callId = str(p.call_id);
      let step = callId ? byCallId.get(callId) ?? null : null;
      if (!step) {
        const bucket = openByName.get(nameKey(agent, tool));
        step = bucket?.shift() ?? null;
      } else if (callId) {
        byCallId.delete(callId);
        const bucket = openByName.get(nameKey(agent, tool));
        if (bucket) {
          const i = bucket.indexOf(step);
          if (i >= 0) bucket.splice(i, 1);
        }
      }

      if (step) {
        step.result = e;
        step.ok = typeof p.ok === 'boolean' ? p.ok : null;
        step.pendingConfirmation = p.pending_confirmation === true;
        step.error = str(p.error);
        step.output = p.output;
        step.citations = extractCitations(p.output);
        // Prefer the result's latency: the call event is usually instantaneous.
        step.latencyMs = num(e.latency_ms) ?? step.latencyMs;
        step.tokens = num(e.tokens) ?? step.tokens;
      } else {
        // Result with no matching call — surface it rather than dropping it.
        const orphan: ToolStep = {
          kind: 'tool',
          key: `tool-orphan-${e.id ?? e.seq}`,
          agent,
          callId,
          tool,
          args: {},
          ok: typeof p.ok === 'boolean' ? p.ok : null,
          pendingConfirmation: p.pending_confirmation === true,
          error: str(p.error),
          output: p.output,
          citations: extractCitations(p.output),
          latencyMs: num(e.latency_ms),
          tokens: num(e.tokens),
          call: null,
          result: e,
          seq: num(e.seq) ?? steps.length,
          ts: e.ts,
        };
        steps.push(orphan);
      }
      continue;
    }

    steps.push({
      kind: 'event',
      key: `event-${e.id ?? `${e.event_type}-${e.seq}`}`,
      agent,
      event: e,
      seq: num(e.seq) ?? steps.length,
      ts: e.ts,
    });
  }

  return steps;
}

/** Collapse the step list into contiguous per-agent lanes. */
export function buildLanes(steps: readonly TimelineStep[]): AgentLane[] {
  const lanes: AgentLane[] = [];
  const depths = new Map<string, number>();

  for (const step of steps) {
    if (!depths.has(step.agent)) {
      // The first agent to emit anything is the orchestrator; the rest are its delegates.
      depths.set(step.agent, depths.size === 0 ? 0 : 1);
    }
    const last = lanes[lanes.length - 1];
    if (last && last.agent === step.agent) {
      last.steps.push(step);
    } else {
      lanes.push({ agent: step.agent, depth: depths.get(step.agent) ?? 1, steps: [step] });
    }
  }
  return lanes;
}

export interface RouteDecisionView {
  agent: string;
  task: string;
  reason: string | null;
  by: string;
}

export interface TurnSummaryStats {
  totalLatencyMs: number;
  totalTokens: number;
  agents: string[];
  errorCount: number;
  toolCalls: number;
  toolFailures: number;
  modelCalls: number;
  routeDecisions: RouteDecisionView[];
  citations: number;
  pendingConfirmations: number;
  startedAt: string | null;
  endedAt: string | null;
}

export function summarizeTurn(events: readonly TraceEvent[]): TurnSummaryStats {
  const ordered = sortEvents(events);
  const steps = buildSteps(ordered);
  const agents: string[] = [];
  let totalLatencyMs = 0;
  let totalTokens = 0;
  let errorCount = 0;
  let modelCalls = 0;
  const routeDecisions: RouteDecisionView[] = [];

  for (const e of ordered) {
    if (!e || typeof e !== 'object') continue;
    const agent = agentOf(e);
    if (!agents.includes(agent)) agents.push(agent);
    totalLatencyMs += num(e.latency_ms) ?? 0;
    totalTokens += num(e.tokens) ?? 0;
    if (e.event_type === 'error') errorCount += 1;
    if (e.event_type === 'model_call') modelCalls += 1;
    if (e.event_type === 'route_decision') {
      const p = payloadOf<RouteDecisionPayload>(e);
      routeDecisions.push({
        agent: str(p.agent) ?? UNKNOWN_AGENT,
        task: str(p.task) ?? '',
        reason: str(p.reason),
        by: agent,
      });
    }
  }

  let toolCalls = 0;
  let toolFailures = 0;
  let citations = 0;
  let pendingConfirmations = 0;
  for (const s of steps) {
    if (s.kind !== 'tool') continue;
    toolCalls += 1;
    if (s.ok === false) toolFailures += 1;
    if (s.pendingConfirmation) pendingConfirmations += 1;
    citations += s.citations.length;
  }

  const first = ordered[0];
  const last = ordered[ordered.length - 1];
  return {
    totalLatencyMs,
    totalTokens,
    agents,
    errorCount,
    toolCalls,
    toolFailures,
    modelCalls,
    routeDecisions,
    citations,
    pendingConfirmations,
    startedAt: first?.ts ?? null,
    endedAt: last?.ts ?? null,
  };
}

/** One-line label for a step, used in the timeline header row. */
export function describeStep(step: TimelineStep): string {
  if (step.kind === 'tool') return step.tool;
  const e = step.event;
  switch (e.event_type) {
    case 'route_decision': {
      const p = payloadOf<RouteDecisionPayload>(e);
      return `route → ${str(p.agent) ?? UNKNOWN_AGENT}`;
    }
    case 'model_call': {
      const p = payloadOf<ModelCallPayload>(e);
      const iter = num(p.iteration);
      return iter === null ? 'model call' : `model call · iteration ${iter}`;
    }
    case 'final_response': {
      const p = payloadOf<FinalResponsePayload>(e);
      const iters = num(p.iterations);
      return iters === null ? 'final response' : `final response · ${iters} iterations`;
    }
    case 'memory_summarize': {
      const p = payloadOf<MemorySummarizePayload>(e);
      const before = num(p.tokens_before);
      const after = num(p.tokens_after);
      return before !== null && after !== null
        ? `memory summarized · ${before} → ${after} tokens`
        : 'memory summarized';
    }
    case 'error': {
      const p = payloadOf<ErrorPayload>(e);
      return `error${p.stage ? ` · ${str(p.stage)}` : ''}`;
    }
    default:
      return String(e.event_type ?? 'event');
  }
}
