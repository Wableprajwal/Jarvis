/**
 * Contract test: the frontend's own parsing code, run over REAL payloads captured from the
 * FastAPI backend (Ollama + Postgres, not the mock server).
 *
 * The unit tests in lib/*.test.ts use hand-written fixtures, so they prove the logic is
 * self-consistent - not that it matches what the backend actually emits. This file closes that
 * gap: `__fixtures__/real-backend.json` was captured from a live run (a RAG turn with citations,
 * a sandboxed code_exec turn, a turn whose email_send is awaiting confirmation, plus metrics,
 * sessions and eval scorecards). If the backend ever changes a payload shape, these break.
 */
import { describe, expect, it } from 'vitest';

import {
  buildLanes,
  buildSteps,
  extractCitations,
  sortEvents,
  summarizeTurn,
} from '../lib/trace';
import {
  categoryRows,
  latestRun,
  overallPassRate,
  sortRunsNewestFirst,
  trendSeries,
} from '../lib/evalSelectors';
import type {
  Health,
  PendingConfirmation,
  Metrics,
  Scorecard,
  SessionSummary,
  TurnSummary,
  TurnTrace,
} from './types';
import captured from './__fixtures__/real-backend.json';

const real = captured as unknown as {
  health: Health;
  sessions: SessionSummary[];
  turns: TurnSummary[];
  turn_rag: TurnTrace;
  turn_code_exec: TurnTrace;
  turn_pending_confirmation: TurnTrace;
  metrics: Metrics;
  eval_runs: Scorecard[];
  confirmations: PendingConfirmation[];
};

describe('real backend: /health and /sessions', () => {
  it('health reports a live provider and model', () => {
    expect(real.health.status).toBe('ok');
    expect(real.health.provider).toBe('ollama');
    expect(real.health.model).toBeTruthy();
  });

  it('sessions carry the fields the sessions rail renders', () => {
    expect(real.sessions.length).toBeGreaterThan(0);
    for (const s of real.sessions) {
      expect(typeof s.id).toBe('string');
      expect(typeof s.preview).toBe('string');
      expect(s.created_at === null || typeof s.created_at === 'string').toBe(true);
    }
  });
});

describe('real backend: trace turns', () => {
  it('turn summaries expose the routed agents the chips show', () => {
    expect(real.turns.length).toBeGreaterThan(0);
    for (const t of real.turns) {
      expect(typeof t.turn_id).toBe('string');
      expect(Array.isArray(t.routed)).toBe(true);
      expect(Number.isFinite(new Date(t.started).getTime())).toBe(true);
    }
  });

  it('events are orderable and every tool_call is joined to its tool_result', () => {
    for (const turn of [real.turn_rag, real.turn_code_exec, real.turn_pending_confirmation]) {
      const events = sortEvents(turn.events);
      expect(events.length).toBeGreaterThan(0);
      const seqs = events.map((e) => e.seq);
      expect([...seqs].sort((a, b) => a - b)).toEqual(seqs);

      const steps = buildSteps(events);
      const toolSteps = steps.filter((s) => s.kind === 'tool');
      expect(toolSteps.length).toBeGreaterThan(0);
      // every tool step resolved: the backend emits call_id on BOTH tool_call and tool_result,
      // so the join never has to fall back to FIFO matching.
      for (const step of toolSteps) {
        if (step.kind !== 'tool') continue;
        expect(step.tool).toBeTruthy();
        expect(step.result).toBeDefined();
      }
    }
  });

  it('the orchestrator lane is depth 0 and sub-agents indent under it', () => {
    const lanes = buildLanes(buildSteps(sortEvents(real.turn_rag.events)));
    expect(lanes.length).toBeGreaterThan(1);
    expect(lanes[0]!.agent).toBe('orchestrator');
    expect(lanes[0]!.depth).toBe(0);
    expect(lanes.some((l) => l.agent === 'research' && l.depth > 0)).toBe(true);
  });

  it('summarizes a real turn with latency, tokens and route decisions', () => {
    const s = summarizeTurn(real.turn_rag.events);
    expect(s.agents).toContain('research');
    expect(s.modelCalls).toBeGreaterThan(0);
    expect(s.totalTokens).toBeGreaterThan(0);
    expect(s.totalLatencyMs).toBeGreaterThan(0);
    expect(s.citations).toBeGreaterThan(0);
    expect(s.routeDecisions[0]!.agent).toBe('research');
    expect(s.routeDecisions[0]!.reason).toBeTruthy();
  });

  it('extracts real rag_search citations (bare-list output shape)', () => {
    const events = sortEvents(real.turn_rag.events);
    const ragResult = events.find(
      (e) => e.event_type === 'tool_result' && (e.payload as { tool?: string }).tool === 'rag_search',
    );
    expect(ragResult).toBeDefined();
    const citations = extractCitations((ragResult!.payload as { output?: unknown }).output);
    expect(citations.length).toBeGreaterThan(0);
    for (const c of citations) {
      expect(c.source).toBeTruthy();
      expect(typeof c.text).toBe('string');
      expect(typeof c.score).toBe('number');
    }
  });

  it('shows the sandboxed code_exec result, including which sandbox ran it', () => {
    const events = sortEvents(real.turn_code_exec.events);
    const result = events.find(
      (e) => e.event_type === 'tool_result' && (e.payload as { tool?: string }).tool === 'code_exec',
    );
    expect(result).toBeDefined();
    const output = (result!.payload as { output?: Record<string, unknown> }).output!;
    expect(output.sandbox).toBeTruthy();
    expect(output.timed_out).toBe(false);
    expect(output.exit_code).toBe(0);
    // steps render without throwing on the nested output object
    expect(buildSteps(events).length).toBeGreaterThan(0);
  });

  it('marks a gated tool call as awaiting approval rather than done', () => {
    const events = sortEvents(real.turn_pending_confirmation.events);
    const gated = events.find(
      (e) =>
        e.event_type === 'tool_result' &&
        (e.payload as { pending_confirmation?: boolean }).pending_confirmation === true,
    );
    expect(gated).toBeDefined();
    const step = buildSteps(events).find((s) => s.kind === 'tool' && s.pendingConfirmation);
    expect(step).toBeDefined();
    expect(summarizeTurn(events).pendingConfirmations).toBeGreaterThan(0);
  });
});

describe('real backend: /metrics', () => {
  it('has per-agent latency and tool success the metrics strip reads', () => {
    expect(real.metrics.latency_by_agent.length).toBeGreaterThan(0);
    for (const row of real.metrics.latency_by_agent) {
      expect(typeof row.agent).toBe('string');
      expect(row.calls).toBeGreaterThan(0);
      expect(row.p50_ms === null || typeof row.p50_ms === 'number').toBe(true);
    }
    for (const row of real.metrics.tool_success) {
      expect(row.success_rate).toBeGreaterThanOrEqual(0);
      expect(row.success_rate).toBeLessThanOrEqual(1);
    }
    expect(real.metrics.tokens!.sessions).toBeGreaterThan(0);
  });
});

describe('real backend: /confirmations and /eval', () => {
  it('a pending confirmation carries what the approval card needs', () => {
    expect(real.confirmations.length).toBeGreaterThan(0);
    const c = real.confirmations[0]!;
    expect(c.id).toBeTruthy(); // must match what POST /confirm/{id} accepts
    expect(c.tool).toBe('email_send');
    expect(c.status).toBe('pending');
    expect(c.arguments).toBeTruthy();
  });

  it('real scorecards drive the dashboard selectors', () => {
    const runs = sortRunsNewestFirst(real.eval_runs);
    expect(runs.length).toBeGreaterThan(0);
    const latest = latestRun(runs)!;
    expect(latest.run_id).toBeTruthy();
    expect(latest.config_hash).toBeTruthy();

    const overall = overallPassRate(latest);
    expect(overall.rate).toBeGreaterThanOrEqual(0);
    expect(overall.rate).toBeLessThanOrEqual(1);
    expect(overall.total).toBeGreaterThan(0);

    const rows = categoryRows(latest, runs[1] ?? null);
    expect(rows.length).toBeGreaterThan(0);
    for (const row of rows) {
      expect(row.passed).toBeLessThanOrEqual(row.total);
    }

    const { series } = trendSeries(runs);
    expect(series.length).toBeGreaterThan(0);
  });

  it('scenario details include the expected/actual the table renders', () => {
    const latest = latestRun(sortRunsNewestFirst(real.eval_runs))!;
    expect(latest.details?.length).toBeGreaterThan(0);
    for (const d of latest.details!) {
      expect(typeof d.scenario_id).toBe('string');
      expect(typeof d.passed).toBe('boolean');
      expect(['routing', 'task']).toContain(d.layer);
    }
  });
});
