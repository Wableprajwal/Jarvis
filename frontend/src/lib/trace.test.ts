import { describe, expect, it } from 'vitest';
import {
  buildLanes,
  buildSteps,
  describeStep,
  extractCitations,
  sortEvents,
  summarizeTurn,
} from './trace';
import type { TraceEvent } from '../api/types';

let seq = 0;
function e(partial: Partial<TraceEvent> & { event_type: string }): TraceEvent {
  seq += 1;
  return {
    seq,
    id: `evt${seq}`,
    session_id: 's1',
    agent_name: 'orchestrator',
    turn_id: 't1',
    payload: {},
    latency_ms: null,
    tokens: null,
    ts: new Date(1700000000000 + seq * 1000).toISOString(),
    ...partial,
  } as TraceEvent;
}

describe('sortEvents', () => {
  it('orders by seq without mutating the input', () => {
    const input = [e({ event_type: 'a', seq: 3 }), e({ event_type: 'b', seq: 1 })];
    const copy = [...input];
    const out = sortEvents(input);
    expect(out.map((x) => x.seq)).toEqual([1, 3]);
    expect(input).toEqual(copy);
  });

  it('survives a null/undefined event list', () => {
    expect(sortEvents(undefined as unknown as TraceEvent[])).toEqual([]);
  });
});

describe('buildSteps — tool_call / tool_result joining', () => {
  it('joins a call with its result by call_id', () => {
    const steps = buildSteps([
      e({
        event_type: 'tool_call',
        agent_name: 'code_agent',
        payload: { tool: 'read_file', arguments: { path: 'a.py' }, call_id: 'c1' },
      }),
      e({
        event_type: 'tool_result',
        agent_name: 'code_agent',
        latency_ms: 42,
        payload: { tool: 'read_file', call_id: 'c1', ok: true, output: { lines: 10 } },
      }),
    ]);

    expect(steps).toHaveLength(1);
    const step = steps[0]!;
    expect(step.kind).toBe('tool');
    if (step.kind !== 'tool') throw new Error('expected a tool step');
    expect(step.tool).toBe('read_file');
    expect(step.ok).toBe(true);
    expect(step.args).toEqual({ path: 'a.py' });
    expect(step.latencyMs).toBe(42);
    expect(step.output).toEqual({ lines: 10 });
  });

  it('joins interleaved calls to the right result, not the nearest one', () => {
    const steps = buildSteps([
      e({ event_type: 'tool_call', payload: { tool: 'rag_search', call_id: 'a' } }),
      e({ event_type: 'tool_call', payload: { tool: 'rag_search', call_id: 'b' } }),
      e({ event_type: 'tool_result', payload: { tool: 'rag_search', call_id: 'b', ok: false, error: 'boom' } }),
      e({ event_type: 'tool_result', payload: { tool: 'rag_search', call_id: 'a', ok: true } }),
    ]);

    expect(steps).toHaveLength(2);
    const [first, second] = steps as [typeof steps[0], typeof steps[0]];
    if (first.kind !== 'tool' || second.kind !== 'tool') throw new Error('expected tool steps');
    expect(first.callId).toBe('a');
    expect(first.ok).toBe(true);
    expect(second.callId).toBe('b');
    expect(second.ok).toBe(false);
    expect(second.error).toBe('boom');
  });

  it('falls back to FIFO matching on tool name when call_id is missing', () => {
    const steps = buildSteps([
      e({ event_type: 'tool_call', agent_name: 'code_agent', payload: { tool: 'read_file' } }),
      e({ event_type: 'tool_call', agent_name: 'code_agent', payload: { tool: 'read_file' } }),
      e({ event_type: 'tool_result', agent_name: 'code_agent', payload: { tool: 'read_file', ok: true } }),
      e({ event_type: 'tool_result', agent_name: 'code_agent', payload: { tool: 'read_file', ok: false } }),
    ]);
    expect(steps).toHaveLength(2);
    const [a, b] = steps as [typeof steps[0], typeof steps[0]];
    if (a.kind !== 'tool' || b.kind !== 'tool') throw new Error('expected tool steps');
    expect(a.ok).toBe(true);
    expect(b.ok).toBe(false);
  });

  it('does not match a result to a call from a different agent', () => {
    const steps = buildSteps([
      e({ event_type: 'tool_call', agent_name: 'code_agent', payload: { tool: 'read_file' } }),
      e({ event_type: 'tool_result', agent_name: 'research_agent', payload: { tool: 'read_file', ok: true } }),
    ]);
    expect(steps).toHaveLength(2);
  });

  it('keeps an unmatched call visible with ok=null', () => {
    const steps = buildSteps([
      e({ event_type: 'tool_call', payload: { tool: 'code_exec', call_id: 'x' } }),
    ]);
    const step = steps[0]!;
    if (step.kind !== 'tool') throw new Error('expected a tool step');
    expect(step.ok).toBeNull();
    expect(step.result).toBeNull();
  });

  it('surfaces an orphan result rather than dropping it', () => {
    const steps = buildSteps([
      e({ event_type: 'tool_result', payload: { tool: 'ghost', call_id: 'z', ok: false } }),
    ]);
    expect(steps).toHaveLength(1);
    const step = steps[0]!;
    if (step.kind !== 'tool') throw new Error('expected a tool step');
    expect(step.call).toBeNull();
    expect(step.tool).toBe('ghost');
  });

  it('flags a pending confirmation', () => {
    const steps = buildSteps([
      e({ event_type: 'tool_call', payload: { tool: 'email_send', call_id: 'p' } }),
      e({
        event_type: 'tool_result',
        payload: { tool: 'email_send', call_id: 'p', ok: false, pending_confirmation: true },
      }),
    ]);
    const step = steps[0]!;
    if (step.kind !== 'tool') throw new Error('expected a tool step');
    expect(step.pendingConfirmation).toBe(true);
  });

  it('tolerates a missing payload and unknown tool names', () => {
    const steps = buildSteps([e({ event_type: 'tool_call', payload: undefined as never })]);
    const step = steps[0]!;
    if (step.kind !== 'tool') throw new Error('expected a tool step');
    expect(step.tool).toBe('unknown_tool');
    expect(step.args).toEqual({});
  });
});

describe('extractCitations', () => {
  it('reads a bare list of citation objects', () => {
    const cites = extractCitations([
      { source: 'docs/a.md', chunk_id: 'a#1', text: 'hello', score: 0.9 },
      { source: 'docs/b.md', chunk_id: 'b#1', text: 'world', score: null },
    ]);
    expect(cites).toHaveLength(2);
    expect(cites[0]?.source).toBe('docs/a.md');
    expect(cites[1]?.score).toBeNull();
  });

  it('reads a {citations: [...]} or {results: [...]} wrapper', () => {
    expect(extractCitations({ citations: [{ source: 's', text: 't' }] })).toHaveLength(1);
    expect(extractCitations({ results: [{ source: 's', text: 't' }] })).toHaveLength(1);
  });

  it('returns nothing for non-citation output', () => {
    expect(extractCitations({ lines: 10 })).toEqual([]);
    expect(extractCitations(null)).toEqual([]);
    expect(extractCitations([{ unrelated: 1 }])).toEqual([]);
  });
});

describe('buildLanes', () => {
  it('groups contiguous steps per agent and indents delegates', () => {
    const lanes = buildLanes(
      buildSteps([
        e({ event_type: 'route_decision', agent_name: 'orchestrator', payload: { agent: 'code_agent' } }),
        e({ event_type: 'model_call', agent_name: 'code_agent' }),
        e({ event_type: 'model_call', agent_name: 'code_agent' }),
        e({ event_type: 'final_response', agent_name: 'orchestrator' }),
      ]),
    );
    expect(lanes.map((l) => l.agent)).toEqual(['orchestrator', 'code_agent', 'orchestrator']);
    expect(lanes.map((l) => l.depth)).toEqual([0, 1, 0]);
    expect(lanes[1]?.steps).toHaveLength(2);
  });
});

describe('summarizeTurn', () => {
  const events = [
    e({ event_type: 'model_call', agent_name: 'orchestrator', latency_ms: 100, tokens: 50 }),
    e({
      event_type: 'route_decision',
      agent_name: 'orchestrator',
      payload: { agent: 'research_agent', task: 'find docs', reason: 'owns rag_search' },
    }),
    e({ event_type: 'tool_call', agent_name: 'research_agent', payload: { tool: 'rag_search', call_id: 'c' } }),
    e({
      event_type: 'tool_result',
      agent_name: 'research_agent',
      latency_ms: 200,
      payload: {
        tool: 'rag_search',
        call_id: 'c',
        ok: true,
        output: [{ source: 'd.md', text: 'x', score: 0.5 }],
      },
    }),
    e({ event_type: 'error', agent_name: 'research_agent', payload: { stage: 'parse', error: 'bad json' } }),
  ];

  it('totals latency, tokens, agents, errors and citations', () => {
    const s = summarizeTurn(events);
    expect(s.totalLatencyMs).toBe(300);
    expect(s.totalTokens).toBe(50);
    expect(s.agents).toEqual(['orchestrator', 'research_agent']);
    expect(s.errorCount).toBe(1);
    expect(s.toolCalls).toBe(1);
    expect(s.toolFailures).toBe(0);
    expect(s.modelCalls).toBe(1);
    expect(s.citations).toBe(1);
  });

  it('surfaces route decisions with their reason', () => {
    const s = summarizeTurn(events);
    expect(s.routeDecisions).toEqual([
      { agent: 'research_agent', task: 'find docs', reason: 'owns rag_search', by: 'orchestrator' },
    ]);
  });

  it('returns zeroed stats for an empty turn', () => {
    const s = summarizeTurn([]);
    expect(s.totalLatencyMs).toBe(0);
    expect(s.agents).toEqual([]);
    expect(s.startedAt).toBeNull();
  });
});

describe('describeStep', () => {
  it('labels known event types and falls back for unknown ones', () => {
    const [route] = buildSteps([
      e({ event_type: 'route_decision', payload: { agent: 'ops_agent' } }),
    ]);
    expect(describeStep(route!)).toBe('route → ops_agent');

    const [mystery] = buildSteps([e({ event_type: 'something_new' })]);
    expect(describeStep(mystery!)).toBe('something_new');

    const [mem] = buildSteps([
      e({ event_type: 'memory_summarize', payload: { tokens_before: 900, tokens_after: 200 } }),
    ]);
    expect(describeStep(mem!)).toContain('900 → 200');
  });
});
