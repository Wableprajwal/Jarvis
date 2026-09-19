import { describe, expect, it } from 'vitest';
import {
  allCategories,
  categoryRows,
  detailCategories,
  filterDetails,
  latestRun,
  overallPassRate,
  regressionsOf,
  sortRunsNewestFirst,
  trendSeries,
} from './evalSelectors';
import type { Scorecard } from '../api/types';

function card(partial: Partial<Scorecard> & { run_id: string; ts: string }): Scorecard {
  return {
    config_hash: 'cfg',
    scores_by_category: {},
    regressions: [],
    previous_run_id: null,
    details: [],
    ...partial,
  };
}

const older = card({
  run_id: 'run_001',
  ts: '2026-01-01T10:00:00Z',
  scores_by_category: {
    routing: { pass_rate: 0.9, passed: 9, total: 10 },
    'RAG-grounding': { pass_rate: 0.8, passed: 8, total: 10 },
  },
});

const newer = card({
  run_id: 'run_002',
  ts: '2026-01-02T10:00:00Z',
  previous_run_id: 'run_001',
  scores_by_category: {
    routing: { pass_rate: 1, passed: 10, total: 10 },
    'RAG-grounding': { pass_rate: 0.5, passed: 5, total: 10 },
  },
  regressions: [{ category: 'RAG-grounding', previous: 0.8, current: 0.5 }],
});

describe('sorting and selection', () => {
  it('sorts runs newest first and picks the latest', () => {
    const sorted = sortRunsNewestFirst([older, newer]);
    expect(sorted.map((r) => r.run_id)).toEqual(['run_002', 'run_001']);
    expect(latestRun([older, newer])?.run_id).toBe('run_002');
  });

  it('returns null for an empty or missing run list', () => {
    expect(latestRun([])).toBeNull();
    expect(sortRunsNewestFirst(undefined as unknown as Scorecard[])).toEqual([]);
  });

  it('collects every category across runs, alphabetically', () => {
    expect(allCategories([older, newer])).toEqual(['RAG-grounding', 'routing']);
  });
});

describe('categoryRows', () => {
  it('computes deltas against the previous run and flags drops', () => {
    const rows = categoryRows(newer, older);
    const rag = rows.find((r) => r.category === 'RAG-grounding');
    const routing = rows.find((r) => r.category === 'routing');

    expect(rag?.delta).toBeCloseTo(-0.3, 5);
    expect(rag?.isRegression).toBe(true);
    expect(routing?.delta).toBeCloseTo(0.1, 5);
    expect(routing?.isRegression).toBe(false);
  });

  it('leaves delta null with no baseline', () => {
    const rows = categoryRows(newer, null);
    // Backend-flagged regressions still show even without a local baseline.
    expect(rows.find((r) => r.category === 'routing')?.delta).toBeNull();
    expect(rows.find((r) => r.category === 'RAG-grounding')?.isRegression).toBe(true);
  });

  it('clamps nonsense pass rates and tolerates missing fields', () => {
    const weird = card({
      run_id: 'x',
      ts: '2026-01-03T10:00:00Z',
      scores_by_category: {
        a: { pass_rate: 1.8, passed: 5, total: 5 },
        b: { pass_rate: Number.NaN as number, passed: 0, total: 0 },
      },
    });
    const rows = categoryRows(weird, null);
    expect(rows.find((r) => r.category === 'a')?.passRate).toBe(1);
    expect(rows.find((r) => r.category === 'b')?.passRate).toBe(0);
  });

  it('returns nothing for a null run', () => {
    expect(categoryRows(null)).toEqual([]);
  });
});

describe('overallPassRate', () => {
  it('weights by scenario count, not by category', () => {
    const c = card({
      run_id: 'w',
      ts: '2026-01-01T00:00:00Z',
      scores_by_category: {
        big: { pass_rate: 0.5, passed: 10, total: 20 },
        small: { pass_rate: 1, passed: 2, total: 2 },
      },
    });
    const { rate, passed, total } = overallPassRate(c);
    expect(passed).toBe(12);
    expect(total).toBe(22);
    expect(rate).toBeCloseTo(12 / 22, 6);
  });

  it('is zero, not NaN, with no scenarios', () => {
    expect(overallPassRate(null).rate).toBe(0);
  });
});

describe('regressionsOf', () => {
  it('keeps only genuine drops and sorts by severity', () => {
    const c = card({
      run_id: 'r',
      ts: '2026-01-01T00:00:00Z',
      regressions: [
        { category: 'small-drop', previous: 0.9, current: 0.85 },
        { category: 'not-a-drop', previous: 0.5, current: 0.7 },
        { category: 'big-drop', previous: 0.9, current: 0.3 },
        { category: 'malformed', previous: Number.NaN as number, current: 0.1 },
      ],
    });
    expect(regressionsOf(c).map((r) => r.category)).toEqual(['big-drop', 'small-drop']);
  });

  it('handles a missing regressions array', () => {
    expect(regressionsOf(card({ run_id: 'n', ts: '2026-01-01T00:00:00Z', regressions: undefined as never }))).toEqual([]);
  });
});

describe('trendSeries', () => {
  it('returns oldest-to-newest points per category', () => {
    const { runs, series } = trendSeries([newer, older], 10);
    expect(runs.map((r) => r.run_id)).toEqual(['run_001', 'run_002']);
    const routing = series.find((s) => s.category === 'routing');
    expect(routing?.points.map((p) => p.value)).toEqual([0.9, 1]);
  });

  it('windows to the last N runs', () => {
    const many = [1, 2, 3, 4, 5].map((i) =>
      card({
        run_id: `run_00${i}`,
        ts: `2026-01-0${i}T00:00:00Z`,
        scores_by_category: { routing: { pass_rate: i / 5, passed: i, total: 5 } },
      }),
    );
    const { runs } = trendSeries(many, 3);
    expect(runs.map((r) => r.run_id)).toEqual(['run_003', 'run_004', 'run_005']);
  });

  it('emits null (a gap) rather than a fabricated zero for a missing category', () => {
    const withExtra = card({
      run_id: 'run_003',
      ts: '2026-01-03T10:00:00Z',
      scores_by_category: { routing: { pass_rate: 1, passed: 10, total: 10 } },
    });
    const { series } = trendSeries([older, newer, withExtra], 10);
    const rag = series.find((s) => s.category === 'RAG-grounding');
    expect(rag?.points.map((p) => p.value)).toEqual([0.8, 0.5, null]);
  });
});

describe('filterDetails', () => {
  const details = [
    { scenario_id: 'a', category: 'routing', layer: 'routing' as const, passed: true },
    { scenario_id: 'b', category: 'routing', layer: 'routing' as const, passed: false },
    { scenario_id: 'c', category: 'RAG-grounding', layer: 'task' as const, passed: false },
  ];

  it('filters by category, status and layer', () => {
    expect(filterDetails(details, { status: 'fail' }).map((d) => d.scenario_id)).toEqual(['b', 'c']);
    expect(filterDetails(details, { category: 'routing' }).map((d) => d.scenario_id)).toEqual(['a', 'b']);
    expect(filterDetails(details, { layer: 'task' }).map((d) => d.scenario_id)).toEqual(['c']);
    expect(filterDetails(details, { category: 'routing', status: 'pass' }).map((d) => d.scenario_id)).toEqual(['a']);
  });

  it('returns everything by default and nothing for a missing list', () => {
    expect(filterDetails(details)).toHaveLength(3);
    expect(filterDetails(undefined)).toEqual([]);
  });

  it('lists the categories present in the details', () => {
    expect(detailCategories(details)).toEqual(['RAG-grounding', 'routing']);
  });
});
