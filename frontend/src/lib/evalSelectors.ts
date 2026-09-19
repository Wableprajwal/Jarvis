/**
 * Pure selectors over eval scorecards.
 *
 * Kept free of React so the regression / trend logic can be unit-tested and so
 * the dashboard never recomputes rules inline.
 */

import type { Regression, ScenarioDetail, Scorecard } from '../api/types';

function isFiniteNum(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

export interface CategoryRow {
  category: string;
  passRate: number;
  passed: number;
  total: number;
  /** Signed delta vs the previous run, or null when there is no comparison. */
  delta: number | null;
  isRegression: boolean;
}

/** Newest run first, tolerating a missing/short timestamp. */
export function sortRunsNewestFirst(runs: readonly Scorecard[]): Scorecard[] {
  return [...(runs ?? [])].filter(Boolean).sort((a, b) => {
    const ta = Date.parse(a?.ts ?? '');
    const tb = Date.parse(b?.ts ?? '');
    if (Number.isFinite(ta) && Number.isFinite(tb)) return tb - ta;
    return 0;
  });
}

export function latestRun(runs: readonly Scorecard[]): Scorecard | null {
  return sortRunsNewestFirst(runs)[0] ?? null;
}

/** All category names seen across the given runs, alphabetically. */
export function allCategories(runs: readonly Scorecard[]): string[] {
  const set = new Set<string>();
  for (const run of runs ?? []) {
    for (const cat of Object.keys(run?.scores_by_category ?? {})) set.add(cat);
  }
  return [...set].sort((a, b) => a.localeCompare(b));
}

/**
 * Per-category rows for a scorecard, annotated with the delta against `previous`
 * and whether the backend flagged that category as a regression.
 *
 * Deltas are computed here rather than trusted from `regressions` alone, because
 * an improvement is also worth showing — but the loud "REGRESSION" flag stays
 * driven by the backend's own list, with a numeric drop as a fallback.
 */
export function categoryRows(
  run: Scorecard | null | undefined,
  previous?: Scorecard | null,
): CategoryRow[] {
  if (!run) return [];
  const flagged = new Set(
    (Array.isArray(run.regressions) ? run.regressions : [])
      .map((r) => r?.category)
      .filter((c): c is string => typeof c === 'string'),
  );
  const scores = run.scores_by_category ?? {};
  const prevScores = previous?.scores_by_category ?? {};

  return Object.keys(scores)
    .sort((a, b) => a.localeCompare(b))
    .map((category) => {
      const s = scores[category];
      const passRate = isFiniteNum(s?.pass_rate) ? clamp01(s.pass_rate) : 0;
      const prev = prevScores[category];
      const delta = isFiniteNum(prev?.pass_rate) ? passRate - clamp01(prev.pass_rate) : null;
      return {
        category,
        passRate,
        passed: isFiniteNum(s?.passed) ? s.passed : 0,
        total: isFiniteNum(s?.total) ? s.total : 0,
        delta,
        isRegression: flagged.has(category) || (delta !== null && delta < 0),
      };
    });
}

export function clamp01(n: number): number {
  if (!Number.isFinite(n)) return 0;
  return Math.min(1, Math.max(0, n));
}

/** Overall pass rate across every category, weighted by scenario count. */
export function overallPassRate(run: Scorecard | null | undefined): {
  rate: number;
  passed: number;
  total: number;
} {
  let passed = 0;
  let total = 0;
  for (const s of Object.values(run?.scores_by_category ?? {})) {
    if (isFiniteNum(s?.passed)) passed += s.passed;
    if (isFiniteNum(s?.total)) total += s.total;
  }
  return { rate: total > 0 ? passed / total : 0, passed, total };
}

/** Normalised regression list: only genuine drops, biggest first. */
export function regressionsOf(run: Scorecard | null | undefined): Regression[] {
  const list = Array.isArray(run?.regressions) ? run!.regressions : [];
  return list
    .filter(
      (r): r is Regression =>
        !!r &&
        typeof r.category === 'string' &&
        isFiniteNum(r.previous) &&
        isFiniteNum(r.current) &&
        r.current < r.previous,
    )
    .sort((a, b) => a.current - a.previous - (b.current - b.previous));
}

export interface TrendPoint {
  runId: string;
  ts: string;
  value: number | null;
}

export interface TrendSeries {
  category: string;
  points: TrendPoint[];
}

/**
 * Pass-rate-per-category series over the last `limit` runs, oldest → newest,
 * so the chart reads left-to-right in time. A category missing from a run
 * yields a null point (a gap), never a fabricated zero.
 */
export function trendSeries(runs: readonly Scorecard[], limit = 10): {
  runs: Scorecard[];
  series: TrendSeries[];
} {
  const newestFirst = sortRunsNewestFirst(runs);
  const window = newestFirst.slice(0, Math.max(1, limit)).reverse();
  const cats = allCategories(window);
  const series = cats.map((category) => ({
    category,
    points: window.map((run) => {
      const s = run.scores_by_category?.[category];
      return {
        runId: run.run_id,
        ts: run.ts,
        value: isFiniteNum(s?.pass_rate) ? clamp01(s.pass_rate) : null,
      };
    }),
  }));
  return { runs: window, series };
}

export type PassFilter = 'all' | 'pass' | 'fail';

export function filterDetails(
  details: readonly ScenarioDetail[] | undefined | null,
  opts: { category?: string; status?: PassFilter; layer?: string } = {},
): ScenarioDetail[] {
  const { category = 'all', status = 'all', layer = 'all' } = opts;
  const list = Array.isArray(details) ? details : [];
  return list.filter((d) => {
    if (!d) return false;
    if (category !== 'all' && d.category !== category) return false;
    if (layer !== 'all' && d.layer !== layer) return false;
    if (status === 'pass' && d.passed !== true) return false;
    if (status === 'fail' && d.passed !== false) return false;
    return true;
  });
}

export function detailCategories(details: readonly ScenarioDetail[] | undefined | null): string[] {
  const set = new Set<string>();
  for (const d of details ?? []) if (d && typeof d.category === 'string') set.add(d.category);
  return [...set].sort((a, b) => a.localeCompare(b));
}
