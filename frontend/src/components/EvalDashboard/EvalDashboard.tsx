import { useEffect, useMemo, useState } from 'react';
import './EvalDashboard.css';
import { api } from '../../api/client';
import type { Scorecard } from '../../api/types';
import { useAsync } from '../../hooks/useAsync';
import {
  allCategories,
  categoryRows,
  detailCategories,
  filterDetails,
  overallPassRate,
  regressionsOf,
  sortRunsNewestFirst,
} from '../../lib/evalSelectors';
import type { PassFilter } from '../../lib/evalSelectors';
import { formatDateTime, formatPercent } from '../../lib/format';
import { EmptyState, ErrorBanner, Spinner } from '../shared/ui';
import { seriesColor, TrendChart } from './TrendChart';

const POLL_MS = 2500;

function deltaLabel(delta: number | null): string {
  if (delta === null) return '';
  const pts = delta * 100;
  if (Math.abs(pts) < 0.5) return '±0';
  return `${pts > 0 ? '+' : '−'}${Math.abs(pts).toFixed(0)} pts`;
}

export function EvalDashboard({
  selectedRunId,
  onSelectRun,
}: {
  selectedRunId: string | null;
  onSelectRun: (runId: string | null) => void;
}) {
  const runsState = useAsync(() => api.evalRuns(), []);
  const [runs, setRuns] = useState<Scorecard[]>([]);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<unknown>(null);
  const [category, setCategory] = useState('all');
  const [status, setStatus] = useState<PassFilter>('all');

  useEffect(() => {
    if (runsState.data) setRuns(sortRunsNewestFirst(runsState.data));
  }, [runsState.data]);

  const selected = useMemo(() => {
    if (!runs.length) return null;
    if (selectedRunId) return runs.find((r) => r.run_id === selectedRunId) ?? runs[0] ?? null;
    return runs[0] ?? null;
  }, [runs, selectedRunId]);

  const previous = useMemo(() => {
    if (!selected) return null;
    if (selected.previous_run_id) {
      const byId = runs.find((r) => r.run_id === selected.previous_run_id);
      if (byId) return byId;
    }
    const idx = runs.findIndex((r) => r.run_id === selected.run_id);
    return idx >= 0 ? runs[idx + 1] ?? null : null;
  }, [runs, selected]);

  const rows = useMemo(() => categoryRows(selected, previous), [selected, previous]);
  const overall = useMemo(() => overallPassRate(selected), [selected]);
  const regressions = useMemo(() => regressionsOf(selected), [selected]);
  const catOrder = useMemo(() => allCategories(runs), [runs]);
  const details = useMemo(
    () => filterDetails(selected?.details, { category, status }),
    [selected, category, status],
  );
  const detailCats = useMemo(() => detailCategories(selected?.details), [selected]);

  async function runEval() {
    setRunning(true);
    setRunError(null);
    try {
      // The run is a background job on the server (a live run takes many minutes - far longer
      // than a browser or proxy keeps one request open), so start it and poll for the result.
      const started = await api.runEval();
      let job = started;
      while (job.status === 'running') {
        await new Promise((r) => setTimeout(r, POLL_MS));
        job = await api.evalJob(started.job_id);
      }
      if (job.status !== 'done' || !job.scorecard) {
        throw new Error(job.error ?? `eval job ended with status "${job.status}"`);
      }
      const card = job.scorecard;
      setRuns((prev) => sortRunsNewestFirst([card, ...prev.filter((r) => r.run_id !== card.run_id)]));
      onSelectRun(card.run_id);
    } catch (err) {
      setRunError(err);
    } finally {
      setRunning(false);
    }
  }

  const loading = runsState.loading && !runsState.data;

  return (
    <div className="view">
      <div className="eval">
        <header className="eval__head">
          <div className="eval__titles">
            <h1>Eval dashboard</h1>
            <div className="eval__sub">
              Two-layer harness: deterministic routing checks plus judge-scored task scenarios.
              Every run is diffed against the previous one.
            </div>
          </div>
          <div className="eval__actions">
            {runs.length > 1 ? (
              <label className="filter">
                Run
                <select
                  className="select"
                  value={selected?.run_id ?? ''}
                  onChange={(e) => onSelectRun(e.target.value || null)}
                >
                  {runs.map((r) => (
                    <option key={r.run_id} value={r.run_id}>
                      {r.run_id} · {formatDateTime(r.ts)}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            <button
              type="button"
              className="btn btn--primary"
              onClick={() => void runEval()}
              disabled={running}
            >
              {running ? <Spinner label="Running evaluation" /> : null}
              {running ? 'Running eval…' : 'Run eval'}
            </button>
          </div>
        </header>

        {runError ? <ErrorBanner error={runError} context="run the eval suite" onRetry={() => void runEval()} /> : null}
        {runsState.error ? (
          <ErrorBanner error={runsState.error} context="load eval runs" onRetry={runsState.reload} />
        ) : null}

        {running ? (
          <div className="banner banner--info" role="status">
            <div className="banner__body">
              <div className="banner__title">
                <Spinner /> Evaluating every scenario against the current config
              </div>
              <div className="banner__detail">
                Offline (mock) runs take seconds; a live run against the local model takes ~15 minutes. The
                page keeps polling, so it is safe to leave this tab open.
              </div>
            </div>
          </div>
        ) : null}

        {loading ? (
          <div className="card" style={{ padding: 'var(--sp-5)' }}>
            <span className="muted">
              <Spinner /> Loading scorecards…
            </span>
          </div>
        ) : null}

        {!loading && !selected && !runsState.error ? (
          <div className="card">
            <EmptyState
              title="No eval runs yet"
              hint="Run the suite to produce the first scorecard and start the trend."
            />
          </div>
        ) : null}

        {selected ? (
          <>
            {/* -------- summary + regressions -------- */}
            <div className="card" style={{ padding: 'var(--sp-4) var(--sp-5)' }}>
              <div className="eval__head" style={{ alignItems: 'center' }}>
                <div>
                  <div className="stat__label">Overall pass rate</div>
                  <div className="hero">
                    <span className="hero__value">{formatPercent(overall.rate, 1)}</span>
                    <span className="hero__unit">
                      {overall.passed}/{overall.total} scenarios
                    </span>
                  </div>
                </div>
                <div className="eval__runmeta" style={{ marginLeft: 'auto' }}>
                  <span>{selected.run_id}</span>
                  <span>{formatDateTime(selected.ts)}</span>
                  <span>config {selected.config_hash || '—'}</span>
                  {previous ? <span>vs {previous.run_id}</span> : <span>no baseline</span>}
                </div>
              </div>
            </div>

            {regressions.length ? (
              <div className="regress" role="alert">
                <div className="regress__title">
                  <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                    <path d="M8 1.8 1.9 13.2h12.2L8 1.8Z" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
                    <path d="M8 6.2v3.1" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
                    <circle cx="8" cy="11.2" r="0.85" fill="currentColor" />
                  </svg>
                  {regressions.length} regression{regressions.length === 1 ? '' : 's'} against{' '}
                  {previous?.run_id ?? 'the previous run'}
                </div>
                <ul className="regress__list">
                  {regressions.map((r) => (
                    <li className="regress__item" key={r.category}>
                      <span className="regress__cat">{r.category}</span>
                      <span className="regress__delta">
                        <span className="regress__prev">{formatPercent(r.previous)}</span>
                        <span aria-hidden="true">→</span>
                        <span className="regress__cur">{formatPercent(r.current)}</span>
                      </span>
                      <span className="regress__drop">
                        −{Math.abs((r.current - r.previous) * 100).toFixed(0)} pts
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <div className="banner banner--info" style={{ borderColor: 'var(--good-border)', background: 'var(--good-soft)', color: 'var(--good)' }}>
                <div className="banner__body">
                  <div className="banner__title">No regressions</div>
                  <div className="banner__detail" style={{ color: 'var(--good)' }}>
                    No category dropped against {previous?.run_id ?? 'the previous run'}.
                  </div>
                </div>
              </div>
            )}

            {/* -------- per-category cards -------- */}
            <div className="catgrid">
              {rows.map((row) => {
                const colorIdx = Math.max(0, catOrder.indexOf(row.category));
                return (
                  <div
                    className={`catcard${row.isRegression ? ' catcard--regressed' : ''}`}
                    key={row.category}
                    style={{ ['--series' as string]: seriesColor(colorIdx) }}
                  >
                    <div className="catcard__name">
                      <span className="catcard__swatch" aria-hidden="true" />
                      {row.category}
                      {row.isRegression ? (
                        <span className="badge badge--fail" style={{ marginLeft: 'auto' }}>
                          regressed
                        </span>
                      ) : null}
                    </div>
                    <div className="catcard__value">
                      <span className="catcard__rate">{formatPercent(row.passRate)}</span>
                      <span className="catcard__count">
                        {row.passed}/{row.total}
                      </span>
                      {row.delta !== null ? (
                        <span
                          className={`catcard__delta${
                            row.delta > 0 ? ' catcard__delta--up' : row.delta < 0 ? ' catcard__delta--down' : ''
                          }`}
                        >
                          {deltaLabel(row.delta)}
                        </span>
                      ) : null}
                    </div>
                    <div className="catcard__bar">
                      <div className="catcard__fill" style={{ width: `${row.passRate * 100}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* -------- trend -------- */}
            <div className="card chart">
              <div className="chart__head">
                <span className="chart__title">Pass rate by category</span>
                <span className="chart__note">
                  last {Math.min(runs.length, 10)} run{runs.length === 1 ? '' : 's'}, oldest to newest
                  · hover a run for exact values
                </span>
              </div>
              <TrendChart runs={runs} limit={10} />
            </div>

            {/* -------- details -------- */}
            <div className="card details">
              <div className="pane-head">
                <span className="pane-head__title">Scenario details</span>
                <span className="pane-head__meta">{selected.details?.length ?? 0} scenarios</span>
              </div>

              <div className="details__filters">
                <label className="filter">
                  Category
                  <select
                    className="select"
                    value={category}
                    onChange={(e) => setCategory(e.target.value)}
                  >
                    <option value="all">All</option>
                    {detailCats.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </label>

                <div className="filter">
                  Result
                  <div className="segmented" role="group" aria-label="Filter by result">
                    {(['all', 'pass', 'fail'] as PassFilter[]).map((s) => (
                      <button
                        key={s}
                        type="button"
                        aria-pressed={status === s}
                        onClick={() => setStatus(s)}
                      >
                        {s === 'all' ? 'All' : s === 'pass' ? 'Passed' : 'Failed'}
                      </button>
                    ))}
                  </div>
                </div>

                <span className="details__count">
                  {details.length} shown
                </span>
              </div>

              <div className="dtable-wrap">
                <table className="dtable">
                  <thead>
                    <tr>
                      <th scope="col">Scenario</th>
                      <th scope="col">Category</th>
                      <th scope="col">Layer</th>
                      <th scope="col">Result</th>
                      <th scope="col">Expected</th>
                      <th scope="col">Actual</th>
                      <th scope="col">Rationale</th>
                    </tr>
                  </thead>
                  <tbody>
                    {details.length === 0 ? (
                      <tr>
                        <td colSpan={7}>
                          <span className="muted">No scenarios match this filter.</span>
                        </td>
                      </tr>
                    ) : null}
                    {details.map((d) => (
                      <tr key={`${d.scenario_id}-${d.category}`}>
                        <td className="dtable__id">{d.scenario_id}</td>
                        <td>{d.category}</td>
                        <td>
                          <span className="chip">{d.layer}</span>
                        </td>
                        <td>
                          <span className={`badge ${d.passed ? 'badge--ok' : 'badge--fail'}`}>
                            {d.passed ? 'pass' : 'fail'}
                          </span>
                        </td>
                        <td className="dtable__exp">{d.expected ?? '—'}</td>
                        <td className={`dtable__act${d.passed ? '' : ' dtable__act--fail'}`}>
                          {d.actual ?? '—'}
                        </td>
                        <td className="dtable__why">{d.rationale ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
