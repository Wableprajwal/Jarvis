import { useEffect, useMemo, useRef, useState } from 'react';
import type { Scorecard } from '../../api/types';
import { trendSeries } from '../../lib/evalSelectors';
import { formatDateTime, formatPercent, shortId } from '../../lib/format';

/**
 * Pass rate per category over the last N runs — hand-drawn SVG, no chart lib.
 *
 * Design notes (why it looks the way it does):
 *  - Categorical colours are assigned in fixed slot order and never cycled; a
 *    category keeps its colour when other categories are added or removed.
 *  - One y-axis only (pass rate, 0–100%). Never a second scale.
 *  - Gaps are gaps: a run missing a category breaks the line rather than
 *    plotting a fabricated zero.
 *  - Identity is never colour-alone: a legend is always present, series are
 *    direct-labelled when there are four or fewer, and the category cards plus
 *    the details table below carry every number in text.
 *  - Hover exposes a crosshair and a tooltip listing every series for that run.
 */

export const SERIES_VARS = [
  'var(--series-1)',
  'var(--series-2)',
  'var(--series-3)',
  'var(--series-4)',
  'var(--series-5)',
] as const;

/** Fixed-order colour for a category — stable across filters and runs. */
export function seriesColor(index: number): string {
  return SERIES_VARS[index % SERIES_VARS.length] as string;
}

const MIN_W = 520;
const H = 260;
/** Right padding only needs to hold the direct labels, which appear at ≤ 4 series. */
const padFor = (directLabel: boolean) => ({
  top: 18,
  right: directLabel ? 108 : 26,
  bottom: 34,
  left: 44,
});

export function TrendChart({ runs, limit = 10 }: { runs: Scorecard[]; limit?: number }) {
  const { runs: window, series } = useMemo(() => trendSeries(runs, limit), [runs, limit]);
  const [hover, setHover] = useState<number | null>(null);

  // Draw at real pixel size rather than scaling a fixed viewBox: that keeps the
  // labels at a readable size at every width and stops the chart from growing
  // absurdly tall on a wide screen.
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [measured, setMeasured] = useState(760);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const update = () => setMeasured(el.clientWidth || 760);
    update();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const W = Math.max(MIN_W, measured);
  const directLabel = series.length > 0 && series.length <= 4;
  const PAD = padFor(directLabel);
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  if (window.length === 0 || series.length === 0) {
    return (
      <div className="muted" style={{ padding: 'var(--sp-5)' }}>
        No runs to chart yet.
      </div>
    );
  }

  const n = window.length;
  const x = (i: number) => PAD.left + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const y = (v: number) => PAD.top + (1 - v) * plotH;

  const gridLines = [0, 0.25, 0.5, 0.75, 1];

  // Build one path per series, breaking the line across null points.
  const paths = series.map((s) => {
    let d = '';
    let penDown = false;
    s.points.forEach((p, i) => {
      if (p.value === null) {
        penDown = false;
        return;
      }
      d += `${penDown ? 'L' : 'M'}${x(i).toFixed(1)} ${y(p.value).toFixed(1)} `;
      penDown = true;
    });
    return d.trim();
  });

  const summary = series
    .map((s) => {
      const last = [...s.points].reverse().find((p) => p.value !== null);
      return `${s.category} ${last ? formatPercent(last.value) : 'no data'}`;
    })
    .join('; ');

  const hoveredRun = hover !== null ? window[hover] : null;

  // Park the tooltip in whichever corner is furthest from the data it describes:
  // horizontally opposite the hovered column, vertically away from that run's values.
  const hoveredValues =
    hover === null
      ? []
      : series.map((s) => s.points[hover]?.value).filter((v): v is number => v !== null && v !== undefined);
  const hoveredMean = hoveredValues.length
    ? hoveredValues.reduce((a, b) => a + b, 0) / hoveredValues.length
    : 0.5;
  const tipStyle: Record<string, string | number> =
    (hover ?? 0) < n / 2 ? { right: 14, left: 'auto' } : { left: PAD.left + 6, right: 'auto' };
  if (hoveredMean >= 0.5) tipStyle['bottom'] = PAD.bottom + 6;
  else tipStyle['top'] = PAD.top;

  return (
    <div>
      <div className="chart__scroll" ref={wrapRef}>
        <svg
          className="chart__svg"
          width={W}
          height={H}
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Pass rate per category over the last ${n} eval runs. Latest: ${summary}.`}
          onMouseLeave={() => setHover(null)}
        >
          {/* y grid + labels */}
          {gridLines.map((g) => (
            <g key={g}>
              <line
                x1={PAD.left}
                x2={PAD.left + plotW}
                y1={y(g)}
                y2={y(g)}
                stroke="var(--chart-grid)"
                strokeWidth={1}
              />
              <text
                x={PAD.left - 8}
                y={y(g) + 3.5}
                textAnchor="end"
                fontSize={10}
                fill="var(--text-3)"
                fontFamily="var(--font-mono)"
              >
                {Math.round(g * 100)}%
              </text>
            </g>
          ))}

          {/* x axis */}
          <line
            x1={PAD.left}
            x2={PAD.left + plotW}
            y1={y(0)}
            y2={y(0)}
            stroke="var(--chart-axis)"
            strokeWidth={1}
          />

          {/* crosshair */}
          {hover !== null ? (
            <line
              x1={x(hover)}
              x2={x(hover)}
              y1={PAD.top}
              y2={PAD.top + plotH}
              stroke="var(--chart-axis)"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
          ) : null}

          {/* run labels */}
          {window.map((r, i) => (
            <text
              key={r.run_id}
              x={x(i)}
              y={H - 12}
              textAnchor="middle"
              fontSize={10}
              fill={hover === i ? 'var(--text)' : 'var(--text-3)'}
              fontFamily="var(--font-mono)"
            >
              {shortId(r.run_id, 8).replace('…', '')}
            </text>
          ))}

          {/* series lines */}
          {series.map((s, si) => (
            <path
              key={s.category}
              d={paths[si] ?? ''}
              fill="none"
              stroke={seriesColor(si)}
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ))}

          {/* markers: 8px, ringed in the surface colour so overlaps stay legible */}
          {series.map((s, si) =>
            s.points.map((p, i) =>
              p.value === null ? null : (
                <circle
                  key={`${s.category}-${i}`}
                  cx={x(i)}
                  cy={y(p.value)}
                  r={hover === i ? 5 : 4}
                  fill={seriesColor(si)}
                  stroke="var(--surface)"
                  strokeWidth={2}
                />
              ),
            ),
          )}

          {/* direct labels at the line end (only when few enough not to collide) */}
          {directLabel
            ? series.map((s) => {
                const lastIdx = [...s.points]
                  .map((p, i) => ({ p, i }))
                  .reverse()
                  .find((e) => e.p.value !== null);
                if (!lastIdx) return null;
                return (
                  <text
                    key={`label-${s.category}`}
                    x={x(lastIdx.i) + 10}
                    y={y(lastIdx.p.value as number) + 3.5}
                    fontSize={11}
                    fill="var(--text-2)"
                    fontWeight={600}
                  >
                    {s.category}
                  </text>
                );
              })
            : null}

          {/* hover hit targets, wider than the marks */}
          {window.map((r, i) => (
            <rect
              key={`hit-${r.run_id}`}
              x={x(i) - plotW / Math.max(1, n * 2)}
              y={PAD.top}
              width={Math.max(24, plotW / Math.max(1, n))}
              height={plotH}
              fill="transparent"
              onMouseEnter={() => setHover(i)}
              onFocus={() => setHover(i)}
              tabIndex={0}
              role="button"
              aria-label={`Run ${r.run_id}, ${formatDateTime(r.ts)}`}
            />
          ))}
        </svg>

        {hoveredRun ? (
          <div
            className="chart__tip"
            style={tipStyle}
          >
            <div className="chart__tiphead">
              {hoveredRun.run_id} · {formatDateTime(hoveredRun.ts)}
            </div>
            {series.map((s, si) => {
              const p = s.points[hover as number];
              return (
                <div className="chart__tiprow" key={s.category}>
                  <span
                    className="chart__tipdot"
                    style={{ background: seriesColor(si) }}
                    aria-hidden="true"
                  />
                  <span>{s.category}</span>
                  <span className="chart__tipval">
                    {p && p.value !== null ? formatPercent(p.value) : '—'}
                  </span>
                </div>
              );
            })}
          </div>
        ) : null}
      </div>

      <div className="legend">
        {series.map((s, si) => (
          <span className="legend__item" key={s.category}>
            <span
              className="legend__swatch"
              style={{ background: seriesColor(si) }}
              aria-hidden="true"
            />
            {s.category}
          </span>
        ))}
      </div>
    </div>
  );
}
