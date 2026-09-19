import type { Metrics } from '../../api/types';
import { agentLabel, formatMs, formatPercent, formatTokens } from '../../lib/format';
import { agentColorVar } from '../Chat/agentColor';

/**
 * Aggregate health strip: per-agent p50/p95 and per-tool success.
 * Numbers are always shown next to the meters, so the bars are decoration
 * rather than the only carrier of the value.
 */
export function MetricsStrip({ metrics }: { metrics: Metrics | null }) {
  if (!metrics) return null;
  const latency = Array.isArray(metrics.latency_by_agent) ? metrics.latency_by_agent : [];
  const tools = Array.isArray(metrics.tool_success) ? metrics.tool_success : [];
  const tokens = metrics.tokens ?? null;

  if (!latency.length && !tools.length && !tokens) return null;

  return (
    <div className="metrics">
      <div className="metrics__block">
        <div className="metrics__title">Latency by agent (p50 / p95)</div>
        {latency.length === 0 ? (
          <div className="muted" style={{ fontSize: 'var(--fs-sm)' }}>No calls recorded yet.</div>
        ) : (
          latency.map((a) => (
            <div className="metricrow" key={a.agent}>
              <span className="metricrow__name">
                <span
                  className="agentpip__dot"
                  style={{ ['--agent' as string]: agentColorVar(a.agent) }}
                  aria-hidden="true"
                />
                {agentLabel(a.agent)}
                <span className="muted" style={{ fontSize: 'var(--fs-xs)' }}>
                  · {a.calls ?? 0} calls
                </span>
              </span>
              <span className="metricrow__nums">
                <span>
                  p50 <b>{formatMs(a.p50_ms)}</b>
                </span>
                <span>
                  p95 <b>{formatMs(a.p95_ms)}</b>
                </span>
              </span>
            </div>
          ))
        )}
      </div>

      <div className="metrics__block">
        <div className="metrics__title">Tool success rate</div>
        {tools.length === 0 ? (
          <div className="muted" style={{ fontSize: 'var(--fs-sm)' }}>No tool calls recorded yet.</div>
        ) : (
          tools.map((t) => {
            const rate =
              typeof t.success_rate === 'number' && Number.isFinite(t.success_rate)
                ? Math.min(1, Math.max(0, t.success_rate))
                : t.calls
                  ? (t.ok ?? 0) / t.calls
                  : 0;
            const tone = rate >= 0.9 ? '' : rate >= 0.7 ? ' meter__fill--warn' : ' meter__fill--bad';
            return (
              <div className="metricrow" key={t.tool}>
                <span className="metricrow__name mono" style={{ fontSize: 'var(--fs-sm)' }}>
                  {t.tool}
                </span>
                <span className="metricrow__nums" style={{ alignItems: 'center' }}>
                  <span className="meter" role="img" aria-label={`${formatPercent(rate)} success`}>
                    <span className={`meter__fill${tone}`} style={{ width: `${rate * 100}%` }} />
                  </span>
                  <span>
                    <b>{formatPercent(rate)}</b> {t.ok ?? 0}/{t.calls ?? 0}
                  </span>
                </span>
              </div>
            );
          })
        )}
      </div>

      <div className="metrics__block">
        <div className="metrics__title">Tokens</div>
        <div className="stat">
          <div className="stat__label">Avg / session</div>
          <div className="stat__value">{formatTokens(tokens?.avg_tokens_per_session ?? null)}</div>
        </div>
        <div className="stat" style={{ marginTop: 'var(--sp-3)' }}>
          <div className="stat__label">Sessions</div>
          <div className="stat__value">{tokens?.sessions ?? '—'}</div>
        </div>
      </div>
    </div>
  );
}
