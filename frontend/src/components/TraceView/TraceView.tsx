import { useEffect, useMemo } from 'react';
import './TraceView.css';
import { api } from '../../api/client';
import { useAsync } from '../../hooks/useAsync';
import { summarizeTurn } from '../../lib/trace';
import { formatMs, formatTime, formatTokens, relativeTime, shortId } from '../../lib/format';
import { EmptyState, ErrorBanner, Spinner } from '../shared/ui';
import { agentColorVar } from '../Chat/agentColor';
import { MetricsStrip } from './MetricsStrip';
import { Timeline } from './Timeline';

export function TraceView({
  selectedTurnId,
  onSelectTurn,
}: {
  selectedTurnId: string | null;
  onSelectTurn: (turnId: string) => void;
}) {
  const turns = useAsync(() => api.turns({ limit: 50 }), []);
  const metrics = useAsync(() => api.metrics(), []);

  const turnList = useMemo(() => turns.data ?? [], [turns.data]);

  // Default to the newest turn so the pane is never empty on arrival.
  useEffect(() => {
    if (!selectedTurnId && turnList.length) {
      const first = turnList[0];
      if (first?.turn_id) onSelectTurn(first.turn_id);
    }
  }, [selectedTurnId, turnList, onSelectTurn]);

  const trace = useAsync(
    () => api.turnTrace(selectedTurnId as string),
    [selectedTurnId],
    { enabled: !!selectedTurnId },
  );

  const events = useMemo(() => {
    const raw = trace.data?.events;
    return Array.isArray(raw) ? raw : [];
  }, [trace.data]);

  const stats = useMemo(() => summarizeTurn(events), [events]);

  return (
    <div className="view">
      <div className="trace">
        <aside className="trace__rail card">
          <div className="pane-head">
            <span className="pane-head__title">Turns</span>
            <span className="pane-head__meta">{turnList.length || 0}</span>
            <button
              type="button"
              className="btn btn--sm"
              onClick={() => {
                turns.reload();
                metrics.reload();
              }}
            >
              Refresh
            </button>
          </div>

          <div className="trace__raillist">
            {turns.loading && !turns.data ? (
              <div className="muted" style={{ padding: 'var(--sp-3)' }}>
                <Spinner /> Loading turns…
              </div>
            ) : null}

            {turns.error ? (
              <div style={{ padding: 'var(--sp-2)' }}>
                <ErrorBanner error={turns.error} context="load turns" onRetry={turns.reload} />
              </div>
            ) : null}

            {!turns.loading && !turns.error && turnList.length === 0 ? (
              <EmptyState title="No traced turns yet" hint="Send a message in Chat to produce one." />
            ) : null}

            {turnList.map((t) => (
              <button
                type="button"
                key={t.turn_id}
                className="turnitem"
                aria-current={t.turn_id === selectedTurnId}
                onClick={() => onSelectTurn(t.turn_id)}
              >
                <span className="turnitem__top">
                  <span className="turnitem__id">{shortId(t.turn_id, 18)}</span>
                  <span className="turnitem__time">{relativeTime(t.started)}</span>
                </span>
                <span className="turnitem__agents">
                  {(Array.isArray(t.routed) ? t.routed : []).map((a, i) => (
                    <span
                      className="agentpip"
                      key={`${a}-${i}`}
                      style={{ ['--agent' as string]: agentColorVar(a) }}
                    >
                      <span className="agentpip__dot" aria-hidden="true" />
                      {a}
                    </span>
                  ))}
                </span>
                <span className="turnitem__stats">
                  <span>{t.events ?? 0} events</span>
                  <span>{formatTokens(t.tokens)} tok</span>
                </span>
              </button>
            ))}
          </div>
        </aside>

        <section className="trace__detail card" aria-label="Turn trace">
          {!selectedTurnId ? (
            <EmptyState title="Select a turn" hint="Pick a turn on the left to inspect its timeline." />
          ) : (
            <>
              <header className="tsummary">
                <div className="tsummary__title">
                  <span className="tsummary__id">{selectedTurnId}</span>
                  {trace.loading ? (
                    <span className="muted">
                      <Spinner /> loading…
                    </span>
                  ) : null}
                  {stats.errorCount > 0 ? (
                    <span className="badge badge--fail">
                      {stats.errorCount} error{stats.errorCount === 1 ? '' : 's'}
                    </span>
                  ) : null}
                  {stats.pendingConfirmations > 0 ? (
                    <span className="badge badge--warn">awaiting approval</span>
                  ) : null}
                  <span className="pane-head__meta">{formatTime(stats.startedAt)}</span>
                </div>

                <div className="tsummary__stats">
                  <div className="stat">
                    <div className="stat__label">Latency</div>
                    <div className="stat__value">{formatMs(stats.totalLatencyMs)}</div>
                  </div>
                  <div className="stat">
                    <div className="stat__label">Tokens</div>
                    <div className="stat__value">{formatTokens(stats.totalTokens)}</div>
                  </div>
                  <div className="stat">
                    <div className="stat__label">Model calls</div>
                    <div className="stat__value">{stats.modelCalls}</div>
                  </div>
                  <div className="stat">
                    <div className="stat__label">Tool calls</div>
                    <div className={`stat__value${stats.toolFailures ? ' stat__value--bad' : ''}`}>
                      {stats.toolCalls}
                      {stats.toolFailures ? (
                        <span style={{ fontSize: 'var(--fs-sm)' }}> · {stats.toolFailures} failed</span>
                      ) : null}
                    </div>
                  </div>
                  <div className="stat">
                    <div className="stat__label">Citations</div>
                    <div className="stat__value">{stats.citations}</div>
                  </div>
                  <div className="stat" style={{ minWidth: 160 }}>
                    <div className="stat__label">Agents involved</div>
                    <div style={{ marginTop: 3, display: 'flex', flexWrap: 'wrap', gap: 'var(--sp-1)' }}>
                      {stats.agents.length === 0 ? (
                        <span className="muted">—</span>
                      ) : (
                        stats.agents.map((a) => (
                          <span
                            className="agentpip"
                            key={a}
                            style={{ ['--agent' as string]: agentColorVar(a) }}
                          >
                            <span className="agentpip__dot" aria-hidden="true" />
                            {a}
                          </span>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              </header>

              <MetricsStrip metrics={metrics.data} />

              <div className="trace__scroll">
                {trace.error ? (
                  <div style={{ padding: 'var(--sp-5)' }}>
                    <ErrorBanner error={trace.error} context="load this trace" onRetry={trace.reload} />
                  </div>
                ) : null}

                {!trace.error && !trace.loading && events.length === 0 ? (
                  <EmptyState
                    title="No events for this turn"
                    hint="The backend returned an empty event list."
                  />
                ) : null}

                {events.length ? <Timeline events={events} /> : null}
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
