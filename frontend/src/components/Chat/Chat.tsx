import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import './Chat.css';
import { api, ApiError } from '../../api/client';
import type { PendingConfirmation, RouteDecision, SessionSummary } from '../../api/types';
import { relativeTime, shortId } from '../../lib/format';
import { EmptyState, ErrorBanner, Spinner } from '../shared/ui';
import { ConfirmationCard } from './ConfirmationCard';
import { agentColorVar } from './agentColor';

const SESSION_KEY = 'jarvis.session_id';
const USER_KEY = 'jarvis.user_id';

function readStored(key: string): string {
  try {
    return window.localStorage.getItem(key) ?? '';
  } catch {
    return '';
  }
}

function writeStored(key: string, value: string): void {
  try {
    if (value) window.localStorage.setItem(key, value);
    else window.localStorage.removeItem(key);
  } catch {
    /* private mode / blocked storage — the app still works, it just forgets. */
  }
}

interface UiMessage {
  key: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  createdAt: string;
  turnId?: string;
  routes?: RouteDecision[];
  truncated?: boolean;
  confirmations?: PendingConfirmation[];
}

let localKey = 0;
const nextKey = () => `m${++localKey}`;

function RouteChips({ routes }: { routes: RouteDecision[] }) {
  if (!routes.length) return null;
  return (
    <>
      {routes.map((r, i) => (
        <span
          className="routechip"
          key={`${r.agent}-${i}`}
          style={{ ['--agent' as string]: agentColorVar(r.agent) }}
          title={r.reason ? `${r.agent}: ${r.reason}` : r.agent}
        >
          <span className="routechip__dot" aria-hidden="true" />
          <span className="routechip__agent">{r.agent || 'unknown'}</span>
          {r.reason ? <span className="routechip__reason">{r.reason}</span> : null}
        </span>
      ))}
    </>
  );
}

export function Chat({ onOpenTrace }: { onOpenTrace: (turnId: string) => void }) {
  const [sessionId, setSessionId] = useState<string>(() => readStored(SESSION_KEY));
  const [userId, setUserId] = useState<string>(() => readStored(USER_KEY));
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [sending, setSending] = useState(false);
  const [sendError, setSendError] = useState<unknown>(null);
  const [historyError, setHistoryError] = useState<unknown>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionsError, setSessionsError] = useState<unknown>(null);
  const [loadingHistory, setLoadingHistory] = useState(false);

  const threadRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  const loadSessions = useCallback(async () => {
    try {
      setSessions(await api.sessions());
      setSessionsError(null);
    } catch (err) {
      setSessionsError(err);
    }
  }, []);

  useEffect(() => {
    void loadSessions();
  }, [loadSessions]);

  // Hydrate the thread whenever the active session changes.
  useEffect(() => {
    let cancelled = false;
    if (!sessionId) {
      setMessages([]);
      setHistoryError(null);
      return;
    }
    setLoadingHistory(true);
    Promise.all([api.sessionMessages(sessionId), api.confirmations(sessionId).catch(() => [])])
      .then(([stored, pending]) => {
        if (cancelled) return;
        const hydrated: UiMessage[] = stored.map((m) => ({
          key: nextKey(),
          role: m.role === 'user' ? 'user' : m.role === 'system' ? 'system' : 'assistant',
          content: m.content ?? '',
          createdAt: m.created_at,
        }));
        // Park any still-pending confirmation on the last assistant message.
        if (pending.length) {
          for (let i = hydrated.length - 1; i >= 0; i -= 1) {
            const msg = hydrated[i];
            if (msg && msg.role === 'assistant') {
              msg.confirmations = pending;
              break;
            }
          }
          if (!hydrated.some((m) => m.confirmations?.length)) {
            hydrated.push({
              key: nextKey(),
              role: 'assistant',
              content: 'An action from an earlier turn is still waiting on your approval.',
              createdAt: new Date().toISOString(),
              confirmations: pending,
            });
          }
        }
        setMessages(hydrated);
        setHistoryError(null);
      })
      .catch((err: unknown) => {
        if (!cancelled) setHistoryError(err);
      })
      .finally(() => {
        if (!cancelled) setLoadingHistory(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  // Keep the newest message in view.
  useEffect(() => {
    const el = threadRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, sending]);

  const newSession = useCallback(() => {
    writeStored(SESSION_KEY, '');
    setSessionId('');
    setMessages([]);
    setSendError(null);
    setHistoryError(null);
    inputRef.current?.focus();
  }, []);

  const send = useCallback(async () => {
    const text = draft.trim();
    if (!text || sending) return;
    setDraft('');
    setSendError(null);
    setSending(true);
    setMessages((prev) => [
      ...prev,
      { key: nextKey(), role: 'user', content: text, createdAt: new Date().toISOString() },
    ]);

    try {
      const res = await api.chat({
        message: text,
        ...(sessionId ? { session_id: sessionId } : {}),
        ...(userId.trim() ? { user_id: userId.trim() } : {}),
      });
      if (res?.session_id && res.session_id !== sessionId) {
        setSessionId(res.session_id);
        writeStored(SESSION_KEY, res.session_id);
      }
      setMessages((prev) => [
        ...prev,
        {
          key: nextKey(),
          role: 'assistant',
          content:
            res?.text ??
            'The agent returned no text for this turn. Open the trace to see where it stopped.',
          createdAt: new Date().toISOString(),
          turnId: res?.turn_id,
          routes: Array.isArray(res?.route_decisions) ? res.route_decisions : [],
          truncated: res?.truncated === true,
          confirmations: Array.isArray(res?.pending_confirmations)
            ? res.pending_confirmations
            : [],
        },
      ]);
      void loadSessions();
    } catch (err) {
      setSendError(err);
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }, [draft, sending, sessionId, userId, loadSessions]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  };

  const activeSession = useMemo(
    () => sessions.find((s) => s.id === sessionId) ?? null,
    [sessions, sessionId],
  );

  return (
    <div className="view">
      <div className="chat">
        <aside className="chat__rail card">
          <div className="chat__railhead">
            <span className="chat__railtitle">Sessions</span>
            <button
              type="button"
              className="btn btn--sm"
              style={{ marginLeft: 'auto' }}
              onClick={newSession}
            >
              New
            </button>
          </div>

          <div className="chat__raillist">
            {/* The rail is narrow, so it gets a compact notice; the full banner
                (with the "start the backend" hint) lives in the thread pane. */}
            {sessionsError ? (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 'var(--sp-2)',
                  padding: 'var(--sp-2) var(--sp-3)',
                  fontSize: 'var(--fs-sm)',
                  color: 'var(--critical-ink)',
                  background: 'var(--critical-soft)',
                  border: '1px solid var(--critical-border)',
                  borderRadius: 'var(--r-md)',
                }}
              >
                <span>Sessions unavailable</span>
                <button
                  type="button"
                  className="btn btn--sm"
                  style={{ marginLeft: 'auto' }}
                  onClick={() => void loadSessions()}
                >
                  Retry
                </button>
              </div>
            ) : null}
            {!sessionsError && sessions.length === 0 ? (
              <div className="muted" style={{ padding: 'var(--sp-3)', fontSize: 'var(--fs-sm)' }}>
                No sessions yet.
              </div>
            ) : null}
            {sessions.map((s) => (
              <button
                type="button"
                key={s.id}
                className="sessionitem"
                aria-current={s.id === sessionId}
                onClick={() => {
                  setSessionId(s.id);
                  writeStored(SESSION_KEY, s.id);
                }}
              >
                <span className="sessionitem__preview">{s.preview || 'Untitled session'}</span>
                <span className="sessionitem__meta">
                  <span>{shortId(s.id, 12)}</span>
                  <span>·</span>
                  <span>{relativeTime(s.created_at)}</span>
                </span>
              </button>
            ))}
          </div>

          <div className="chat__railfoot">
            <label className="chat__field">
              <span className="chat__label">User id (optional)</span>
              <input
                className="input"
                value={userId}
                placeholder="e.g. prajwal"
                onChange={(e) => {
                  setUserId(e.target.value);
                  writeStored(USER_KEY, e.target.value);
                }}
              />
            </label>
            <div className="muted" style={{ fontSize: 'var(--fs-xs)' }}>
              Active session:{' '}
              <span className="mono">{sessionId ? shortId(sessionId, 16) : 'new'}</span>
              {activeSession?.user_id ? ` · ${activeSession.user_id}` : ''}
            </div>
          </div>
        </aside>

        <section className="chat__main card" aria-label="Conversation">
          {sendError || historyError || sessionsError ? (
            <div className="chat__banners" style={{ paddingTop: 'var(--sp-4)' }}>
              {sendError ? (
                <ErrorBanner error={sendError} context="send that message" onRetry={() => setSendError(null)} />
              ) : null}
              {historyError ? (
                <ErrorBanner
                  error={historyError}
                  context="load this session's history"
                  onRetry={() => setSessionId((s) => s)}
                />
              ) : !sendError && sessionsError ? (
                <ErrorBanner
                  error={sessionsError}
                  context="reach the backend"
                  onRetry={() => void loadSessions()}
                />
              ) : null}
            </div>
          ) : null}

          <div className="chat__thread" ref={threadRef}>
            {loadingHistory ? (
              <div className="muted">
                <Spinner /> Loading conversation…
              </div>
            ) : null}

            {!loadingHistory && messages.length === 0 ? (
              <EmptyState
                title="Ask Jarvis something"
                hint="The orchestrator routes your request to the Code, Research or Ops agent — every step is traced."
              >
                <div style={{ marginTop: 'var(--sp-3)', display: 'grid', gap: 'var(--sp-2)', maxWidth: 520 }}>
                  {[
                    'Why is checkout latency up since Tuesday? Check the repo and the runbook.',
                    'Summarise the retry policy in our docs and email the on-call a digest.',
                    'Generate pytest cases for billing/proration.py',
                  ].map((s) => (
                    <button
                      key={s}
                      type="button"
                      className="btn"
                      style={{ height: 'auto', padding: 'var(--sp-2) var(--sp-4)', textAlign: 'left', justifyContent: 'flex-start', whiteSpace: 'normal' }}
                      onClick={() => {
                        setDraft(s);
                        inputRef.current?.focus();
                      }}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </EmptyState>
            ) : null}

            {messages.map((m) => (
              <article key={m.key} className={`msg msg--${m.role}`}>
                <div className="msg__role">
                  <span>{m.role === 'user' ? 'You' : m.role === 'system' ? 'System' : 'Jarvis'}</span>
                  {m.truncated ? <span className="badge badge--warn">truncated</span> : null}
                </div>
                <div className="msg__bubble">{m.content}</div>

                {m.role === 'assistant' && (m.routes?.length || m.turnId) ? (
                  <div className="msg__footer">
                    {m.routes?.length ? <RouteChips routes={m.routes} /> : null}
                    {m.turnId ? (
                      <button
                        type="button"
                        className="msg__tracelink"
                        onClick={() => onOpenTrace(m.turnId as string)}
                      >
                        View trace
                        <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
                          <path
                            d="M3.5 1.5 7 5l-3.5 3.5"
                            stroke="currentColor"
                            strokeWidth="1.6"
                            fill="none"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </svg>
                      </button>
                    ) : null}
                  </div>
                ) : null}

                {m.confirmations?.length
                  ? m.confirmations.map((c) => (
                      <ConfirmationCard
                        key={c.id}
                        confirmation={c}
                        onResolved={() => void loadSessions()}
                      />
                    ))
                  : null}
              </article>
            ))}

            {sending ? (
              <div className="msg msg--assistant" aria-live="polite">
                <div className="msg__role">Jarvis</div>
                <div className="typing">
                  <span className="typing__dot" />
                  <span className="typing__dot" />
                  <span className="typing__dot" />
                  <span className="visually-hidden">Jarvis is working…</span>
                </div>
              </div>
            ) : null}
          </div>

          <div className="composer">
            <div className="composer__row">
              <textarea
                ref={inputRef}
                className="composer__input"
                value={draft}
                rows={1}
                placeholder="Ask about the repo, the docs, or an operation to schedule…"
                aria-label="Message"
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={onKeyDown}
                disabled={sending}
              />
              <button
                type="button"
                className="btn btn--primary"
                onClick={() => void send()}
                disabled={sending || !draft.trim()}
              >
                {sending ? <Spinner label="Sending" /> : null}
                Send
              </button>
            </div>
            <div className="composer__hint">
              <span>
                <kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for a new line
              </span>
              <span>Side-effecting tools pause for your approval.</span>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

export type { UiMessage };
export { ApiError };
