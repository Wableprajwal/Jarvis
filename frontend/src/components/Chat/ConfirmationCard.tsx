import { useState } from 'react';
import { api, ApiError } from '../../api/client';
import type { PendingConfirmation } from '../../api/types';
import { prettyJson } from '../../lib/format';
import { Collapsible, JsonBlock, Spinner } from '../shared/ui';

function renderValue(v: unknown): string {
  if (typeof v === 'string') return v || '—';
  if (Array.isArray(v)) {
    if (!v.length) return '—';
    return v.map((x) => (typeof x === 'string' ? x : prettyJson(x))).join(', ');
  }
  if (v === null || v === undefined) return '—';
  if (typeof v === 'object') return prettyJson(v);
  return String(v);
}

/**
 * A confirmation-gated tool call awaiting a human decision.
 *
 * Nothing has run yet when this renders: the backend parked the call and is
 * waiting for POST /confirm/{id}.
 */
export function ConfirmationCard({
  confirmation,
  onResolved,
}: {
  confirmation: PendingConfirmation;
  onResolved?: (id: string, status: string) => void;
}) {
  const [status, setStatus] = useState<string>(confirmation.status ?? 'pending');
  const [busy, setBusy] = useState<'approve' | 'reject' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);

  const resolved = status === 'confirmed' || status === 'rejected';
  const args = confirmation.arguments && typeof confirmation.arguments === 'object'
    ? confirmation.arguments
    : {};
  const entries = Object.entries(args);

  async function decide(approve: boolean) {
    setBusy(approve ? 'approve' : 'reject');
    setError(null);
    try {
      const res = await api.confirm(confirmation.id, approve);
      const next = res?.status ?? (approve ? 'confirmed' : 'rejected');
      setStatus(next);
      setResult(res?.result ?? null);
      onResolved?.(confirmation.id, next);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.offline
            ? 'The backend is not reachable, so nothing was sent.'
            : err.message
          : String(err),
      );
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className={`confirm${resolved ? ' confirm--resolved' : ''}`}>
      <div className="confirm__head">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          {status === 'confirmed' ? (
            <>
              <circle cx="8" cy="8" r="6.6" stroke="currentColor" strokeWidth="1.4" />
              <path
                d="m5 8.2 2.1 2.1L11 6.3"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </>
          ) : status === 'rejected' ? (
            <>
              <circle cx="8" cy="8" r="6.6" stroke="currentColor" strokeWidth="1.4" />
              <path d="m5.6 5.6 4.8 4.8M10.4 5.6l-4.8 4.8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </>
          ) : (
            <>
              <path
                d="M8 1.8 1.9 13.2h12.2L8 1.8Z"
                stroke="currentColor"
                strokeWidth="1.4"
                strokeLinejoin="round"
              />
              <path d="M8 6.4v3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
              <circle cx="8" cy="11.1" r="0.8" fill="currentColor" />
            </>
          )}
        </svg>
        {resolved ? (
          <span>
            {status === 'confirmed' ? 'Approved' : 'Rejected'} — <code className="confirm__tool">{confirmation.tool}</code>
          </span>
        ) : (
          <span>
            Approval required — <code className="confirm__tool">{confirmation.tool}</code>
          </span>
        )}
        <span className="chip" style={{ marginLeft: 'auto' }}>
          {confirmation.agent || 'agent'}
        </span>
      </div>

      <div className="confirm__body">
        {entries.length ? (
          <dl className="confirm__args">
            {entries.map(([k, v]) => (
              <div key={k} style={{ display: 'contents' }}>
                <dt className="confirm__key">{k}</dt>
                <dd className="confirm__val" style={{ margin: 0 }}>
                  {renderValue(v)}
                </dd>
              </div>
            ))}
          </dl>
        ) : (
          <div className="muted">No arguments supplied.</div>
        )}

        {result ? (
          <div style={{ marginTop: 'var(--sp-3)' }}>
            <Collapsible label="Tool result">
              <JsonBlock value={result} />
            </Collapsible>
          </div>
        ) : null}

        {error ? (
          <div className="banner" role="alert" style={{ marginTop: 'var(--sp-3)' }}>
            <div className="banner__body">
              <div className="banner__title">Could not record your decision</div>
              <div className="banner__detail">{error}</div>
            </div>
          </div>
        ) : null}
      </div>

      {resolved ? null : (
        <div className="confirm__actions">
          <span className="confirm__note">
            This tool has side effects outside Jarvis. Nothing runs until you approve.
          </span>
          <button
            type="button"
            className="btn btn--danger btn--sm"
            onClick={() => void decide(false)}
            disabled={busy !== null}
          >
            {busy === 'reject' ? <Spinner label="Rejecting" /> : null}
            Reject
          </button>
          <button
            type="button"
            className="btn btn--primary btn--sm"
            onClick={() => void decide(true)}
            disabled={busy !== null}
          >
            {busy === 'approve' ? <Spinner label="Approving" /> : null}
            Approve
          </button>
        </div>
      )}
    </div>
  );
}
