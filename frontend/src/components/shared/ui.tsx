/** Small shared presentational primitives. Styles live in styles/base.css. */

import { useId, useState } from 'react';
import type { ReactNode } from 'react';
import { ApiError } from '../../api/client';
import { prettyJson } from '../../lib/format';

export function Spinner({ label }: { label?: string }) {
  return (
    <>
      <span className="spinner" aria-hidden="true" />
      <span className="visually-hidden">{label ?? 'Loading'}</span>
    </>
  );
}

export function ErrorBanner({
  error,
  onRetry,
  context,
}: {
  error: unknown;
  onRetry?: () => void;
  context?: string;
}) {
  if (!error) return null;
  const isApi = error instanceof ApiError;
  const offline = isApi && error.offline;
  const title = offline
    ? 'Backend offline'
    : context
      ? `Could not ${context}`
      : 'Something went wrong';
  const detail = offline
    ? 'Nothing is answering on the API. Start the backend (uvicorn on :8000) or run `npm run mock` for fixture data.'
    : error instanceof Error
      ? error.message
      : String(error);

  return (
    <div className="banner" role="alert">
      <span className="banner__icon" aria-hidden="true">
        <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
          <circle cx="8" cy="8" r="6.6" stroke="currentColor" strokeWidth="1.4" />
          <path d="M8 4.6v4.2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
          <circle cx="8" cy="11.3" r="0.85" fill="currentColor" />
        </svg>
      </span>
      <div className="banner__body">
        <div className="banner__title">{title}</div>
        <div className="banner__detail">{detail}</div>
      </div>
      {onRetry ? (
        <div className="banner__actions">
          <button type="button" className="btn btn--sm" onClick={onRetry}>
            Retry
          </button>
        </div>
      ) : null}
    </div>
  );
}

export function EmptyState({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children?: ReactNode;
}) {
  return (
    <div className="empty">
      <div className="empty__title">{title}</div>
      {hint ? <div>{hint}</div> : null}
      {children}
    </div>
  );
}

export function Caret({ open }: { open: boolean }) {
  return (
    <svg
      className={`disclosure__caret${open ? ' disclosure__caret--open' : ''}`}
      width="9"
      height="9"
      viewBox="0 0 10 10"
      aria-hidden="true"
    >
      <path d="M3.5 1.5 7 5l-3.5 3.5" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

/** Keyboard-accessible disclosure wrapping a scrollable JSON/pre block. */
export function Collapsible({
  label,
  defaultOpen = false,
  children,
}: {
  label: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  return (
    <div>
      <button
        type="button"
        className="disclosure"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((v) => !v)}
      >
        <Caret open={open} />
        {label}
      </button>
      <div id={id} hidden={!open} style={{ marginTop: open ? 6 : 0 }}>
        {open ? children : null}
      </div>
    </div>
  );
}

export function JsonBlock({ value }: { value: unknown }) {
  const text = typeof value === 'string' ? value : prettyJson(value);
  return <pre className="jsonblock">{text}</pre>;
}
