/**
 * Typed HTTP client for the Jarvis backend.
 *
 * Base URL comes from VITE_API_BASE and defaults to "/api", which the Vite dev
 * server proxies to http://localhost:8000 with the prefix stripped — so the app
 * never issues a cross-origin request and no CORS config is needed.
 *
 * Every call funnels through `request()`, which normalises failures into a
 * single `ApiError` carrying the backend's `{detail}` when it sent one.
 */

import type {
  ChatRequest,
  ChatResponse,
  ConfirmResponse,
  Health,
  Metrics,
  PendingConfirmation,
  Scorecard,
  SessionSummary,
  StoredMessage,
  TurnSummary,
  TurnTrace,
} from './types';

export class ApiError extends Error {
  readonly status: number;
  /** True when the request never reached the server (backend down, DNS, abort). */
  readonly offline: boolean;

  constructor(message: string, status: number, offline = false) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.offline = offline;
  }
}

const rawBase: string =
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE) || '/api';

/** Strip a trailing slash so `${base}${path}` never doubles up. */
export function normalizeBase(base: string): string {
  return base.replace(/\/+$/, '');
}

export const API_BASE = normalizeBase(rawBase);

/** Pull a human-usable message out of whatever the server returned. */
export function extractDetail(body: unknown, fallback: string): string {
  if (typeof body === 'string' && body.trim()) return body.trim();
  if (body && typeof body === 'object') {
    const detail = (body as { detail?: unknown }).detail;
    if (typeof detail === 'string' && detail.trim()) return detail.trim();
    // FastAPI validation errors arrive as a list of {loc, msg, type}
    if (Array.isArray(detail)) {
      const msgs = detail
        .map((d) => (d && typeof d === 'object' ? (d as { msg?: unknown }).msg : null))
        .filter((m): m is string => typeof m === 'string');
      if (msgs.length) return msgs.join('; ');
    }
  }
  return fallback;
}

export function buildQuery(params: Record<string, string | number | undefined | null>): string {
  const usable = Object.entries(params).filter(
    ([, v]) => v !== undefined && v !== null && v !== '',
  );
  if (!usable.length) return '';
  const sp = new URLSearchParams();
  for (const [k, v] of usable) sp.set(k, String(v));
  return `?${sp.toString()}`;
}

export interface RequestOptions {
  method?: string;
  body?: unknown;
  signal?: AbortSignal;
  /** Injected in tests. */
  fetchImpl?: typeof fetch;
}

export async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, signal, fetchImpl } = opts;
  const doFetch = fetchImpl ?? globalThis.fetch;
  const url = `${API_BASE}${path}`;

  let res: Response;
  try {
    res = await doFetch(url, {
      method,
      headers: body === undefined ? { Accept: 'application/json' } : {
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      ...(signal ? { signal } : {}),
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new ApiError(
      `Could not reach the Jarvis backend (${method} ${url}): ${msg}`,
      0,
      true,
    );
  }

  const text = await res.text();
  let parsed: unknown = null;
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }
  }

  if (!res.ok) {
    throw new ApiError(
      extractDetail(parsed, `${res.status} ${res.statusText || 'Request failed'}`),
      res.status,
    );
  }

  return parsed as T;
}

/** Endpoints that return lists: tolerate a null/omitted body from a young backend. */
async function requestList<T>(path: string, opts: RequestOptions = {}): Promise<T[]> {
  const data = await request<T[] | null>(path, opts);
  return Array.isArray(data) ? data : [];
}

export const api = {
  health: (o?: RequestOptions) => request<Health>('/health', o),

  chat: (body: ChatRequest, o?: RequestOptions) =>
    request<ChatResponse>('/chat', { ...o, method: 'POST', body }),

  sessions: (o?: RequestOptions) => requestList<SessionSummary>('/sessions', o),

  sessionMessages: (sessionId: string, o?: RequestOptions) =>
    requestList<StoredMessage>(`/sessions/${encodeURIComponent(sessionId)}/messages`, o),

  turns: (params: { session_id?: string; limit?: number } = {}, o?: RequestOptions) =>
    requestList<TurnSummary>(`/traces/turns${buildQuery(params)}`, o),

  turnTrace: (turnId: string, o?: RequestOptions) =>
    request<TurnTrace>(`/traces/turn/${encodeURIComponent(turnId)}`, o),

  metrics: (o?: RequestOptions) => request<Metrics>('/metrics', o),

  confirmations: (sessionId?: string, o?: RequestOptions) =>
    requestList<PendingConfirmation>(
      `/confirmations${buildQuery({ session_id: sessionId })}`,
      o,
    ),

  confirm: (id: string, approve: boolean, o?: RequestOptions) =>
    request<ConfirmResponse>(`/confirm/${encodeURIComponent(id)}`, {
      ...o,
      method: 'POST',
      body: { approve },
    }),

  runEval: (o?: RequestOptions) => request<Scorecard>('/eval/run', { ...o, method: 'POST' }),

  evalRuns: (o?: RequestOptions) => requestList<Scorecard>('/eval/runs', o),

  evalRun: (runId: string, o?: RequestOptions) =>
    request<Scorecard>(`/eval/runs/${encodeURIComponent(runId)}`, o),
};

export type Api = typeof api;
