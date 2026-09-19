/** Small display helpers. Pure, so they are safe to unit-test and reuse. */

export function formatMs(ms: number | null | undefined): string {
  if (typeof ms !== 'number' || !Number.isFinite(ms)) return '—';
  if (ms < 1) return '<1 ms';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
  const mins = Math.floor(ms / 60_000);
  const secs = Math.round((ms % 60_000) / 1000);
  return `${mins}m ${secs}s`;
}

export function formatTokens(n: number | null | undefined): string {
  if (typeof n !== 'number' || !Number.isFinite(n)) return '—';
  if (Math.abs(n) < 1000) return String(Math.round(n));
  return `${(n / 1000).toFixed(1)}k`;
}

export function formatPercent(rate: number | null | undefined, digits = 0): string {
  if (typeof rate !== 'number' || !Number.isFinite(rate)) return '—';
  return `${(rate * 100).toFixed(digits)}%`;
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return String(iso);
  return new Date(t).toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return String(iso);
  return new Date(t).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function relativeTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return String(iso);
  const diff = Date.now() - t;
  const abs = Math.abs(diff);
  const mins = Math.round(abs / 60_000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export function shortId(id: string | null | undefined, keep = 8): string {
  if (!id) return '—';
  return id.length <= keep ? id : `${id.slice(0, keep)}…`;
}

export function prettyJson(value: unknown): string {
  if (value === undefined) return '';
  try {
    return JSON.stringify(value, null, 2) ?? String(value);
  } catch {
    return String(value);
  }
}

/** Compact one-line rendering of tool arguments for a collapsed row. */
export function summarizeArgs(args: Record<string, unknown> | undefined | null): string {
  if (!args || typeof args !== 'object') return '';
  const parts: string[] = [];
  for (const [k, v] of Object.entries(args)) {
    let rendered: string;
    if (typeof v === 'string') rendered = v.length > 48 ? `"${v.slice(0, 48)}…"` : `"${v}"`;
    else if (v === null) rendered = 'null';
    else if (typeof v === 'object') rendered = Array.isArray(v) ? `[${v.length}]` : '{…}';
    else rendered = String(v);
    parts.push(`${k}: ${rendered}`);
    if (parts.join(', ').length > 120) break;
  }
  return parts.join(', ');
}

/** Human label for an agent name like "research_agent" → "Research". */
export function agentLabel(agent: string | null | undefined): string {
  if (!agent) return 'unknown';
  return agent
    .replace(/_?agent$/i, '')
    .replace(/[_-]+/g, ' ')
    .trim()
    .replace(/^\w/, (c) => c.toUpperCase()) || agent;
}
