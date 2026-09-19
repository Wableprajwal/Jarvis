import { describe, expect, it, vi } from 'vitest';
import { ApiError, buildQuery, extractDetail, normalizeBase, request } from './client';

function jsonResponse(body: unknown, init: { status?: number; statusText?: string } = {}) {
  const { status = 200, statusText = 'OK' } = init;
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText,
    text: () => Promise.resolve(typeof body === 'string' ? body : JSON.stringify(body)),
  } as unknown as Response;
}

describe('normalizeBase', () => {
  it('strips trailing slashes so paths never double up', () => {
    expect(normalizeBase('/api/')).toBe('/api');
    expect(normalizeBase('http://x/y///')).toBe('http://x/y');
    expect(normalizeBase('/api')).toBe('/api');
  });
});

describe('buildQuery', () => {
  it('encodes present params and drops empty ones', () => {
    expect(buildQuery({ session_id: 's 1', limit: 20 })).toBe('?session_id=s+1&limit=20');
    expect(buildQuery({ session_id: undefined, limit: null, other: '' })).toBe('');
  });
});

describe('extractDetail', () => {
  it('prefers FastAPI {detail}', () => {
    expect(extractDetail({ detail: 'nope' }, 'fallback')).toBe('nope');
  });

  it('joins a validation-error detail list', () => {
    expect(
      extractDetail({ detail: [{ msg: 'field required' }, { msg: 'too long' }] }, 'fallback'),
    ).toBe('field required; too long');
  });

  it('falls back for empty, missing or unusable bodies', () => {
    expect(extractDetail(null, 'fallback')).toBe('fallback');
    expect(extractDetail({}, 'fallback')).toBe('fallback');
    expect(extractDetail({ detail: '   ' }, 'fallback')).toBe('fallback');
    expect(extractDetail('plain text error', 'fallback')).toBe('plain text error');
  });
});

describe('request', () => {
  it('parses a successful JSON response', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({ status: 'ok' }));
    await expect(request<{ status: string }>('/health', { fetchImpl })).resolves.toEqual({
      status: 'ok',
    });
    expect(fetchImpl).toHaveBeenCalledOnce();
    expect(String(fetchImpl.mock.calls[0]?.[0])).toContain('/health');
  });

  it('raises an offline ApiError when the network rejects', async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));
    const err = await request('/health', { fetchImpl }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).offline).toBe(true);
    expect((err as ApiError).status).toBe(0);
    expect((err as ApiError).message).toContain('Failed to fetch');
  });

  it('raises an ApiError carrying the backend detail on a non-2xx', async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValue(jsonResponse({ detail: 'session not found' }, { status: 404 }));
    const err = await request('/sessions/x/messages', { fetchImpl }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(404);
    expect((err as ApiError).offline).toBe(false);
    expect((err as ApiError).message).toBe('session not found');
  });

  it('falls back to status text when the error body is not JSON', async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValue(jsonResponse('<html>gateway</html>', { status: 502, statusText: 'Bad Gateway' }));
    const err = await request('/chat', { fetchImpl }).catch((e: unknown) => e);
    expect((err as ApiError).message).toContain('gateway');
    expect((err as ApiError).status).toBe(502);
  });

  it('sends a JSON body and content-type only when there is a body', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({ ok: true }));
    await request('/chat', { method: 'POST', body: { message: 'hi' }, fetchImpl });
    const init = fetchImpl.mock.calls[0]?.[1] as RequestInit;
    expect(init.method).toBe('POST');
    expect(init.body).toBe('{"message":"hi"}');
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('application/json');

    const getFetch = vi.fn().mockResolvedValue(jsonResponse({ ok: true }));
    await request('/health', { fetchImpl: getFetch });
    const getInit = getFetch.mock.calls[0]?.[1] as RequestInit;
    expect(getInit.body).toBeUndefined();
    expect((getInit.headers as Record<string, string>)['Content-Type']).toBeUndefined();
  });

  it('returns null for an empty 200 body rather than throwing', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(''));
    await expect(request('/confirmations', { fetchImpl })).resolves.toBeNull();
  });
});
