/**
 * Minimal data-fetching hook: run an async fn, keep {data, error, loading},
 * ignore results from superseded calls, and expose a manual `reload`.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export interface AsyncState<T> {
  data: T | null;
  error: unknown;
  loading: boolean;
  reload: () => void;
  setData: (updater: T | ((prev: T | null) => T | null)) => void;
}

export function useAsync<T>(
  fn: () => Promise<T>,
  deps: readonly unknown[],
  options: { enabled?: boolean } = {},
): AsyncState<T> {
  const { enabled = true } = options;
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(enabled);
  const [nonce, setNonce] = useState(0);
  const runId = useRef(0);

  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }
    const id = ++runId.current;
    setLoading(true);
    fnRef
      .current()
      .then((result) => {
        if (runId.current !== id) return;
        setData(result);
        setError(null);
      })
      .catch((err: unknown) => {
        if (runId.current !== id) return;
        setError(err);
      })
      .finally(() => {
        if (runId.current !== id) return;
        setLoading(false);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce, enabled]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);

  return { data, error, loading, reload, setData };
}
