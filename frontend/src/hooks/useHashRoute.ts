/**
 * Hash-based routing — no router dependency, and every view is linkable.
 *
 *   #/chat
 *   #/traces          #/traces/<turn_id>
 *   #/evals           #/evals/<run_id>
 */

import { useCallback, useEffect, useState } from 'react';

export type ViewName = 'chat' | 'traces' | 'evals';

export interface Route {
  view: ViewName;
  param: string | null;
}

export const VIEWS: { id: ViewName; label: string; path: string }[] = [
  { id: 'chat', label: 'Chat', path: '#/chat' },
  { id: 'traces', label: 'Traces', path: '#/traces' },
  { id: 'evals', label: 'Evals', path: '#/evals' },
];

export function parseHash(hash: string): Route {
  const clean = hash.replace(/^#\/?/, '').split('?')[0] ?? '';
  const [head = '', ...rest] = clean.split('/').filter(Boolean);
  const param = rest.length ? decodeURIComponent(rest.join('/')) : null;
  switch (head) {
    case 'traces':
      return { view: 'traces', param };
    case 'evals':
      return { view: 'evals', param };
    case 'chat':
    default:
      return { view: 'chat', param };
  }
}

export function routeToHash(view: ViewName, param?: string | null): string {
  return param ? `#/${view}/${encodeURIComponent(param)}` : `#/${view}`;
}

export function useHashRoute(): [Route, (view: ViewName, param?: string | null) => void] {
  const [route, setRoute] = useState<Route>(() =>
    parseHash(typeof window === 'undefined' ? '' : window.location.hash),
  );

  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener('hashchange', onChange);
    if (!window.location.hash) window.location.replace('#/chat');
    return () => window.removeEventListener('hashchange', onChange);
  }, []);

  const navigate = useCallback((view: ViewName, param?: string | null) => {
    window.location.hash = routeToHash(view, param ?? null);
  }, []);

  return [route, navigate];
}
