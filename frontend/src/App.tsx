import { useCallback } from 'react';
import './App.css';
import { api } from './api/client';
import { useAsync } from './hooks/useAsync';
import { useHashRoute, VIEWS } from './hooks/useHashRoute';
import type { ViewName } from './hooks/useHashRoute';
import { Chat } from './components/Chat/Chat';
import { TraceView } from './components/TraceView/TraceView';
import { EvalDashboard } from './components/EvalDashboard/EvalDashboard';

/** Provider/model badge in the top nav; doubles as the backend-reachability light. */
function HealthPill() {
  const { data, error, loading } = useAsync(() => api.health(), []);

  if (loading && !data && !error) {
    return (
      <span className="statuspill statuspill--loading">
        <span className="statuspill__dot" />
        <span>checking backend…</span>
      </span>
    );
  }
  if (error || !data) {
    return (
      <span className="statuspill statuspill--offline" title="No response from the Jarvis API">
        <span className="statuspill__dot" />
        <span>backend offline</span>
      </span>
    );
  }
  return (
    <span className="statuspill" title={`provider ${data.provider} · model ${data.model}`}>
      <span className="statuspill__dot" />
      <span>{data.provider || 'provider'}</span>
      <span className="statuspill__sep" aria-hidden="true">
        /
      </span>
      <span className="statuspill__model">{data.model || 'unknown model'}</span>
    </span>
  );
}

export default function App() {
  const [route, navigate] = useHashRoute();

  const openTrace = useCallback(
    (turnId: string) => {
      navigate('traces', turnId);
    },
    [navigate],
  );

  return (
    <div className="app">
      <header className="nav">
        <div className="nav__brand">
          <span className="nav__mark" aria-hidden="true">
            J
          </span>
          <span className="nav__name">Jarvis</span>
        </div>

        <nav className="nav__tabs" aria-label="Views">
          {VIEWS.map((v) => (
            <a
              key={v.id}
              className="nav__tab"
              href={v.path}
              aria-current={route.view === v.id ? 'page' : undefined}
            >
              {v.label}
            </a>
          ))}
        </nav>

        <div className="nav__right">
          <HealthPill />
        </div>
      </header>

      <main className="app__main">
        {route.view === 'chat' ? <Chat onOpenTrace={openTrace} /> : null}
        {route.view === 'traces' ? (
          <TraceView
            selectedTurnId={route.param}
            onSelectTurn={(id: string) => navigate('traces', id)}
          />
        ) : null}
        {route.view === 'evals' ? (
          <EvalDashboard
            selectedRunId={route.param}
            onSelectRun={(id: string | null) => navigate('evals', id)}
          />
        ) : null}
      </main>
    </div>
  );
}

export type { ViewName };
