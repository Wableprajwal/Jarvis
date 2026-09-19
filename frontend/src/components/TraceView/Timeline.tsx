import type {
  Citation,
  ErrorPayload,
  FinalResponsePayload,
  MemorySummarizePayload,
  ModelCallPayload,
  RouteDecisionPayload,
  TraceEvent,
} from '../../api/types';
import { buildLanes, buildSteps, describeStep, payloadOf } from '../../lib/trace';
import type { TimelineStep, ToolStep } from '../../lib/trace';
import { agentLabel, formatMs, formatTime, formatTokens, summarizeArgs } from '../../lib/format';
import { Collapsible, JsonBlock } from '../shared/ui';
import { agentColorVar } from '../Chat/agentColor';

function Nums({ latency, tokens }: { latency: number | null; tokens: number | null }) {
  if (latency === null && tokens === null) return null;
  return (
    <span className="step__nums">
      {latency !== null ? <span>{formatMs(latency)}</span> : null}
      {tokens !== null ? <span>{formatTokens(tokens)} tok</span> : null}
    </span>
  );
}

function Citations({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return (
    <div className="cites">
      <div className="cites__title">
        Retrieved context · {citations.length} passage{citations.length === 1 ? '' : 's'}
      </div>
      {citations.map((c, i) => {
        const score = typeof c.score === 'number' && Number.isFinite(c.score)
          ? Math.min(1, Math.max(0, c.score))
          : null;
        return (
          <blockquote className="cite" key={`${c.chunk_id || c.source}-${i}`}>
            <div className="cite__head">
              <span className="cite__source">{c.source}</span>
              {c.chunk_id ? <span className="cite__chunk">{c.chunk_id}</span> : null}
              {score !== null ? (
                <span className="cite__score">
                  <span className="cite__scorebar" aria-hidden="true">
                    <span className="cite__scorefill" style={{ width: `${score * 100}%` }} />
                  </span>
                  <span className="cite__scorenum">{score.toFixed(2)}</span>
                </span>
              ) : null}
            </div>
            <div className="cite__text">{c.text}</div>
          </blockquote>
        );
      })}
    </div>
  );
}

function ToolStepCard({ step }: { step: ToolStep }) {
  const failed = step.ok === false && !step.pendingConfirmation;
  const pending = step.pendingConfirmation;
  const args = summarizeArgs(step.args);
  const cls = `step${failed ? ' step--failed' : ''}${pending ? ' step--pending' : ''}`;

  return (
    <div className={cls}>
      <div className="step__card">
        <div className="step__head">
          <span className="step__kind">tool</span>
          <span className="step__title step__title--mono">{step.tool}</span>
          {pending ? (
            <span className="badge badge--warn">awaiting approval</span>
          ) : step.ok === true ? (
            <span className="badge badge--ok">ok</span>
          ) : step.ok === false ? (
            <span className="badge badge--fail">failed</span>
          ) : (
            <span className="badge">no result</span>
          )}
          <Nums latency={step.latencyMs} tokens={step.tokens} />
        </div>

        <div className="step__body">
          {args ? <div className="step__args">{args}</div> : null}

          {step.error ? (
            <div className="step__text" style={{ color: 'var(--critical)' }}>
              {step.error}
            </div>
          ) : null}

          <Citations citations={step.citations} />

          {step.output !== undefined && step.citations.length === 0 ? (
            <Collapsible label="Output">
              <JsonBlock value={step.output} />
            </Collapsible>
          ) : null}

          {Object.keys(step.args).length ? (
            <Collapsible label="Arguments (raw)">
              <JsonBlock value={step.args} />
            </Collapsible>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function EventStepCard({ event }: { event: TraceEvent }) {
  const type = event.event_type;
  const title = describeStep({ kind: 'event', key: '', agent: event.agent_name, event, seq: event.seq, ts: event.ts });

  if (type === 'route_decision') {
    const p = payloadOf<RouteDecisionPayload>(event);
    return (
      <div className="step step--route">
        <div className="step__card">
          <div className="step__head">
            <span className="step__kind">route</span>
            <span className="step__title">→ {p.agent ?? 'unknown agent'}</span>
            <Nums latency={event.latency_ms} tokens={event.tokens} />
          </div>
          <div className="step__body">
            <div className="route__why">
              {p.task ? (
                <>
                  <span className="route__k">Task</span>
                  <span className="route__v">{p.task}</span>
                </>
              ) : null}
              <span className="route__k">Why</span>
              <span className="route__v">
                {p.reason || 'No reason recorded for this routing decision.'}
              </span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (type === 'error') {
    const p = payloadOf<ErrorPayload>(event);
    return (
      <div className="step step--error">
        <div className="step__card">
          <div className="step__head">
            <span className="step__kind">error</span>
            <span className="step__title">{p.stage ? `stage: ${p.stage}` : 'error'}</span>
            <Nums latency={event.latency_ms} tokens={event.tokens} />
          </div>
          <div className="step__body">
            <div className="step__text">{p.error || 'No error detail recorded.'}</div>
          </div>
        </div>
      </div>
    );
  }

  if (type === 'model_call') {
    const p = payloadOf<ModelCallPayload>(event);
    const toolCalls = Array.isArray(p.tool_calls) ? p.tool_calls : [];
    return (
      <div className="step">
        <div className="step__card">
          <div className="step__head">
            <span className="step__kind">model</span>
            <span className="step__title">{title}</span>
            {p.stop_reason ? <span className="badge">{p.stop_reason}</span> : null}
            {p.parse_error ? <span className="badge badge--fail">parse error</span> : null}
            <Nums
              latency={event.latency_ms}
              tokens={
                event.tokens ??
                ((p.prompt_tokens ?? 0) + (p.completion_tokens ?? 0) || null)
              }
            />
          </div>
          <div className="step__body">
            {p.text ? <div className="step__text">{p.text}</div> : null}
            {p.parse_error ? (
              <div className="step__text" style={{ color: 'var(--critical)' }}>
                {p.parse_error}
              </div>
            ) : null}
            {toolCalls.length ? (
              <div className="step__head" style={{ padding: 0, gap: 'var(--sp-2)' }}>
                <span className="step__kind">requested</span>
                {toolCalls.map((t, i) => (
                  <span className="badge badge--accent" key={`${t}-${i}`}>
                    {t}
                  </span>
                ))}
              </div>
            ) : null}
            {typeof p.prompt_tokens === 'number' || typeof p.completion_tokens === 'number' ? (
              <div className="metricrow__nums">
                <span>prompt {formatTokens(p.prompt_tokens ?? null)}</span>
                <span>completion {formatTokens(p.completion_tokens ?? null)}</span>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    );
  }

  if (type === 'final_response') {
    const p = payloadOf<FinalResponsePayload>(event);
    return (
      <div className="step">
        <div className="step__card">
          <div className="step__head">
            <span className="step__kind">final</span>
            <span className="step__title">{title}</span>
            <Nums latency={event.latency_ms} tokens={event.tokens} />
          </div>
          {p.text ? (
            <div className="step__body">
              <div className="step__text">{p.text}</div>
            </div>
          ) : null}
        </div>
      </div>
    );
  }

  if (type === 'memory_summarize') {
    const p = payloadOf<MemorySummarizePayload>(event);
    const before = p.tokens_before ?? null;
    const after = p.tokens_after ?? null;
    return (
      <div className="step">
        <div className="step__card">
          <div className="step__head">
            <span className="step__kind">memory</span>
            <span className="step__title">
              summarized {formatTokens(before)} → {formatTokens(after)} tokens
            </span>
            <Nums latency={event.latency_ms} tokens={null} />
          </div>
        </div>
      </div>
    );
  }

  // Unknown event type — render it rather than swallowing it.
  return (
    <div className="step">
      <div className="step__card">
        <div className="step__head">
          <span className="step__kind">{String(type)}</span>
          <span className="step__title">{title}</span>
          <Nums latency={event.latency_ms} tokens={event.tokens} />
        </div>
        <div className="step__body">
          <Collapsible label="Payload">
            <JsonBlock value={event.payload} />
          </Collapsible>
        </div>
      </div>
    </div>
  );
}

function StepCard({ step }: { step: TimelineStep }) {
  return step.kind === 'tool' ? <ToolStepCard step={step} /> : <EventStepCard event={step.event} />;
}

/** Indented, per-agent vertical timeline of one turn. */
export function Timeline({ events }: { events: TraceEvent[] }) {
  const steps = buildSteps(events);
  const lanes = buildLanes(steps);

  return (
    <div className="timeline">
      {lanes.map((lane, i) => (
        <section
          className="lane"
          key={`${lane.agent}-${i}`}
          style={{
            ['--depth' as string]: lane.depth,
            ['--agent' as string]: agentColorVar(lane.agent),
          }}
          aria-label={`${agentLabel(lane.agent)} steps`}
        >
          <header className="lane__head">
            <span className="agentpip" style={{ ['--agent' as string]: agentColorVar(lane.agent) }}>
              <span className="agentpip__dot" aria-hidden="true" />
              {lane.agent}
            </span>
            <span className="lane__rule" aria-hidden="true" />
            <span className="muted" style={{ fontSize: 'var(--fs-xs)' }}>
              {formatTime(lane.steps[0]?.ts ?? null)}
            </span>
          </header>
          <div className="lane__body">
            {lane.steps.map((s) => (
              <StepCard step={s} key={s.key} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
