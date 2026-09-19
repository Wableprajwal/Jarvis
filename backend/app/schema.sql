-- Idempotent schema. Applied at startup by app.db.Database.init_schema().

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL DEFAULT 'default',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    summary     TEXT
);
-- Phase 6: a closed session is ready for the long-term fact extraction pass.
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS closed_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    message     JSONB NOT NULL,          -- the normalized message dict, verbatim
    tokens      INTEGER NOT NULL DEFAULT 0,
    summarized  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_messages_session ON messages (session_id, id);

CREATE TABLE IF NOT EXISTS facts (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    source      TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, key)
);

CREATE TABLE IF NOT EXISTS trace_events (
    seq         BIGSERIAL PRIMARY KEY,   -- global, gap-tolerant total order
    id          TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    agent_name  TEXT NOT NULL,
    turn_id     TEXT,
    event_type  TEXT NOT NULL,
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
    latency_ms  DOUBLE PRECISION,
    tokens      INTEGER,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_trace_session ON trace_events (session_id, seq);
CREATE INDEX IF NOT EXISTS ix_trace_turn ON trace_events (turn_id, seq);
CREATE INDEX IF NOT EXISTS ix_trace_type ON trace_events (event_type, ts);

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id            TEXT PRIMARY KEY,
    config_hash       TEXT NOT NULL,
    ts                TIMESTAMPTZ NOT NULL DEFAULT now(),
    scores_by_category JSONB NOT NULL,
    details           JSONB NOT NULL DEFAULT '{}'::jsonb
);
