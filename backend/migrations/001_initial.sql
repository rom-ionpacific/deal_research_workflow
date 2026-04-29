-- 001_initial.sql -- create the research schema and the three core tables.
-- Idempotent: safe to re-run.

-- pg_trgm lives in the dealcloud schema in this DB (deal_cloud_enhancer
-- created it there). search_path must include dealcloud so gin_trgm_ops
-- and similarity() resolve unqualified for the rest of the migration.
SET search_path TO research, dealcloud, public;

CREATE EXTENSION IF NOT EXISTS pgcrypto;        -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pg_trgm;          -- trigram similarity for orgs/search

CREATE SCHEMA IF NOT EXISTS research;

SET search_path TO research, dealcloud, public;

-- ============================================================
-- session
-- ============================================================
CREATE TABLE IF NOT EXISTS session (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    originator_email         TEXT NOT NULL,
    title                    TEXT,
    current_version_id       UUID,                  -- FK added below (circular)
    redo_version_id          UUID,
    forked_from_version_id   UUID,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_session_user
    ON session(originator_email, updated_at DESC);

-- ============================================================
-- session_version  -- append-only state log (DAG via parent_id)
-- ============================================================
CREATE TABLE IF NOT EXISTS session_version (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    parent_id       UUID REFERENCES session_version(id),    -- NULL at root
    undo_unit_id    UUID NOT NULL,                          -- groups versions per user turn
    phase           TEXT NOT NULL CHECK (phase IN
        ('org_select','entity_select','data_room_setup','data_room_view')),
    state           JSONB NOT NULL,
    source          TEXT NOT NULL CHECK (source IN
        ('user_action','ai_tool_call','external_link','session_fork','phase_transition')),
    ai_message_id   UUID,                                    -- FK added below (circular)
    summary         TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_version_session
    ON session_version(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_version_parent
    ON session_version(parent_id);
CREATE INDEX IF NOT EXISTS idx_version_undo
    ON session_version(session_id, undo_unit_id);

-- ============================================================
-- session_chat_message  -- dialogue log (links to versions for state-at-time)
-- ============================================================
CREATE TABLE IF NOT EXISTS session_chat_message (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          UUID NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    phase               TEXT NOT NULL,
    role                TEXT NOT NULL CHECK (role IN ('user','assistant','tool')),
    content             JSONB NOT NULL,
    pre_version_id      UUID REFERENCES session_version(id),
    post_version_id     UUID REFERENCES session_version(id),
    parent_message_id   UUID REFERENCES session_chat_message(id),
    model_id            TEXT,
    tokens_in           INTEGER,
    tokens_out          INTEGER,
    latency_ms          INTEGER,
    error               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_msg_session
    ON session_chat_message(session_id, created_at);

-- ============================================================
-- Late-binding FKs (resolve circular references)
-- ============================================================
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_session_current_version') THEN
        ALTER TABLE session
            ADD CONSTRAINT fk_session_current_version
            FOREIGN KEY (current_version_id) REFERENCES session_version(id);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_session_redo_version') THEN
        ALTER TABLE session
            ADD CONSTRAINT fk_session_redo_version
            FOREIGN KEY (redo_version_id) REFERENCES session_version(id);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_session_forked_from') THEN
        ALTER TABLE session
            ADD CONSTRAINT fk_session_forked_from
            FOREIGN KEY (forked_from_version_id) REFERENCES session_version(id);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_version_ai_message') THEN
        ALTER TABLE session_version
            ADD CONSTRAINT fk_version_ai_message
            FOREIGN KEY (ai_message_id) REFERENCES session_chat_message(id);
    END IF;
END$$;

-- Trigram indexes on dealcloud.organization for fast orgs/search.
-- Created in dealcloud schema; safe to re-run.
CREATE INDEX IF NOT EXISTS idx_org_name_trgm
    ON dealcloud.organization USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_org_alias_trgm
    ON dealcloud.organization_alias USING gin (alias gin_trgm_ops);
