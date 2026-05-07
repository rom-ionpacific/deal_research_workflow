-- 004_session_starring.sql -- starred sessions + auto-rename lock.
-- Idempotent: safe to re-run.

SET search_path TO research, dealcloud, public;

ALTER TABLE research.session
    ADD COLUMN IF NOT EXISTS is_starred BOOLEAN NOT NULL DEFAULT FALSE,
    -- TRUE means the title should not be auto-renamed (set on manual
    -- edit AND after the first-org-selection auto-rename, so the title
    -- changes once and then sticks).
    ADD COLUMN IF NOT EXISTS title_is_locked BOOLEAN NOT NULL DEFAULT FALSE;

-- Sort key for the sessions list: starred first, then by recency.
CREATE INDEX IF NOT EXISTS idx_session_starred_updated
    ON research.session(originator_email, is_starred DESC, updated_at DESC);
