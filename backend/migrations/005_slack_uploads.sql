-- =====================================================================
-- 005: Slack file uploads Todd can read
-- =====================================================================
-- When someone drops a file into a DM with Todd (or into a thread where
-- he's mentioned), we download it, extract its text via the dce
-- /internal/extract-text endpoint, and store the result here.
--
-- Why a table and not just the conversation history:
--
--   1. A DM is ONE eternal slack_conversation row, and message_history
--      is trimmed to HISTORY_CAP entries. A file uploaded three weeks
--      ago must still be readable today, long after the turn that
--      introduced it fell out of the window.
--   2. Extracted bodies run to tens of thousands of characters. Putting
--      that in message_history means re-sending it to Anthropic on
--      EVERY subsequent turn, forever. Instead the turn carries a short
--      excerpt and the model pulls the rest through read_uploaded_file
--      when it actually needs it -- the same split as
--      read_document_summary vs read_document.
--   3. Extraction is the expensive part (download + parse + possible
--      OCR). Keying on Slack's own file id makes a re-read free.

CREATE TABLE IF NOT EXISTS research.slack_uploaded_file (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Slack's file id (e.g. 'F07ABCDEF'). The natural key: stable,
    -- unique workspace-wide, and what the model gets handed to read a
    -- file back.
    slack_file_id  TEXT NOT NULL UNIQUE,
    team_id        TEXT NOT NULL,
    channel_id     TEXT NOT NULL,
    thread_ts      TEXT,                     -- NULL for DMs, as elsewhere
    slack_user_id  TEXT NOT NULL,            -- who uploaded it
    user_email     TEXT NOT NULL,
    name           TEXT NOT NULL,
    mimetype       TEXT,
    filetype       TEXT,                     -- Slack's short label ('pdf')
    size_bytes     BIGINT,
    permalink      TEXT,
    -- Extraction outcome. ok=false rows are kept deliberately: the
    -- reason a file could not be read ('too_large', 'unsupported_mime',
    -- 'no_text_extracted') is something Todd should be able to tell the
    -- user on a later turn, and caching it stops us re-downloading a
    -- 40 MB video every time someone asks about it.
    ok             BOOLEAN NOT NULL DEFAULT FALSE,
    body           TEXT,
    total_chars    INTEGER NOT NULL DEFAULT 0,
    error          TEXT,
    extracted_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- "What has been uploaded in this conversation?" -- the lookup behind
-- resolving a file by NAME when the model doesn't have the id to hand.
CREATE INDEX IF NOT EXISTS idx_slack_upload_convo
    ON research.slack_uploaded_file(team_id, channel_id,
                                    COALESCE(thread_ts, ''), created_at DESC);

CREATE INDEX IF NOT EXISTS idx_slack_upload_user
    ON research.slack_uploaded_file(user_email, created_at DESC);
