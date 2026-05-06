-- 002_todd.sql -- Todd the Walrus support: 5 dossier functions + Slack
--                 conversation state. Idempotent.
--
-- The 5 functions live in `dealcloud` schema (read-only over existing
-- DC data) and accept `org_ids INTEGER[]` -- the bundle of organization
-- IDs that resolve to the same canonical company via supersede/cluster
-- walking. Each returns a single JSONB object shaped per
-- todd_walrus.md.
--
-- Slack conversation state lives in `research` schema next to
-- `session` since Todd-spawned sessions can graduate into full research
-- workflows.

SET search_path TO research, dealcloud, public;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =====================================================================
-- Helper: alias IDs for a bundle of orgs
-- =====================================================================
-- Most question functions need to walk through `organization_alias` to
-- reach entity-junction tables (email_thread_organization etc.).
-- This helper centralises that lookup. STABLE so callers can compose it
-- in WHERE/JOIN without re-evaluation per row.
CREATE OR REPLACE FUNCTION dealcloud.todd_alias_ids(org_ids INTEGER[])
RETURNS TABLE(alias_id INTEGER) LANGUAGE sql STABLE AS $$
    SELECT id
      FROM dealcloud.organization_alias
     WHERE organization_id = ANY(org_ids);
$$;

-- =====================================================================
-- Q1: org_portfolio_status
-- =====================================================================
-- "in_portfolio" => any deal with status in (Portfolio Company,
-- Partnership) where the org is the counterparty OR an underlying.
-- (Strict definition: confirmed positions only. Active Pipeline and
-- Warming Station are tracked under Q2's "assessed" lens.)
-- Underlying rows include both DealCloud-sourced and LLM-derived
-- (from IC-memo doc tags via the deal_cloud_enhancer enrichment).
-- connection_source values: 'dealcloud', 'llm_derived',
-- 'dealcloud+llm_derived'. derived_n_docs is the supporting evidence
-- count for derived rows (NULL for dealcloud-only).
-- doc_only_underlying_hints: documents that mention BOTH the bundle AND
-- another DC-counterparty, EXCLUDING firm-level overview decks. A doc
-- with >10 distinct org mentions is treated as a portfolio summary,
-- not a deal-specific document, and skipped.
CREATE OR REPLACE FUNCTION dealcloud.org_portfolio_status(org_ids INTEGER[])
RETURNS jsonb LANGUAGE sql STABLE AS $$
WITH
counterparty_all AS (
    SELECT d.id AS deal_id, d.name, d.transaction_type, d.status,
           d.new_deal_date AS date
      FROM dealcloud.deal d
     WHERE d.organization_id = ANY(org_ids)
       AND d.status IN ('Portfolio Company', 'Partnership')
),
counterparty_top AS (
    SELECT * FROM counterparty_all
     ORDER BY date DESC NULLS LAST, deal_id DESC
     LIMIT 5
),
underlying_all AS (
    -- Include both is_underlying and is_value_driver. DealCloud's two
    -- source fields (Organization.RelatedDeal -> is_underlying,
    -- Deal.Companiesexposure -> is_value_driver) aren't perfectly
    -- consistent: ~100 DUC rows have only is_value_driver=TRUE. The
    -- design intent is that every value driver is also an underlying
    -- company. So treat any DUC row as a portfolio-relationship signal.
    SELECT d.id AS deal_id, d.name AS deal_name,
           d.organization_id AS parent_org_id,
           o_parent.name      AS parent_org_name,
           d.status, d.new_deal_date AS date,
           duc.connection_source,
           duc.derived_n_docs,
           duc.is_value_driver
      FROM dealcloud.deal_underlying_company duc
      JOIN dealcloud.deal d ON d.id = duc.deal_id
      LEFT JOIN dealcloud.organization o_parent ON o_parent.id = d.organization_id
     WHERE duc.organization_id = ANY(org_ids)
       AND (duc.is_underlying = TRUE OR duc.is_value_driver = TRUE)
       AND d.status IN ('Portfolio Company', 'Partnership')
),
underlying_top AS (
    -- Sort dealcloud-confirmed rows ahead of llm_derived-only ones, then
    -- by date. Keeps highest-confidence positions visible first.
    SELECT * FROM underlying_all
     ORDER BY (connection_source = 'llm_derived') ASC,
              date DESC NULLS LAST, deal_id DESC
     LIMIT 5
),
target_doc_ids AS (
    SELECT DISTINCT doa.document_id AS doc_id
      FROM dealcloud.document_organization_alias doa
     WHERE doa.organization_alias_id IN (
           SELECT alias_id FROM dealcloud.todd_alias_ids(org_ids))
),
-- Cap: skip docs that mention >10 distinct orgs. Those are firm-level
-- overview decks / fund summaries, not deal-specific evidence.
target_doc_ids_filtered AS (
    SELECT td.doc_id
      FROM target_doc_ids td
     WHERE (
        SELECT COUNT(DISTINCT oa.organization_id)
          FROM dealcloud.document_organization_alias doa2
          JOIN dealcloud.organization_alias oa ON oa.id = doa2.organization_alias_id
         WHERE doa2.document_id = td.doc_id
     ) <= 10
),
dc_counterparty_orgs AS (
    SELECT DISTINCT organization_id
      FROM dealcloud.deal
     WHERE organization_id IS NOT NULL
),
hint_pairs AS (
    -- (document, co-mentioned DC counterparty) pairs, post-cap
    SELECT td.doc_id,
           oa_other.organization_id AS co_org_id,
           o_other.name             AS co_org_name
      FROM target_doc_ids_filtered td
      JOIN dealcloud.document_organization_alias doa_other
           ON doa_other.document_id = td.doc_id
      JOIN dealcloud.organization_alias oa_other
           ON oa_other.id = doa_other.organization_alias_id
      JOIN dealcloud.organization o_other
           ON o_other.id = oa_other.organization_id
     WHERE oa_other.organization_id IN (SELECT organization_id FROM dc_counterparty_orgs)
       AND NOT (oa_other.organization_id = ANY(org_ids))
     GROUP BY 1, 2, 3
),
hint_top AS (
    SELECT hp.doc_id        AS document_id,
           doc.name          AS document_name,
           hp.co_org_id      AS co_mentioned_dc_org_id,
           hp.co_org_name    AS co_mentioned_dc_org_name,
           LEFT(COALESCE(doc.summary, ''), 200) AS context_snippet
      FROM hint_pairs hp
      JOIN dealcloud.document doc ON doc.id = hp.doc_id
     ORDER BY doc.modified_at DESC NULLS LAST
     LIMIT 3
)
SELECT jsonb_build_object(
    'in_portfolio',
        ((SELECT COUNT(*) FROM counterparty_all) > 0
         OR (SELECT COUNT(*) FROM underlying_all) > 0),
    'as_counterparty', jsonb_build_object(
        'count', (SELECT COUNT(*) FROM counterparty_all),
        'deals', COALESCE(
            (SELECT jsonb_agg(to_jsonb(c)) FROM counterparty_top c),
            '[]'::jsonb)
    ),
    'as_underlying', jsonb_build_object(
        'count', (SELECT COUNT(*) FROM underlying_all),
        'deals', COALESCE(
            (SELECT jsonb_agg(to_jsonb(u)) FROM underlying_top u),
            '[]'::jsonb)
    ),
    'doc_only_underlying_hints', jsonb_build_object(
        'count', (SELECT COUNT(*) FROM hint_pairs),
        'samples', COALESCE(
            (SELECT jsonb_agg(to_jsonb(h)) FROM hint_top h),
            '[]'::jsonb)
    )
);
$$;

-- =====================================================================
-- Q2: org_deal_history
-- =====================================================================
-- Same data as Q1 minus the status filter. Includes Closed / Dropped /
-- in-flight / etc.
CREATE OR REPLACE FUNCTION dealcloud.org_deal_history(org_ids INTEGER[])
RETURNS jsonb LANGUAGE sql STABLE AS $$
WITH
counterparty_all AS (
    SELECT d.id AS deal_id, d.name, d.transaction_type, d.status,
           d.new_deal_date AS date
      FROM dealcloud.deal d
     WHERE d.organization_id = ANY(org_ids)
),
counterparty_top AS (
    SELECT * FROM counterparty_all
     ORDER BY date DESC NULLS LAST, deal_id DESC
     LIMIT 10
),
underlying_all AS (
    -- See Q1's underlying_all comment re: is_underlying vs is_value_driver.
    SELECT d.id AS deal_id, d.name AS deal_name,
           o_parent.name AS parent_org_name,
           d.status, d.new_deal_date AS date,
           duc.connection_source,
           duc.derived_n_docs,
           duc.is_value_driver
      FROM dealcloud.deal_underlying_company duc
      JOIN dealcloud.deal d ON d.id = duc.deal_id
      LEFT JOIN dealcloud.organization o_parent ON o_parent.id = d.organization_id
     WHERE duc.organization_id = ANY(org_ids)
       AND (duc.is_underlying = TRUE OR duc.is_value_driver = TRUE)
),
underlying_top AS (
    SELECT * FROM underlying_all
     ORDER BY (connection_source = 'llm_derived') ASC,
              date DESC NULLS LAST, deal_id DESC
     LIMIT 10
),
status_breakdown AS (
    SELECT COALESCE(d.status, 'Unknown') AS status, COUNT(*) AS n
      FROM (
            SELECT status FROM counterparty_all
            UNION ALL
            SELECT status FROM underlying_all
      ) d
     GROUP BY 1
)
SELECT jsonb_build_object(
    'assessed',
        ((SELECT COUNT(*) FROM counterparty_all) > 0
         OR (SELECT COUNT(*) FROM underlying_all) > 0),
    'deals_total',
        ((SELECT COUNT(*) FROM counterparty_all)
         + (SELECT COUNT(*) FROM underlying_all)),
    'by_status', COALESCE(
        (SELECT jsonb_object_agg(status, n) FROM status_breakdown),
        '{}'::jsonb),
    'as_counterparty', COALESCE(
        (SELECT jsonb_agg(to_jsonb(c)) FROM counterparty_top c),
        '[]'::jsonb),
    'as_underlying', COALESCE(
        (SELECT jsonb_agg(to_jsonb(u)) FROM underlying_top u),
        '[]'::jsonb)
);
$$;

-- =====================================================================
-- Q3: org_ion_contacts
-- =====================================================================
-- Active definitions per channel:
--   email:            email_thread_participant.message_count > 0
--   calendar:         response_status IN ('accepted','tentativelyAccepted')
--                     OR is_organizer
--   dc_communication: every logged interaction = active
--   slack:            NOT broken down per Ion employee in V1.
--                     slack_message_group.participants is a TEXT[] of
--                     real-names (no email mapping); attribution is
--                     unreliable. Slack still rolls up into Q5's
--                     channel totals.
CREATE OR REPLACE FUNCTION dealcloud.org_ion_contacts(org_ids INTEGER[])
RETURNS jsonb LANGUAGE sql STABLE AS $$
WITH
alias_ids AS (SELECT alias_id FROM dealcloud.todd_alias_ids(org_ids)),
-- Email touches per Ion email
email_touches AS (
    SELECT etp.email AS ion_email,
           MAX(etp.name) AS ion_name,
           SUM(CASE WHEN etp.message_count > 0 THEN 1 ELSE 0 END)::INT  AS active,
           SUM(CASE WHEN etp.message_count = 0 THEN 1 ELSE 0 END)::INT  AS passive,
           MIN(et.first_message_at) AS first_touch,
           MAX(et.last_message_at)  AS last_touch
      FROM dealcloud.email_thread_organization eto
      JOIN dealcloud.email_thread et ON et.id = eto.thread_id
      JOIN dealcloud.email_thread_participant etp ON etp.thread_id = et.id
     WHERE eto.organization_alias_id IN (SELECT alias_id FROM alias_ids)
       AND etp.is_internal = TRUE
     GROUP BY etp.email
),
-- Calendar touches per Ion email
calendar_touches AS (
    SELECT cep.email AS ion_email,
           MAX(cep.name) AS ion_name,
           SUM(CASE WHEN cep.is_organizer
                       OR cep.response_status IN ('accepted','tentativelyAccepted')
                    THEN 1 ELSE 0 END)::INT AS active,
           SUM(CASE WHEN NOT (cep.is_organizer
                       OR cep.response_status IN ('accepted','tentativelyAccepted'))
                    THEN 1 ELSE 0 END)::INT AS passive,
           MIN(ce.start_time) AS first_touch,
           MAX(ce.start_time) AS last_touch
      FROM dealcloud.calendar_event_organization ceo
      JOIN dealcloud.calendar_event ce ON ce.id = ceo.event_id
      JOIN dealcloud.calendar_event_participant cep ON cep.event_id = ce.id
     WHERE ceo.organization_alias_id IN (SELECT alias_id FROM alias_ids)
       AND cep.is_internal = TRUE
     GROUP BY cep.email
),
-- DC communications per Ion email (all active by definition)
comm_touches AS (
    SELECT ce.employee_email AS ion_email,
           MAX(ce.employee_name) AS ion_name,
           COUNT(*)::INT AS active,
           0::INT        AS passive,
           MIN(c.date)   AS first_touch,
           MAX(c.date)   AS last_touch
      FROM dealcloud.communication_organization co
      JOIN dealcloud.communication c ON c.id = co.communication_id
      JOIN dealcloud.communication_employee ce ON ce.communication_id = c.id
     WHERE co.organization_id = ANY(org_ids)
     GROUP BY ce.employee_email
),
unioned AS (
    SELECT ion_email, ion_name, 'email'  AS channel, active, passive,
           first_touch, last_touch FROM email_touches
    UNION ALL
    SELECT ion_email, ion_name, 'calendar', active, passive,
           first_touch, last_touch FROM calendar_touches
    UNION ALL
    SELECT ion_email, ion_name, 'dc_communication', active, passive,
           first_touch, last_touch FROM comm_touches
),
per_contact AS (
    SELECT ion_email,
           MAX(ion_name) AS ion_name,
           SUM(active)::INT  AS active_total,
           SUM(passive)::INT AS passive_total,
           jsonb_object_agg(channel, active)
                FILTER (WHERE active > 0)  AS by_channel_active,
           jsonb_object_agg(channel, passive)
                FILTER (WHERE passive > 0) AS by_channel_passive,
           MIN(first_touch) AS first_touch,
           MAX(last_touch)  AS last_touch
      FROM unioned
     WHERE ion_email IS NOT NULL
     GROUP BY ion_email
),
top_contacts AS (
    SELECT * FROM per_contact
     ORDER BY (active_total > 0) DESC,
              active_total DESC,
              passive_total DESC,
              last_touch DESC NULLS LAST
     LIMIT 5
),
top_contacts_json AS (
    SELECT jsonb_build_object(
        'ion_email',          ion_email,
        'ion_name',           ion_name,
        'active_touches',     active_total,
        'passive_touches',    passive_total,
        'by_channel_active',  COALESCE(by_channel_active,  '{}'::jsonb),
        'by_channel_passive', COALESCE(by_channel_passive, '{}'::jsonb),
        'first_touch',        first_touch,
        'last_touch',         last_touch
    ) AS j
      FROM top_contacts
),
-- Last-touch-by-channel: pick the most recent thread/event/comm/slack
-- linked to the org bundle, return a small label.
last_email AS (
    SELECT et.last_message_at AS date,
           et.subject          AS subject_or_summary
      FROM dealcloud.email_thread_organization eto
      JOIN dealcloud.email_thread et ON et.id = eto.thread_id
     WHERE eto.organization_alias_id IN (SELECT alias_id FROM alias_ids)
     ORDER BY et.last_message_at DESC NULLS LAST LIMIT 1
),
last_calendar AS (
    SELECT ce.start_time AS date, ce.subject AS subject
      FROM dealcloud.calendar_event_organization ceo
      JOIN dealcloud.calendar_event ce ON ce.id = ceo.event_id
     WHERE ceo.organization_alias_id IN (SELECT alias_id FROM alias_ids)
     ORDER BY ce.start_time DESC NULLS LAST LIMIT 1
),
last_slack AS (
    SELECT TO_TIMESTAMP(CAST(SPLIT_PART(smg.last_ts, '.', 1) AS BIGINT)) AS date,
           sc.name AS channel,
           LEFT(COALESCE(smg.summary, ''), 200) AS summary
      FROM dealcloud.slack_message_group_organization smgo
      JOIN dealcloud.slack_message_group smg ON smg.id = smgo.message_group_id
      JOIN dealcloud.slack_channel sc ON sc.id = smg.channel_id
     WHERE smgo.organization_alias_id IN (SELECT alias_id FROM alias_ids)
       AND smg.last_ts ~ '^[0-9]+(\.[0-9]+)?$'
     ORDER BY smg.last_ts DESC LIMIT 1
),
last_comm AS (
    SELECT c.date, c.communication_type AS type, c.subject
      FROM dealcloud.communication_organization co
      JOIN dealcloud.communication c ON c.id = co.communication_id
     WHERE co.organization_id = ANY(org_ids)
     ORDER BY c.date DESC NULLS LAST LIMIT 1
)
SELECT jsonb_build_object(
    'top_contacts', COALESCE(
        (SELECT jsonb_agg(j) FROM top_contacts_json),
        '[]'::jsonb),
    'last_touch_overall', GREATEST(
        (SELECT date FROM last_email),
        (SELECT date FROM last_calendar),
        (SELECT date FROM last_slack),
        (SELECT date FROM last_comm)),
    'last_touch_by_channel', jsonb_build_object(
        'email',            (SELECT to_jsonb(le) FROM last_email le),
        'calendar',         (SELECT to_jsonb(lc) FROM last_calendar lc),
        'slack',            (SELECT to_jsonb(ls) FROM last_slack ls),
        'dc_communication', (SELECT to_jsonb(lcm) FROM last_comm lcm))
);
$$;

-- =====================================================================
-- Q4: org_their_contacts
-- =====================================================================
-- Domain-match boost: if the contact's email domain appears in
-- domain_organization for any org in the bundle, surface first.
-- Falls back to active vs passive ranking.
CREATE OR REPLACE FUNCTION dealcloud.org_their_contacts(org_ids INTEGER[])
RETURNS jsonb LANGUAGE sql STABLE AS $$
WITH
alias_ids AS (SELECT alias_id FROM dealcloud.todd_alias_ids(org_ids)),
their_domains AS (
    SELECT DISTINCT domain
      FROM dealcloud.domain_organization
     WHERE organization_id = ANY(org_ids)
),
-- External email touches via email_thread_participant
email_external AS (
    SELECT etp.email,
           MAX(etp.name) AS name,
           SUM(CASE WHEN etp.message_count > 0 THEN 1 ELSE 0 END)::INT AS active,
           SUM(CASE WHEN etp.message_count = 0 THEN 1 ELSE 0 END)::INT AS passive,
           MIN(et.first_message_at) AS first_touch,
           MAX(et.last_message_at)  AS last_touch
      FROM dealcloud.email_thread_organization eto
      JOIN dealcloud.email_thread et ON et.id = eto.thread_id
      JOIN dealcloud.email_thread_participant etp ON etp.thread_id = et.id
     WHERE eto.organization_alias_id IN (SELECT alias_id FROM alias_ids)
       AND etp.is_internal = FALSE
       AND etp.email IS NOT NULL
     GROUP BY etp.email
),
-- External calendar
calendar_external AS (
    SELECT cep.email,
           MAX(cep.name) AS name,
           SUM(CASE WHEN cep.is_organizer
                       OR cep.response_status IN ('accepted','tentativelyAccepted')
                    THEN 1 ELSE 0 END)::INT AS active,
           SUM(CASE WHEN NOT (cep.is_organizer
                       OR cep.response_status IN ('accepted','tentativelyAccepted'))
                    THEN 1 ELSE 0 END)::INT AS passive,
           MIN(ce.start_time) AS first_touch,
           MAX(ce.start_time) AS last_touch
      FROM dealcloud.calendar_event_organization ceo
      JOIN dealcloud.calendar_event ce ON ce.id = ceo.event_id
      JOIN dealcloud.calendar_event_participant cep ON cep.event_id = ce.id
     WHERE ceo.organization_alias_id IN (SELECT alias_id FROM alias_ids)
       AND cep.is_internal = FALSE
       AND cep.email IS NOT NULL
     GROUP BY cep.email
),
-- External via DC contacts surface — every DC-tracked contact attached
-- to any org in the bundle, plus their CRM-logged communications.
comm_external AS (
    SELECT ct.email,
           MAX(COALESCE(ct.first_name || ' ' || ct.last_name,
                        ct.first_name, ct.last_name)) AS name,
           COUNT(*)::INT AS active,
           0::INT        AS passive,
           MIN(c.date)   AS first_touch,
           MAX(c.date)   AS last_touch
      FROM dealcloud.communication_organization co
      JOIN dealcloud.communication c ON c.id = co.communication_id
      JOIN dealcloud.communication_contact cc ON cc.communication_id = c.id
      JOIN dealcloud.contact ct ON ct.id = cc.contact_id
     WHERE co.organization_id = ANY(org_ids)
       AND ct.email IS NOT NULL
     GROUP BY ct.email
),
unioned AS (
    SELECT email, name, 'email' AS channel, active, passive, first_touch, last_touch
      FROM email_external
    UNION ALL
    SELECT email, name, 'calendar', active, passive, first_touch, last_touch
      FROM calendar_external
    UNION ALL
    SELECT email, name, 'dc_communication', active, passive, first_touch, last_touch
      FROM comm_external
),
per_email AS (
    SELECT email,
           MAX(name) AS name,
           SUM(active)::INT  AS active_total,
           SUM(passive)::INT AS passive_total,
           jsonb_object_agg(channel, active)
                FILTER (WHERE active > 0)  AS by_channel_active,
           jsonb_object_agg(channel, passive)
                FILTER (WHERE passive > 0) AS by_channel_passive,
           MIN(first_touch) AS first_touch,
           MAX(last_touch)  AS last_touch
      FROM unioned
     GROUP BY email
),
-- Annotate with DC-contact metadata when present. DISTINCT ON keeps
-- one row per email even when a contact has several contact_organization
-- entries; the ORDER BY picks the strongest relationship.
annotated AS (
    SELECT DISTINCT ON (pe.email)
           pe.email, pe.name,
           pe.active_total, pe.passive_total,
           pe.by_channel_active, pe.by_channel_passive,
           pe.first_touch, pe.last_touch,
           ct.id AS contact_id,
           ct.job_title,
           ct.is_in_dealcloud,
           co_org.relationship_type AS dc_relationship,
           (SUBSTRING(pe.email FROM POSITION('@' IN pe.email) + 1)
                IN (SELECT domain FROM their_domains)) AS domain_matches_org
      FROM per_email pe
      LEFT JOIN dealcloud.contact ct ON ct.email = pe.email
      LEFT JOIN dealcloud.contact_organization co_org
             ON co_org.contact_id = ct.id
            AND co_org.organization_id = ANY(org_ids)
     ORDER BY pe.email,
              CASE co_org.relationship_type
                WHEN 'primary'              THEN 1
                WHEN 'secondary'            THEN 2
                WHEN 'previous_employment'  THEN 3
                ELSE 4
              END NULLS LAST,
              ct.id NULLS LAST
),
top_contacts AS (
    SELECT * FROM annotated
     ORDER BY domain_matches_org DESC NULLS LAST,
              (active_total > 0) DESC,
              active_total DESC,
              passive_total DESC,
              last_touch DESC NULLS LAST
     LIMIT 5
),
top_contacts_json AS (
    SELECT jsonb_build_object(
        'contact_id',         contact_id,
        'name',               name,
        'email',              email,
        'job_title',          job_title,
        'in_dealcloud',       COALESCE(is_in_dealcloud, FALSE),
        'dc_relationship',    dc_relationship,
        'domain_matches_org', COALESCE(domain_matches_org, FALSE),
        'active_touches',     active_total,
        'passive_touches',    passive_total,
        'by_channel_active',  COALESCE(by_channel_active,  '{}'::jsonb),
        'by_channel_passive', COALESCE(by_channel_passive, '{}'::jsonb),
        'first_touch',        first_touch,
        'last_touch',         last_touch
    ) AS j
      FROM top_contacts
)
SELECT jsonb_build_object(
    'top_contacts', COALESCE(
        (SELECT jsonb_agg(j) FROM top_contacts_json),
        '[]'::jsonb),
    'total_distinct_contacts',
        (SELECT COUNT(*) FROM per_email),
    'non_dc_contacts_count',
        (SELECT COUNT(*) FROM per_email pe
          WHERE NOT EXISTS (SELECT 1 FROM dealcloud.contact ct
                             WHERE ct.email = pe.email)),
    'their_domains', COALESCE(
        (SELECT jsonb_agg(domain ORDER BY domain) FROM their_domains),
        '[]'::jsonb)
);
$$;

-- =====================================================================
-- Q5: org_communication_timeline
-- =====================================================================
-- Per-channel counts + first/last + ascending activity-by-quarter
-- (zero-count quarters skipped). Documents split into total vs
-- deal-related (deal-related = at least one document_deal row).
CREATE OR REPLACE FUNCTION dealcloud.org_communication_timeline(org_ids INTEGER[])
RETURNS jsonb LANGUAGE sql STABLE AS $$
WITH
alias_ids AS (SELECT alias_id FROM dealcloud.todd_alias_ids(org_ids)),
email_dates AS (
    SELECT et.last_message_at AS ts
      FROM dealcloud.email_thread_organization eto
      JOIN dealcloud.email_thread et ON et.id = eto.thread_id
     WHERE eto.organization_alias_id IN (SELECT alias_id FROM alias_ids)
),
calendar_dates AS (
    SELECT ce.start_time AS ts
      FROM dealcloud.calendar_event_organization ceo
      JOIN dealcloud.calendar_event ce ON ce.id = ceo.event_id
     WHERE ceo.organization_alias_id IN (SELECT alias_id FROM alias_ids)
),
slack_dates AS (
    SELECT TO_TIMESTAMP(CAST(SPLIT_PART(smg.last_ts, '.', 1) AS BIGINT)) AS ts
      FROM dealcloud.slack_message_group_organization smgo
      JOIN dealcloud.slack_message_group smg ON smg.id = smgo.message_group_id
     WHERE smgo.organization_alias_id IN (SELECT alias_id FROM alias_ids)
       AND smg.last_ts ~ '^[0-9]+(\.[0-9]+)?$'
),
comm_dates AS (
    SELECT c.date AS ts
      FROM dealcloud.communication_organization co
      JOIN dealcloud.communication c ON c.id = co.communication_id
     WHERE co.organization_id = ANY(org_ids)
),
doc_rows AS (
    SELECT DISTINCT doa.document_id AS doc_id
      FROM dealcloud.document_organization_alias doa
     WHERE doa.organization_alias_id IN (SELECT alias_id FROM alias_ids)
),
doc_dates AS (
    -- "deal_related" = path lives under a deal-files folder or a Project
    -- folder. document_deal is too sparse (~800 rows total) to be the
    -- primary signal; path-based catches the bulk of deal-folder content
    -- on both the "Deal Files" drive and ION Pacific Share/Common/Deal
    -- files. ILIKE is case-insensitive in PostgreSQL.
    SELECT doc.modified_at AS ts,
           (doc.path ILIKE '%/Deal files/%'
            OR doc.path ILIKE '%/Project %') AS deal_related
      FROM doc_rows dr
      JOIN dealcloud.document doc ON doc.id = dr.doc_id
),
all_touches AS (
    SELECT ts FROM email_dates    WHERE ts IS NOT NULL
    UNION ALL SELECT ts FROM calendar_dates WHERE ts IS NOT NULL
    UNION ALL SELECT ts FROM slack_dates    WHERE ts IS NOT NULL
    UNION ALL SELECT ts FROM comm_dates     WHERE ts IS NOT NULL
    UNION ALL SELECT ts FROM doc_dates      WHERE ts IS NOT NULL
),
quarters AS (
    SELECT to_char(date_trunc('quarter', ts), 'YYYY-"Q"Q') AS quarter,
           COUNT(*)::INT AS n
      FROM all_touches
     GROUP BY 1
     ORDER BY 1
),
agg AS (
    SELECT
        (SELECT MIN(ts) FROM all_touches) AS first_touch,
        (SELECT MAX(ts) FROM all_touches) AS last_touch,
        (SELECT COUNT(*) FROM all_touches) AS total_touches
)
SELECT jsonb_build_object(
    'first_touch',  agg.first_touch,
    'last_touch',   agg.last_touch,
    'duration_days',
        CASE WHEN agg.first_touch IS NULL OR agg.last_touch IS NULL THEN NULL
             ELSE EXTRACT(DAY FROM agg.last_touch - agg.first_touch)::INT END,
    'total_touches', agg.total_touches,
    'by_channel', jsonb_build_object(
        'email', jsonb_build_object(
            'count', (SELECT COUNT(*) FROM email_dates WHERE ts IS NOT NULL),
            'first', (SELECT MIN(ts) FROM email_dates),
            'last',  (SELECT MAX(ts) FROM email_dates)),
        'slack', jsonb_build_object(
            'count', (SELECT COUNT(*) FROM slack_dates WHERE ts IS NOT NULL),
            'first', (SELECT MIN(ts) FROM slack_dates),
            'last',  (SELECT MAX(ts) FROM slack_dates)),
        'calendar', jsonb_build_object(
            'count', (SELECT COUNT(*) FROM calendar_dates WHERE ts IS NOT NULL),
            'first', (SELECT MIN(ts) FROM calendar_dates),
            'last',  (SELECT MAX(ts) FROM calendar_dates)),
        'dc_communication', jsonb_build_object(
            'count', (SELECT COUNT(*) FROM comm_dates WHERE ts IS NOT NULL),
            'first', (SELECT MIN(ts) FROM comm_dates),
            'last',  (SELECT MAX(ts) FROM comm_dates)),
        'documents', jsonb_build_object(
            'count_total',        (SELECT COUNT(*) FROM doc_rows),
            'count_deal_related', (SELECT COUNT(*) FROM doc_dates
                                    WHERE deal_related),
            'first', (SELECT MIN(ts) FROM doc_dates),
            'last',  (SELECT MAX(ts) FROM doc_dates))
    ),
    'activity_by_quarter', COALESCE(
        (SELECT jsonb_agg(jsonb_build_object('quarter', quarter, 'count', n)
                          ORDER BY quarter)
           FROM quarters),
        '[]'::jsonb)
)
FROM agg;
$$;

-- =====================================================================
-- Slack conversation state
-- =====================================================================
-- Tracks the per-Slack-thread (or DM channel) Todd state machine.
-- (team_id, channel_id, thread_ts) is the lookup key for "is there a
-- live conversation here?". thread_ts is NULL for DMs.

CREATE TABLE IF NOT EXISTS research.slack_conversation (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id               TEXT NOT NULL,
    channel_id            TEXT NOT NULL,
    thread_ts             TEXT,                  -- NULL for DMs
    user_email            TEXT NOT NULL,
    slack_user_id         TEXT NOT NULL,
    phase                 TEXT NOT NULL DEFAULT 'search'
                          CHECK (phase IN ('search','disambiguating','answering','done','error')),
    selected_org_ids      INTEGER[],
    bundled_canonical_id  INTEGER,               -- the canonical org head for the bundle
    research_session_id   UUID REFERENCES research.session(id) ON DELETE SET NULL,
    -- Anthropic message history (list of role/content blocks). Trimmed
    -- to last ~20 turns when persisted. Used to resume tool loops.
    message_history       JSONB NOT NULL DEFAULT '[]'::jsonb,
    last_error            TEXT,
    started_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at              TIMESTAMPTZ
);

-- One live conversation per (team, channel, thread). Closed conversations
-- (ended_at IS NOT NULL) are exempt so a follow-up Slack mention can
-- start a new one in the same channel.
CREATE UNIQUE INDEX IF NOT EXISTS idx_slack_conv_lookup
    ON research.slack_conversation(team_id, channel_id, COALESCE(thread_ts, ''))
 WHERE ended_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_slack_conv_user
    ON research.slack_conversation(user_email);
CREATE INDEX IF NOT EXISTS idx_slack_conv_started
    ON research.slack_conversation(started_at DESC);

-- =====================================================================
-- Slack event idempotency
-- =====================================================================
-- Slack retries up to 3x on timeout. Dedupe on Slack's event_id.
-- Pruned periodically (rows older than 1 day are safe to drop).

CREATE TABLE IF NOT EXISTS research.slack_event_dedupe (
    event_id     TEXT PRIMARY KEY,
    received_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_slack_dedupe_received
    ON research.slack_event_dedupe(received_at);
