-- 003_todd_bundle.sql -- Bundle helper for Todd. Walks
-- organization.superseded_by_org_id chains and returns the distinct
-- terminal canonical IDs for a set of input org_ids. Idempotent.
--
-- Canonical = the org row whose superseded_by_org_id IS NULL. After the
-- 2026-04-29 consolidation pass, ~1,073 orgs were marked superseded; a
-- search hit for any of them should resolve to the canonical head.
--
-- When the cluster-first redesign ships (per org_clustering_redesign.md)
-- this function will be replaced by `bundle_via_cluster_head` -- same
-- signature, swap the body. Todd-side callers won't change.

SET search_path TO research, dealcloud, public;

CREATE OR REPLACE FUNCTION dealcloud.bundle_via_supersede(org_ids INTEGER[])
RETURNS INTEGER[] LANGUAGE sql STABLE AS $$
    WITH RECURSIVE walker(id, depth) AS (
        SELECT id, 0
          FROM dealcloud.organization
         WHERE id = ANY(org_ids)
        UNION ALL
        SELECT o.superseded_by_org_id, walker.depth + 1
          FROM walker
          JOIN dealcloud.organization o ON o.id = walker.id
         WHERE o.superseded_by_org_id IS NOT NULL
           AND walker.depth < 20  -- defensive: cycle/long-chain guard
    )
    SELECT array_agg(DISTINCT canonical_id)
      FROM (
            SELECT walker.id AS canonical_id
              FROM walker
              JOIN dealcloud.organization o ON o.id = walker.id
             WHERE o.superseded_by_org_id IS NULL
      ) c;
$$;
