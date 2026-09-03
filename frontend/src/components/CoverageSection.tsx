import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, type CoverageCriterion } from "../lib/api";

function formatReviewedAt(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return iso;
  }
}

/**
 * Data-room checklist coverage (see memory: data_room_coverage_analysis).
 * Read-only view + a "Scan" action that drains the room's not-yet-checked
 * documents in polled batches (each batch is a real, bounded Gemini call on
 * dce -- never a silent background loop, always a visible "Scanning…"
 * state the user asked for). All Found/Unconfirmed/Candidate-Gap logic
 * lives in deal_cloud_enhancer; this component only renders it.
 *
 * Statuses (see api.ts's CoverageCriterion.status):
 *   Found                                            -- confident, trust it
 *   Found — high hit count, review before trusting    -- confident-looking
 *     but the hit count is a statistical outlier for this room; the design
 *     found this catches real over-triggering (e.g. a vague catch-all
 *     criterion matching many unrelated docs) as often as it catches
 *     legitimately-numerous evidence (e.g. one NDA per counterparty) --
 *     it's a "look at this" flag, not a verdict either way.
 *   Found (keyword only...)                          -- filename/summary
 *     matched a known alias but no LLM-confirmed hit; weaker than Found.
 *   Unconfirmed                                       -- NOT a gap. A doc
 *     that could plausibly hold this couldn't be read (unsupported format,
 *     OCR gap, or -- for criteria without keyword aliases yet -- any
 *     unreadable doc in the room at all). Never state this to a
 *     counterparty as missing.
 *   Candidate Gap                                     -- the only status
 *     that means "we looked everywhere we could read and found nothing."
 *     Still needs human confirm before referencing externally -- see the
 *     design's human-review-gate principle. GapReviewControls below is
 *     that gate: Confirm ("real gap, chase the counterparty") or Dismiss
 *     ("not applicable to this deal"), recorded append-only server-side
 *     so a later change of mind is auditable, not a silent overwrite.
 *   Scanning                                          -- not yet processed
 *     by the checklist matcher. Click "Scan" to process it.
 */

const STATUS_STYLE: Record<string, string> = {
  Found: "bg-emerald-100 text-emerald-800 border-emerald-300",
  "Found — high hit count, review before trusting":
    "bg-amber-100 text-amber-800 border-amber-300",
  "Found (keyword only — not LLM-confirmed, needs review)":
    "bg-amber-50 text-amber-700 border-amber-200",
  Unconfirmed: "bg-slate-100 text-slate-600 border-slate-300",
  "Candidate Gap": "bg-red-100 text-red-800 border-red-300",
  Scanning: "bg-blue-50 text-blue-600 border-blue-200",
};

function StatusBadge({ status }: { status: string }) {
  const style = STATUS_STYLE[status] ?? "bg-slate-100 text-slate-600 border-slate-300";
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium border ${style}`}
    >
      {status}
    </span>
  );
}

/** The human-review gate for a Candidate Gap (step 10a). Shows the current
 * review (if any) with a "change" affordance, or Confirm/Dismiss buttons +
 * an optional note if never reviewed. Every action appends a new review
 * server-side (see api.ts) -- re-reviewing after a change of mind is
 * expected, not blocked. */
function GapReviewControls({
  roomId,
  criterionId,
  review,
}: {
  roomId: number;
  criterionId: number;
  review: CoverageCriterion["review"];
}) {
  const qc = useQueryClient();
  const [editing, setEditing] = useState(!review);
  const [note, setNote] = useState("");
  const [submitting, setSubmitting] = useState<"confirmed_gap" | "dismissed" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const submit = async (status: "confirmed_gap" | "dismissed") => {
    setSubmitting(status);
    setError(null);
    try {
      await api.setDataRoomCoverageReview(roomId, criterionId, status, note || undefined);
      await qc.invalidateQueries({ queryKey: ["data-room-coverage", roomId] });
      setEditing(false);
      setNote("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(null);
    }
  };

  if (review && !editing) {
    const label = review.status === "confirmed_gap" ? "Confirmed gap" : "Dismissed";
    const style =
      review.status === "confirmed_gap"
        ? "text-red-700 bg-red-50 border-red-200"
        : "text-slate-500 bg-slate-50 border-slate-200";
    return (
      <div
        className={`mt-2 ml-1 px-2 py-1.5 rounded border text-xs flex items-start justify-between gap-2 ${style}`}
      >
        <div>
          <span className="font-medium">{label}</span> by {review.reviewed_by} on{" "}
          {formatReviewedAt(review.reviewed_at)}
          {review.note && <div className="mt-0.5 italic">"{review.note}"</div>}
        </div>
        <button
          type="button"
          onClick={() => setEditing(true)}
          className="text-slate-400 hover:text-slate-600 underline shrink-0"
        >
          change
        </button>
      </div>
    );
  }

  return (
    <div className="mt-2 ml-1 space-y-1.5">
      <input
        type="text"
        value={note}
        onChange={(e) => setNote(e.target.value)}
        placeholder="Optional note (e.g. why this is/isn't a real gap)"
        className="w-full text-xs border border-slate-300 rounded px-2 py-1 focus:outline-none focus:border-slate-500"
      />
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => void submit("confirmed_gap")}
          disabled={submitting !== null}
          className="text-xs px-2.5 py-1 rounded bg-red-600 text-white hover:bg-red-700 disabled:opacity-50"
        >
          {submitting === "confirmed_gap" ? "Confirming…" : "Confirm gap"}
        </button>
        <button
          type="button"
          onClick={() => void submit("dismissed")}
          disabled={submitting !== null}
          className="text-xs px-2.5 py-1 rounded border border-slate-300 text-slate-600 hover:bg-slate-50 disabled:opacity-50"
        >
          {submitting === "dismissed" ? "Dismissing…" : "Dismiss (not applicable)"}
        </button>
        {review && (
          <button
            type="button"
            onClick={() => setEditing(false)}
            className="text-xs text-slate-400 hover:text-slate-600 underline"
          >
            cancel
          </button>
        )}
      </div>
      {error && <div className="text-xs text-red-600">{error}</div>}
    </div>
  );
}

function CriterionRow({ roomId, c }: { roomId: number; c: CoverageCriterion }) {
  const isGap = c.status === "Candidate Gap";
  const hasDetail = c.hits.length > 0 || c.keyword_hits.length > 0;
  const [expanded, setExpanded] = useState(false);
  const canExpand = hasDetail || isGap;

  return (
    <div className="px-3 py-2">
      <button
        type="button"
        onClick={() => canExpand && setExpanded((v) => !v)}
        className={
          "w-full flex items-center justify-between gap-2 text-left " +
          (canExpand ? "cursor-pointer" : "cursor-default")
        }
      >
        <span className="text-sm text-slate-700 flex-1">
          {c.criterion}
          {c.importance === "core" && (
            <span className="ml-1.5 text-[10px] uppercase tracking-wide text-slate-400">
              core
            </span>
          )}
          {isGap && c.review && (
            <span className="ml-1.5 text-[10px] uppercase tracking-wide text-slate-400">
              reviewed
            </span>
          )}
        </span>
        <StatusBadge status={c.status} />
      </button>

      {expanded && hasDetail && (
        <div className="mt-2 ml-1 pl-2 border-l-2 border-slate-200 space-y-1.5">
          {c.hit_total != null && c.hit_total > c.hits.length && (
            <div className="text-[10px] uppercase tracking-wide text-slate-400">
              {c.hits.length} document{c.hits.length === 1 ? "" : "s"} ·{" "}
              {c.hit_total} matching files
            </div>
          )}
          {c.hits.map((h, i) => (
            <div key={i} className="text-xs">
              <span className="font-medium text-slate-600">{h.doc_name}</span>
              {h.copies != null && h.copies > 1 && (
                <span className="text-slate-400">
                  {" "}
                  ({h.copies} copies)
                </span>
              )}
              <span className="text-slate-400"> ({h.present}) — </span>
              <span className="text-slate-500 italic">"{h.evidence}"</span>
            </div>
          ))}
          {c.keyword_hits.length > 0 && (
            <div className="text-xs text-slate-500">
              Filename/summary match only: {c.keyword_hits.join(", ")}
            </div>
          )}
        </div>
      )}

      {expanded && isGap && (
        <GapReviewControls roomId={roomId} criterionId={c.criterion_id} review={c.review} />
      )}
    </div>
  );
}

export default function CoverageSection({ roomId }: { roomId: number }) {
  const qc = useQueryClient();
  const [scanning, setScanning] = useState(false);
  const [scanPhase, setScanPhase] = useState<
    "reading_files" | "classifying"
  >("classifying");
  const [docsRead, setDocsRead] = useState(0);
  const [scanError, setScanError] = useState<string | null>(null);
  const [collapsedCats, setCollapsedCats] = useState<Set<string>>(new Set());

  const coverage = useQuery({
    queryKey: ["data-room-coverage", roomId],
    queryFn: () => api.getDataRoomCoverage(roomId),
  });

  const runScan = async () => {
    setScanning(true);
    setScanError(null);
    setDocsRead(0);
    try {
      // Drain in a polled loop -- each call is one bounded, real batch on
      // dce (~25 docs, real Gemini calls, a few seconds each). Deliberately
      // sequential (await, not fire-and-forget) so we never have two scans
      // racing the same room, and the visible "Scanning…" state stays
      // accurate to what's actually happening.
      for (;;) {
        const res = await api.scanDataRoomCoverageBatch(roomId);
        // A room with unread documents spends its first calls in dce's
        // 'reading_files' phase. Label that distinctly: it can take a
        // while, and a generic "Scanning…" would look stuck.
        setScanPhase(res.phase);
        if (res.phase === "reading_files") {
          setDocsRead((n) => n + res.docs_read);
        }
        if (res.remaining <= 0) break;
      }
      await qc.invalidateQueries({ queryKey: ["data-room-coverage", roomId] });
    } catch (err) {
      setScanError(err instanceof Error ? err.message : String(err));
    } finally {
      setScanning(false);
    }
  };

  if (coverage.isLoading) {
    return (
      <section className="mt-4 border border-slate-200 rounded-md p-4 text-sm text-slate-500">
        Loading coverage…
      </section>
    );
  }
  if (coverage.error) {
    // 503 with dce_internal_not_configured is expected in local dev
    // without DCE_INTERNAL_URL/SECRET set -- surface plainly, don't
    // pretend coverage is unavailable for some deeper reason.
    return (
      <section className="mt-4 border border-slate-200 rounded-md p-4 text-sm text-red-600">
        Coverage unavailable: {(coverage.error as Error).message}
      </section>
    );
  }
  if (!coverage.data || coverage.data.criteria.length === 0) return null;

  const criteria = coverage.data.criteria;
  const needsScan = criteria.some((c) => c.status === "Scanning");
  const counts = criteria.reduce<Record<string, number>>((acc, c) => {
    const bucket = c.status.startsWith("Found") ? "Found" : c.status;
    acc[bucket] = (acc[bucket] ?? 0) + 1;
    return acc;
  }, {});

  const byCategory = new Map<string, CoverageCriterion[]>();
  for (const c of criteria) {
    const list = byCategory.get(c.category) ?? [];
    list.push(c);
    byCategory.set(c.category, list);
  }

  const toggleCategory = (cat: string) =>
    setCollapsedCats((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });

  return (
    <section className="mt-4 border border-slate-200 rounded-md">
      <div className="px-4 py-2 bg-slate-50 border-b border-slate-200 flex items-center justify-between flex-wrap gap-2">
        <div className="text-sm font-semibold text-slate-700">
          Data room coverage
          <span className="ml-2 text-xs font-normal text-slate-500">
            {counts["Found"] ?? 0} found · {counts["Unconfirmed"] ?? 0} unconfirmed ·{" "}
            {counts["Candidate Gap"] ?? 0} candidate gap
            {(counts["Scanning"] ?? 0) > 0 && ` · ${counts["Scanning"]} not yet scanned`}
          </span>
        </div>
        {needsScan && (
          <button
            type="button"
            onClick={() => void runScan()}
            disabled={scanning}
            className="text-xs px-3 py-1 rounded-md bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50"
          >
            {!scanning
              ? "Scan for coverage"
              : scanPhase === "reading_files"
                ? `Reading documents${docsRead > 0 ? ` (${docsRead})` : ""}…`
                : "Scanning…"}
          </button>
        )}
      </div>

      {scanError && (
        <div className="px-4 py-2 text-xs text-red-600 border-b border-slate-200">
          {scanError}
        </div>
      )}

      {(counts["Candidate Gap"] ?? 0) > 0 && (
        <div className="px-4 py-2 text-xs text-slate-500 border-b border-slate-200 bg-red-50/50">
          Candidate gaps still need human confirmation before referencing
          externally -- an LLM check across every readable document found
          nothing, but that isn't the same as certainty.
        </div>
      )}

      <div className="divide-y divide-slate-200">
        {[...byCategory.entries()].map(([category, items]) => {
          const isCollapsed = collapsedCats.has(category);
          return (
            <div key={category}>
              <button
                type="button"
                onClick={() => toggleCategory(category)}
                className="w-full flex items-center justify-between px-3 py-1.5 bg-slate-50/50 text-xs font-medium text-slate-500 hover:bg-slate-100"
              >
                <span>{category}</span>
                <span>{isCollapsed ? "▸" : "▾"}</span>
              </button>
              {!isCollapsed && (
                <div className="divide-y divide-slate-100">
                  {items.map((c) => (
                    <CriterionRow key={c.criterion_id} roomId={roomId} c={c} />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
