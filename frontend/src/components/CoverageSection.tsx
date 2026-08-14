import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, type CoverageCriterion } from "../lib/api";

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
 *     design's human-review-gate principle. This UI does not (yet) have
 *     a confirm/dismiss action; it surfaces gaps for a human to triage.
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

function CriterionRow({ c }: { c: CoverageCriterion }) {
  const [expanded, setExpanded] = useState(false);
  const hasDetail = c.hits.length > 0 || c.keyword_hits.length > 0;

  return (
    <div className="px-3 py-2">
      <button
        type="button"
        onClick={() => hasDetail && setExpanded((v) => !v)}
        className={
          "w-full flex items-center justify-between gap-2 text-left " +
          (hasDetail ? "cursor-pointer" : "cursor-default")
        }
      >
        <span className="text-sm text-slate-700 flex-1">
          {c.criterion}
          {c.importance === "core" && (
            <span className="ml-1.5 text-[10px] uppercase tracking-wide text-slate-400">
              core
            </span>
          )}
        </span>
        <StatusBadge status={c.status} />
      </button>

      {expanded && hasDetail && (
        <div className="mt-2 ml-1 pl-2 border-l-2 border-slate-200 space-y-1.5">
          {c.hits.map((h, i) => (
            <div key={i} className="text-xs">
              <span className="font-medium text-slate-600">{h.doc_name}</span>
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
    </div>
  );
}

export default function CoverageSection({ roomId }: { roomId: number }) {
  const qc = useQueryClient();
  const [scanning, setScanning] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const [collapsedCats, setCollapsedCats] = useState<Set<string>>(new Set());

  const coverage = useQuery({
    queryKey: ["data-room-coverage", roomId],
    queryFn: () => api.getDataRoomCoverage(roomId),
  });

  const runScan = async () => {
    setScanning(true);
    setScanError(null);
    try {
      // Drain in a polled loop -- each call is one bounded, real batch on
      // dce (~25 docs, real Gemini calls, a few seconds each). Deliberately
      // sequential (await, not fire-and-forget) so we never have two scans
      // racing the same room, and the visible "Scanning…" state stays
      // accurate to what's actually happening.
      for (;;) {
        const res = await api.scanDataRoomCoverageBatch(roomId);
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
            {scanning ? "Scanning…" : "Scan for coverage"}
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
                    <CriterionRow key={c.criterion_id} c={c} />
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
