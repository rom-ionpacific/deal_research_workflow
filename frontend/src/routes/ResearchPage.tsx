import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import ChatPanel from "../components/ChatPanel";
import OrgCard from "../components/OrgCard";
import { api, type OrgSearchResult, type Phase } from "../lib/api";

const PHASES: Phase[] = [
  "org_select",
  "entity_select",
  "data_room_setup",
  "data_room_view",
];

export default function ResearchPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const qc = useQueryClient();

  const session = useQuery({
    queryKey: ["session", sessionId],
    queryFn: () => api.getSession(sessionId!),
    enabled: Boolean(sessionId),
  });

  if (session.isLoading)
    return <div className="p-6 text-slate-500">Loading session...</div>;
  if (session.error)
    return (
      <div className="p-6 text-red-600 text-sm">
        {(session.error as Error).message}
      </div>
    );
  if (!session.data) return null;

  const { current_version } = session.data;

  return (
    <div className="h-full grid grid-cols-[1fr_400px]">
      <div className="overflow-y-auto">
        <div className="max-w-3xl mx-auto p-6">
          <PhaseStepper currentPhase={current_version.phase} />
          {current_version.phase === "org_select" && (
            <OrgSelectPhase
              sessionId={sessionId!}
              parentVersionId={current_version.id}
              state={current_version.state}
              onSaved={() =>
                qc.invalidateQueries({ queryKey: ["session", sessionId] })
              }
            />
          )}
          {current_version.phase !== "org_select" && (
            <div className="mt-8 text-slate-500 text-sm">
              Phase <code>{current_version.phase}</code> not built yet — coming
              next.
            </div>
          )}
        </div>
      </div>
      <ChatPanel
        sessionId={sessionId!}
        phase={current_version.phase}
        parentVersionId={current_version.id}
      />
    </div>
  );
}

function PhaseStepper({ currentPhase }: { currentPhase: Phase }) {
  const idx = PHASES.indexOf(currentPhase);
  return (
    <ol className="flex items-center gap-2 text-sm mb-8">
      {PHASES.map((p, i) => (
        <li
          key={p}
          className={
            "px-3 py-1 rounded-full border " +
            (i === idx
              ? "bg-slate-900 text-white border-slate-900"
              : i < idx
              ? "bg-slate-200 text-slate-700"
              : "text-slate-400 border-dashed")
          }
        >
          {i + 1}. {p.replace("_", " ")}
        </li>
      ))}
    </ol>
  );
}

function OrgSelectPhase({
  sessionId,
  parentVersionId,
  state,
  onSaved,
}: {
  sessionId: string;
  parentVersionId: string;
  state: Record<string, unknown>;
  onSaved: () => void;
}) {
  const [q, setQ] = useState("");
  const [debouncedQ, setDebouncedQ] = useState("");
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQ(q.trim()), 300);
    return () => clearTimeout(t);
  }, [q]);

  const selected = (state.selected_org_ids as number[] | undefined) ?? [];

  // Search + selected-by-ids are independent queries. Search is debounced
  // and only fires when the user types; selected-by-ids fires whenever
  // the selection changes (incl. via AI tool calls -> version_created
  // -> ['session', sessionId] invalidates -> selected[] updates here).
  const search = useQuery({
    queryKey: ["orgs", "search", debouncedQ],
    queryFn: () => api.searchOrgs(debouncedQ, 15),
    enabled: debouncedQ.length > 0,
  });

  const selectedQuery = useQuery({
    queryKey: ["orgs", "by-ids", [...selected].sort((a, b) => a - b)],
    queryFn: () => api.getOrgsByIds(selected),
    enabled: selected.length > 0,
    // Selection's enriched data is stable -- the underlying
    // organization_summary table refreshes nightly, so a 5min stale
    // window is plenty.
    staleTime: 5 * 60_000,
  });

  const append = useMutation({
    mutationFn: (newSelected: number[]) =>
      api.appendVersion(sessionId, {
        parent_id: parentVersionId,
        phase: "org_select",
        state: { ...state, selected_org_ids: newSelected, user_query: q },
        summary:
          newSelected.length > selected.length
            ? `Added org`
            : `Removed org`,
      }),
    onSuccess: onSaved,
  });

  const toggle = (org_id: number) => {
    const next = selected.includes(org_id)
      ? selected.filter((x) => x !== org_id)
      : [...selected, org_id];
    append.mutate(next);
  };

  // Hide already-selected orgs from search results so the list isn't
  // duplicated -- they're already shown above in the sticky panel.
  const searchVisible = (search.data ?? []).filter(
    (r) => !selected.includes(r.org_id)
  );

  return (
    <div>
      <h2 className="text-lg font-semibold mb-1">
        Phase 1 — Select organizations
      </h2>
      <p className="text-sm text-slate-500 mb-4">
        Search by name and click candidates to add or remove. Selected orgs
        stay pinned at the top regardless of search query. Each click is
        a new session version (deep-link friendly, undoable).
      </p>

      {selected.length > 0 && (
        <div className="mb-6">
          <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">
            Selected ({selected.length})
          </div>
          {selectedQuery.isLoading && (
            <div className="text-xs text-slate-500">Loading details...</div>
          )}
          <div className="space-y-2">
            {(selectedQuery.data ?? []).map((r: OrgSearchResult) => (
              <OrgCard
                key={r.org_id}
                org={r}
                selected={true}
                onToggle={() => toggle(r.org_id)}
                disabled={append.isPending}
              />
            ))}
          </div>
        </div>
      )}

      <input
        type="text"
        autoFocus
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Type a company name..."
        className="w-full border border-slate-300 rounded-md px-3 py-2 mb-4"
      />

      {search.isLoading && (
        <div className="text-slate-500 text-sm">Searching...</div>
      )}
      {search.error && (
        <div className="text-red-600 text-sm">
          {(search.error as Error).message}
        </div>
      )}
      {debouncedQ.length > 0 && !search.isLoading &&
        searchVisible.length === 0 && (search.data?.length ?? 0) === 0 && (
        <div className="text-slate-500 text-sm">No matches.</div>
      )}

      <div className="space-y-2">
        {searchVisible.map((r: OrgSearchResult) => (
          <OrgCard
            key={r.org_id}
            org={r}
            selected={false}
            onToggle={() => toggle(r.org_id)}
            disabled={append.isPending}
          />
        ))}
      </div>
    </div>
  );
}
