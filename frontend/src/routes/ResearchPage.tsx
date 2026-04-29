import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

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
    <div className="max-w-5xl mx-auto p-6">
      <PhaseStepper currentPhase={current_version.phase} />
      {current_version.phase === "org_select" && (
        <OrgSelectPhase
          sessionId={sessionId!}
          parentVersionId={current_version.id}
          state={current_version.state}
          onSaved={() => qc.invalidateQueries({ queryKey: ["session", sessionId] })}
        />
      )}
      {current_version.phase !== "org_select" && (
        <div className="mt-8 text-slate-500 text-sm">
          Phase <code>{current_version.phase}</code> not built yet — coming next.
        </div>
      )}
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

  const search = useQuery({
    queryKey: ["orgs", "search", debouncedQ],
    queryFn: () => api.searchOrgs(debouncedQ, 15),
    enabled: debouncedQ.length > 0,
  });

  const selected = (state.selected_org_ids as number[] | undefined) ?? [];

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

  return (
    <div>
      <h2 className="text-lg font-semibold mb-1">
        Phase 1 — Select organizations
      </h2>
      <p className="text-sm text-slate-500 mb-4">
        Search by name; click candidates to add or remove from the selection.
        Each click creates a new session version (deep-link friendly, undoable).
      </p>

      <input
        type="text"
        autoFocus
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Type a company name..."
        className="w-full border border-slate-300 rounded-md px-3 py-2 mb-4"
      />

      {selected.length > 0 && (
        <div className="mb-4">
          <div className="text-xs uppercase tracking-wide text-slate-500 mb-1">
            Selected ({selected.length})
          </div>
          <div className="flex flex-wrap gap-2">
            {selected.map((id) => (
              <span
                key={id}
                className="px-2 py-1 rounded-full bg-slate-900 text-white text-xs"
              >
                org_id={id}
              </span>
            ))}
          </div>
        </div>
      )}

      {search.isLoading && (
        <div className="text-slate-500 text-sm">Searching...</div>
      )}
      {search.error && (
        <div className="text-red-600 text-sm">
          {(search.error as Error).message}
        </div>
      )}

      <ul className="divide-y border rounded-md">
        {search.data?.map((r: OrgSearchResult) => {
          const isSel = selected.includes(r.org_id);
          return (
            <li key={r.org_id}>
              <button
                onClick={() => toggle(r.org_id)}
                disabled={append.isPending}
                className={
                  "w-full text-left px-4 py-3 hover:bg-slate-50 flex items-center gap-3 " +
                  (isSel ? "bg-slate-100" : "")
                }
              >
                <input type="checkbox" readOnly checked={isSel} />
                <div className="flex-1">
                  <div className="font-medium">{r.name}</div>
                  <div className="text-xs text-slate-500">
                    {r.why_match} · score {r.score.toFixed(2)}
                  </div>
                </div>
                <code className="text-xs text-slate-400">#{r.org_id}</code>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
