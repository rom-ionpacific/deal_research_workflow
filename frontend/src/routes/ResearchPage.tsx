import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";

import ChatPanel from "../components/ChatPanel";
import DataRoomSetupPhase from "../components/DataRoomSetupPhase";
import EntitySelectPhase from "../components/EntitySelectPhase";
import OrgCard from "../components/OrgCard";
import {
  api,
  type OrgSearchResult,
  type Phase,
  type SessionWithCurrent,
} from "../lib/api";
import { useChat } from "../stores/chat";

const PHASES: Phase[] = [
  "org_select",
  "entity_select",
  "data_room_setup",
  "data_room_view",
];

export default function ResearchPage() {
  const { sessionId } = useParams<{ sessionId: string }>();

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
      {/* min-h-0 on each grid cell -- items default to
          min-height: auto, which lets them expand to fit content
          and breaks the columns' inner scrolling. */}
      <div className="min-h-0 overflow-y-auto">
        <div className="max-w-3xl mx-auto p-6">
          <PhaseStepper currentPhase={current_version.phase} />
          {current_version.phase === "org_select" && (
            <OrgSelectPhase
              sessionId={sessionId!}
              parentVersionId={current_version.id}
              state={current_version.state}
            />
          )}
          {current_version.phase === "entity_select" && (
            <EntitySelectPhase
              sessionId={sessionId!}
              parentVersionId={current_version.id}
              state={current_version.state}
            />
          )}
          {current_version.phase === "data_room_setup" && (
            <DataRoomSetupPhase
              sessionId={sessionId!}
              parentVersionId={current_version.id}
              state={current_version.state}
            />
          )}
          {current_version.phase === "data_room_view" && (
            <DataRoomViewPlaceholder
              dataRoomId={
                (current_version.state as { data_room_id?: number })
                  .data_room_id ?? null
              }
            />
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

function DataRoomViewPlaceholder({
  dataRoomId,
}: {
  dataRoomId: number | null;
}) {
  // Phase 4 (data_room_view) UI isn't built yet. The build is async --
  // the data-room-builder cron in deal_cloud_enhancer picks it up
  // every ~2min, uploads entities to ToltIQ, runs the playlist, saves
  // answers. For now we just confirm the room is queued + give the
  // org-history-viewer link where the answers will appear once ready.
  return (
    <div className="mt-8 max-w-prose">
      <h2 className="text-lg font-semibold mb-2">
        Phase 4 — Data room view (placeholder)
      </h2>
      <p className="text-sm text-slate-600 mb-3">
        Data room {dataRoomId !== null ? `#${dataRoomId} ` : ""}is queued.
        The builder cron will upload your selected entities to ToltIQ,
        run the question playlist, and save answers — typically 10–15
        minutes end to end.
      </p>
      <p className="text-sm text-slate-600">
        Phase 4 (in-app Q&A view with citations back to entities) isn't
        built yet. In the meantime answers land in
        <code className="text-xs bg-slate-100 px-1 mx-1 rounded">
          dealcloud.historical_data_room_answer
        </code>
        and surface in the org-history-viewer's "AI Overview" tab.
      </p>
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
}: {
  sessionId: string;
  parentVersionId: string;
  state: Record<string, unknown>;
}) {
  const qc = useQueryClient();

  const [q, setQ] = useState("");
  const [debouncedQ, setDebouncedQ] = useState("");
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQ(q.trim()), 300);
    return () => clearTimeout(t);
  }, [q]);

  const selected = (state.selected_org_ids as number[] | undefined) ?? [];

  const search = useQuery({
    queryKey: ["orgs", "search", debouncedQ],
    queryFn: () => api.searchOrgs(debouncedQ, 15),
    enabled: debouncedQ.length > 0,
  });

  const selectedQuery = useQuery({
    queryKey: ["orgs", "by-ids", [...selected].sort((a, b) => a - b)],
    queryFn: () => api.getOrgsByIds(selected),
    enabled: selected.length > 0,
    staleTime: 5 * 60_000,
  });

  // Per-card pending visual: only the org_id currently being toggled
  // shows the disabled state, so other cards stay clickable.
  const [pendingIds, setPendingIds] = useState<Set<number>>(new Set());

  // Promise chain: serialise POSTs so each one's parent_id references
  // the version_id committed by the previous one. Without this, two
  // quick clicks both send parent_id = original version, and the
  // second 409s on the server's optimistic-concurrency check.
  const queueRef = useRef<Promise<void>>(Promise.resolve());

  const toggle = (org_id: number) => {
    const cached = qc.getQueryData<SessionWithCurrent>(["session", sessionId]);
    if (!cached) return;

    const cur =
      (cached.current_version.state.selected_org_ids as
        | number[]
        | undefined) ?? [];
    const isSelected = cur.includes(org_id);
    const next = isSelected
      ? cur.filter((x) => x !== org_id)
      : [...cur, org_id];

    // (1) Optimistic session patch -- UI updates this frame, no wait.
    qc.setQueryData<SessionWithCurrent>(["session", sessionId], {
      ...cached,
      current_version: {
        ...cached.current_version,
        state: {
          ...cached.current_version.state,
          selected_org_ids: next,
          user_query: q,
        },
      },
    });

    // (3) Pre-warm the by-ids cache for the new selected key so the
    // sticky panel doesn't fetch when `selected` changes. We have the
    // toggled org's enriched data either in the search results (when
    // adding) or already in the selected query data (when removing).
    if (next.length > 0) {
      const sortedNext = [...next].sort((a, b) => a - b);
      const sortedCur = [...cur].sort((a, b) => a - b);
      const curData =
        qc.getQueryData<OrgSearchResult[]>([
          "orgs",
          "by-ids",
          sortedCur,
        ]) ?? [];
      let nextData: OrgSearchResult[];
      if (isSelected) {
        nextData = curData.filter((r) => r.org_id !== org_id);
      } else {
        const fromSearch = search.data?.find((r) => r.org_id === org_id);
        nextData = fromSearch ? [...curData, fromSearch] : curData;
      }
      qc.setQueryData(["orgs", "by-ids", sortedNext], nextData);
    }

    // (4) Per-card visual pending.
    setPendingIds((prev) => {
      const n = new Set(prev);
      n.add(org_id);
      return n;
    });

    // Serialised POST: chain behind any prior in-flight click. The
    // .catch swallows prior errors so one bad POST doesn't sink the
    // entire queue for the session.
    queueRef.current = queueRef.current
      .catch(() => undefined)
      .then(async () => {
        // Read the LATEST committed version_id from cache. If a prior
        // POST in the queue just landed, this picks up its new id and
        // chains correctly.
        const latest = qc.getQueryData<SessionWithCurrent>([
          "session",
          sessionId,
        ]);
        const parentId = latest?.current_version.id ?? parentVersionId;

        try {
          // (2) Use the POST response directly to update cache. We
          // preserve the cache's `state` if it differs from the server's
          // (i.e. additional optimistic patches landed while this POST
          // was in flight) -- the next chained POST will commit them.
          const data = await api.appendVersion(sessionId, {
            parent_id: parentId,
            phase: "org_select",
            state: { ...state, selected_org_ids: next, user_query: q },
            summary: isSelected ? "Removed org" : "Added org",
          });
          qc.setQueryData<SessionWithCurrent | undefined>(
            ["session", sessionId],
            (old) => {
              if (!old) {
                return { session: data.session, current_version: data.version };
              }
              const serverIds =
                (data.version.state.selected_org_ids as
                  | number[]
                  | undefined) ?? [];
              const cacheIds =
                (old.current_version.state.selected_org_ids as
                  | number[]
                  | undefined) ?? [];
              const hasLaterPatches =
                JSON.stringify(serverIds) !== JSON.stringify(cacheIds);
              if (hasLaterPatches) {
                // Keep optimistic state; only update the version_id so
                // the next chained POST has the right parent_id.
                return {
                  session: data.session,
                  current_version: {
                    ...data.version,
                    state: old.current_version.state,
                  },
                };
              }
              return { session: data.session, current_version: data.version };
            },
          );
        } catch (err) {
          console.error("toggle org_id=" + org_id + " failed:", err);
          // Recover by refetching authoritative state. Optimistic
          // patches that hadn't been committed yet are discarded;
          // user can retry.
          qc.invalidateQueries({ queryKey: ["session", sessionId] });
        } finally {
          setPendingIds((prev) => {
            const n = new Set(prev);
            n.delete(org_id);
            return n;
          });
        }
      });
  };

  // Hide already-selected orgs from search results so the list isn't
  // duplicated -- they're already shown above in the sticky panel.
  const searchVisible = (search.data ?? []).filter(
    (r) => !selected.includes(r.org_id),
  );

  // Publish a per-turn UI snapshot to the chat store so the orchestrator
  // can answer "out of these, pick the financial institutions"-style
  // questions without re-running search. The displayed list mirrors what
  // the user actually sees: selected first (sticky panel), then the
  // un-selected search results in display order. Trim to compact rows so
  // we don't blow the prompt up.
  const setUIContext = useChat((s) => s.setUIContext);
  useEffect(() => {
    const compact = (r: OrgSearchResult) => ({
      org_id: r.org_id,
      name: r.name,
      why_match: r.why_match,
      score: r.score,
      document_count: r.document_count,
      communication_count: r.communication_count,
    });
    const selectedDisplay = (selectedQuery.data ?? []).map(compact);
    const candidatesDisplay = [
      ...selectedDisplay,
      ...searchVisible.map(compact),
    ];
    setUIContext(sessionId, {
      phase: "org_select",
      search_query: debouncedQ,
      displayed_candidates: candidatesDisplay,
      selected_orgs: selectedDisplay,
    });
    // We intentionally don't clear on unmount -- when the page navigates
    // to entity_select the next phase will overwrite (or null) it.
  }, [
    sessionId,
    setUIContext,
    debouncedQ,
    selectedQuery.data,
    // search.data is referenced via searchVisible but we depend on it
    // directly so the effect re-runs when the underlying query updates.
    search.data,
    // selected list change -> searchVisible recomputes -> we need a dep.
    selected.length,
  ]);

  return (
    <div>
      <h2 className="text-lg font-semibold mb-1">
        Phase 1 — Select organizations
      </h2>
      <p className="text-sm text-slate-500 mb-3">
        Search by name and click candidates to add or remove. Selected orgs
        sit at the top of the list and scroll with it. Each click is a new
        session version (deep-link friendly, undoable).
      </p>

      {/* Search box stays sticky at the top of the scrolling parent
          (overflow-y-auto in ResearchPage) so it's always reachable. */}
      <div className="sticky top-0 bg-white pb-2 z-20 -mx-1 px-1">
        <input
          type="text"
          autoFocus
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Type a company name..."
          className="w-full border border-slate-300 rounded-md px-3 py-2"
        />
      </div>

      {/* Selected section + search results in one scroll flow. Selected
          orgs are visually grouped at the top via the header, but they
          scroll with everything else so a long selection doesn't
          fill the viewport and hide search results. */}
      <div className="space-y-2 mt-2">
        {selected.length > 0 && (
          <div className="pt-1 pb-3 border-b border-slate-200 mb-2">
            <div className="text-[10px] uppercase tracking-wide text-slate-500 mb-1">
              Selected ({selected.length})
            </div>
            {selectedQuery.isLoading && (
              <div className="text-xs text-slate-500">Loading details...</div>
            )}
            <div className="space-y-1.5">
              {(selectedQuery.data ?? []).map((r: OrgSearchResult) => (
                <OrgCard
                  key={r.org_id}
                  org={r}
                  selected={true}
                  onToggle={() => toggle(r.org_id)}
                  disabled={pendingIds.has(r.org_id)}
                />
              ))}
            </div>
          </div>
        )}

        {search.isLoading && (
          <div className="text-slate-500 text-sm pt-1">Searching...</div>
        )}
        {search.error && (
          <div className="text-red-600 text-sm pt-1">
            {(search.error as Error).message}
          </div>
        )}
        {debouncedQ.length > 0 &&
          !search.isLoading &&
          searchVisible.length === 0 &&
          (search.data?.length ?? 0) === 0 && (
            <div className="text-slate-500 text-sm pt-1">No matches.</div>
          )}

        {searchVisible.map((r: OrgSearchResult) => (
          <OrgCard
            key={r.org_id}
            org={r}
            selected={false}
            onToggle={() => toggle(r.org_id)}
            disabled={pendingIds.has(r.org_id)}
          />
        ))}
      </div>

      {/* Phase nav. Advance disabled until at least one org selected;
          backend's advance_to_entity_select tool also refuses an empty
          selection, so this matches. No "back" button -- Phase 1 is
          the first phase. */}
      <div className="flex items-center justify-end mt-6 pt-4 border-t border-slate-200">
        <button
          type="button"
          disabled={selected.length === 0}
          onClick={async () => {
            try {
              const cached = qc.getQueryData<SessionWithCurrent>([
                "session",
                sessionId,
              ]);
              const parentId =
                cached?.current_version.id ?? parentVersionId;
              const data = await api.appendVersion(sessionId, {
                parent_id: parentId,
                phase: "entity_select",
                state: {
                  inherits_from_version: parentId,
                  selected_org_ids: selected,
                  selected_entity_ids: {
                    document: [],
                    email_thread: [],
                    calendar_event: [],
                    slack_message_group: [],
                  },
                },
                summary: "Advance to entity_select",
              });
              qc.setQueryData(["session", sessionId], {
                session: data.session,
                current_version: data.version,
              });
            } catch (err) {
              console.error("advance failed", err);
              qc.invalidateQueries({ queryKey: ["session", sessionId] });
            }
          }}
          className="px-3 py-2 bg-slate-900 text-white text-sm rounded-md disabled:opacity-40"
        >
          Advance to entity_select →
        </button>
      </div>
    </div>
  );
}
