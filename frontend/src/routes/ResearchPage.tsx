import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";

import ChatPanel from "../components/ChatPanel";
import DataRoomSetupPhase from "../components/DataRoomSetupPhase";
import DataRoomViewPhase from "../components/DataRoomViewPhase";
import EntitySelectPhase from "../components/EntitySelectPhase";
import OrgCard from "../components/OrgCard";
import {
  api,
  type OrgSearchResult,
  type Phase,
  type SearchMode,
  type Session,
  type SessionWithCurrent,
} from "../lib/api";
import { useChat } from "../stores/chat";

// The user-facing step list. Each step maps to one or two underlying
// phases; data_room_setup + data_room_view both surface under
// "historical_data_room" because the user shouldn't see them as
// separate workflow stages (the underlying phase swap is just whether
// the room has been built yet).
type Step = "org_select" | "entity_select" | "historical_data_room";

const STEPS: Step[] = ["org_select", "entity_select", "historical_data_room"];

const STEP_LABEL: Record<Step, string> = {
  org_select: "Org select",
  entity_select: "Entity select",
  historical_data_room: "Historical data room",
};

function phaseToStep(phase: Phase): Step {
  if (phase === "org_select" || phase === "entity_select") return phase;
  return "historical_data_room";
}

export default function ResearchPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  // Optional ?only=toltiq|claude lets reviewers see the room as if it
  // had been built single-provider, even when room.provider is 'both'.
  // Filters preset answers, hides the other Ask button + add-provider
  // banner, and pins the AI assistant to that provider. Pure
  // presentation-layer override -- the data is unchanged.
  const [searchParams] = useSearchParams();
  const onlyParam = searchParams.get("only");
  const forcedProvider: "toltiq" | "claude" | null =
    onlyParam === "toltiq" || onlyParam === "claude" ? onlyParam : null;

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
    <ResizableSplit>
      {/* min-h-0 on each grid cell -- items default to
          min-height: auto, which lets them expand to fit content
          and breaks the columns' inner scrolling. */}
      <div className="min-h-0 overflow-y-auto">
        <div className="max-w-3xl mx-auto p-6">
          <SessionTitleBar session={session.data.session} />
          <PhaseStepper
            sessionId={sessionId!}
            currentPhase={current_version.phase}
            currentVersionId={current_version.id}
            state={current_version.state}
          />
          <TopPhaseNav
            sessionId={sessionId!}
            currentPhase={current_version.phase}
            currentVersionId={current_version.id}
            state={current_version.state}
          />
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
            <DataRoomViewPhase
              sessionId={sessionId!}
              state={current_version.state}
              forcedProvider={forcedProvider}
            />
          )}
        </div>
      </div>
      <ChatPanel
        sessionId={sessionId!}
        phase={current_version.phase}
        parentVersionId={current_version.id}
        forcedProvider={forcedProvider}
      />
    </ResizableSplit>
  );
}

const CHAT_WIDTH_STORAGE_KEY = "research:chatWidth";
const CHAT_WIDTH_DEFAULT = 400;
const CHAT_WIDTH_MIN = 280;
const MAIN_WIDTH_MIN = 320;

/** Two-column split with a draggable gutter that resizes the right
 * (chat) pane. Width is persisted to localStorage so it survives
 * navigation and reload. Min/max are clamped so neither pane collapses
 * past usable; on window resize we re-clamp so the chat can never
 * exceed `window.innerWidth - MAIN_WIDTH_MIN`. */
function ResizableSplit({ children }: { children: [React.ReactNode, React.ReactNode] }) {
  const [main, chat] = children;
  const [chatWidth, setChatWidth] = useState<number>(() => {
    if (typeof window === "undefined") return CHAT_WIDTH_DEFAULT;
    const raw = window.localStorage.getItem(CHAT_WIDTH_STORAGE_KEY);
    const parsed = raw == null ? NaN : Number(raw);
    return Number.isFinite(parsed) && parsed >= CHAT_WIDTH_MIN
      ? parsed
      : CHAT_WIDTH_DEFAULT;
  });

  // Re-clamp on viewport resize so a chat that was wide on a big screen
  // doesn't squeeze the main content off-screen when the window shrinks.
  useEffect(() => {
    const handle = () => {
      setChatWidth((w) => {
        const max = window.innerWidth - MAIN_WIDTH_MIN;
        return Math.max(CHAT_WIDTH_MIN, Math.min(max, w));
      });
    };
    handle();
    window.addEventListener("resize", handle);
    return () => window.removeEventListener("resize", handle);
  }, []);

  useEffect(() => {
    window.localStorage.setItem(CHAT_WIDTH_STORAGE_KEY, String(chatWidth));
  }, [chatWidth]);

  const draggingRef = useRef(false);

  const startDrag = (e: React.MouseEvent) => {
    e.preventDefault();
    draggingRef.current = true;
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";

    const onMove = (ev: MouseEvent) => {
      if (!draggingRef.current) return;
      const max = window.innerWidth - MAIN_WIDTH_MIN;
      const next = Math.max(
        CHAT_WIDTH_MIN,
        Math.min(max, window.innerWidth - ev.clientX),
      );
      setChatWidth(next);
    };
    const onUp = () => {
      draggingRef.current = false;
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  // Double-click resets to default — quick escape hatch from an
  // accidentally-tiny or accidentally-huge chat pane.
  const onDoubleClick = () => setChatWidth(CHAT_WIDTH_DEFAULT);

  return (
    <div
      className="h-full grid"
      style={{ gridTemplateColumns: `1fr 5px ${chatWidth}px` }}
    >
      {main}
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize AI assistant panel"
        onMouseDown={startDrag}
        onDoubleClick={onDoubleClick}
        title="Drag to resize · double-click to reset"
        className="cursor-col-resize bg-slate-200 hover:bg-slate-400 active:bg-slate-500 transition-colors"
      />
      {chat}
    </div>
  );
}

function SessionTitleBar({ session }: { session: Session }) {
  const qc = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(session.title ?? "");
  const [saving, setSaving] = useState(false);

  // When the session title changes server-side (auto-rename on first
  // org selection, or another tab edited it), keep the un-editing
  // input in sync.
  useEffect(() => {
    if (!editing) setDraft(session.title ?? "");
  }, [session.title, editing]);

  const patchAndCache = (
    patch: { title?: string; is_starred?: boolean },
    optimistic: Partial<Session>,
  ) => {
    // Optimistic update on both the canonical session-with-version
    // cache (drives this page) and the sessions list cache (drives
    // the home page).
    qc.setQueryData<SessionWithCurrent>(["session", session.id], (old) =>
      old
        ? { ...old, session: { ...old.session, ...optimistic } }
        : old,
    );
    qc.setQueryData<Session[]>(["sessions"], (old) =>
      old?.map((s) =>
        s.id === session.id ? { ...s, ...optimistic } : s,
      ),
    );
    return api.updateSession(session.id, patch).then((updated) => {
      qc.setQueryData<SessionWithCurrent>(["session", session.id], (old) =>
        old ? { ...old, session: updated } : old,
      );
      qc.setQueryData<Session[]>(["sessions"], (old) =>
        old?.map((s) => (s.id === session.id ? updated : s)),
      );
    });
  };

  const onToggleStar = () => {
    const next = !session.is_starred;
    void patchAndCache({ is_starred: next }, { is_starred: next });
  };

  const onSaveTitle = async () => {
    const trimmed = draft.trim();
    if (!trimmed || trimmed === (session.title ?? "")) {
      setEditing(false);
      setDraft(session.title ?? "");
      return;
    }
    setSaving(true);
    try {
      await patchAndCache(
        { title: trimmed },
        { title: trimmed, title_is_locked: true },
      );
      setEditing(false);
    } catch (err) {
      console.error("rename failed", err);
      // Roll back the input to the canonical title; the cache rollback
      // happens via invalidate below.
      qc.invalidateQueries({ queryKey: ["session", session.id] });
      qc.invalidateQueries({ queryKey: ["sessions"] });
      setDraft(session.title ?? "");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex items-center gap-2 mb-3">
      <StarButton
        starred={session.is_starred}
        onClick={onToggleStar}
        ariaLabel={session.is_starred ? "Unstar session" : "Star session"}
      />
      {editing ? (
        <input
          type="text"
          autoFocus
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void onSaveTitle();
            if (e.key === "Escape") {
              setEditing(false);
              setDraft(session.title ?? "");
            }
          }}
          onBlur={() => void onSaveTitle()}
          disabled={saving}
          maxLength={200}
          className="flex-1 text-lg font-semibold border border-slate-300 rounded-md px-2 py-1"
        />
      ) : (
        // Inner flex so the pen sits right after the title text, not at
        // the far edge of the row. min-w-0 lets the title truncate while
        // keeping the pen visible.
        <div className="flex items-center gap-1.5 min-w-0">
          <h1 className="text-lg font-semibold truncate">
            {session.title ?? "Untitled session"}
          </h1>
          <button
            type="button"
            onClick={() => setEditing(true)}
            className="p-1 rounded text-slate-400 hover:text-slate-700 hover:bg-slate-100 shrink-0"
            aria-label="Rename session"
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              {/* pencil glyph (lucide-style) */}
              <path d="M12 20h9" />
              <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
            </svg>
          </button>
        </div>
      )}
    </div>
  );
}

function StarButton({
  starred,
  onClick,
  ariaLabel,
}: {
  starred: boolean;
  onClick: () => void;
  ariaLabel: string;
}) {
  // Inline SVG so we don't pull in an icon dep. Two states:
  //  - starred=false: outline grey star
  //  - starred=true:  filled gold star
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={ariaLabel}
      className="p-1 rounded hover:bg-slate-100 transition-colors"
    >
      <svg
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill={starred ? "#facc15" : "none"}
        stroke={starred ? "#ca8a04" : "#94a3b8"}
        strokeWidth="2"
        strokeLinejoin="round"
      >
        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
      </svg>
    </button>
  );
}

function SearchModeToggle({
  mode,
  onChange,
}: {
  mode: SearchMode;
  onChange: (m: SearchMode) => void;
}) {
  // Segmented control. Trigram = name/alias text match; Hybrid = both
  // legs merged via RRF; Semantic = embedding cosine only. Help text
  // under the buttons calls out which is doing what so the user
  // doesn't have to remember.
  const modes: Array<{ k: SearchMode; label: string; hint: string }> = [
    { k: "trigram", label: "Name", hint: "Exact / fuzzy name match." },
    { k: "hybrid", label: "Hybrid", hint: "Name + meaning, merged." },
    { k: "semantic", label: "Meaning", hint: "Embedding similarity only." },
  ];
  return (
    <div className="flex items-center gap-2 mt-1 text-xs text-slate-500">
      <span className="text-slate-400">Match:</span>
      <div className="inline-flex rounded-md border border-slate-300 overflow-hidden">
        {modes.map((m, i) => (
          <button
            key={m.k}
            type="button"
            onClick={() => onChange(m.k)}
            className={
              "px-2 py-1 transition-colors " +
              (mode === m.k
                ? "bg-slate-900 text-white"
                : "bg-white hover:bg-slate-50 text-slate-700") +
              (i > 0 ? " border-l border-slate-300" : "")
            }
          >
            {m.label}
          </button>
        ))}
      </div>
      <span className="truncate">
        {modes.find((m) => m.k === mode)?.hint}
      </span>
    </div>
  );
}


function SelectAllMatchesHeader({
  matchIds,
  selectedIds,
  disabled,
  onToggle,
}: {
  matchIds: number[];
  selectedIds: number[];
  disabled?: boolean;
  onToggle: () => void;
}) {
  const ref = useRef<HTMLInputElement | null>(null);
  const selectedSet = new Set(selectedIds);
  const onPage = matchIds.filter((id) => selectedSet.has(id)).length;
  const allChecked = onPage === matchIds.length && matchIds.length > 0;
  const someChecked = onPage > 0 && !allChecked;

  useEffect(() => {
    if (ref.current) ref.current.indeterminate = someChecked;
  }, [someChecked]);

  const label = allChecked
    ? `Deselect all ${matchIds.length} matching results`
    : someChecked
    ? `Select remaining ${matchIds.length - onPage} matching results`
    : `Select all ${matchIds.length} matching results`;

  return (
    <div className="flex items-center gap-2 px-3 py-2 mb-1 bg-slate-50 border border-slate-200 rounded-md">
      <input
        ref={ref}
        type="checkbox"
        checked={allChecked}
        onChange={onToggle}
        disabled={disabled}
        className="cursor-pointer disabled:opacity-50"
      />
      <button
        type="button"
        onClick={onToggle}
        disabled={disabled}
        className="text-xs text-slate-700 hover:underline disabled:opacity-50 disabled:no-underline"
      >
        {label}
      </button>
    </div>
  );
}

function PhaseStepper({
  sessionId,
  currentPhase,
  currentVersionId,
  state,
}: {
  sessionId: string;
  currentPhase: Phase;
  currentVersionId: string;
  state: Record<string, unknown>;
}) {
  const qc = useQueryClient();
  const currentStep = phaseToStep(currentPhase);
  const idx = STEPS.indexOf(currentStep);

  const onTokenClick = (target: Step, i: number) => {
    // Only allow back-navigation via tokens. Forward needs the
    // explicit Advance/Build buttons in the phase content so the user
    // can't skip required selections.
    if (i >= idx) return;
    void navigateToStep({
      qc,
      sessionId,
      parentVersionId: currentVersionId,
      currentState: state,
      target,
    });
  };

  return (
    <ol className="flex items-center gap-1 text-sm mb-4">
      {STEPS.map((s, i) => {
        const isActive = i === idx;
        const isPast = i < idx;
        return (
          <li key={s} className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => onTokenClick(s, i)}
              disabled={!isPast}
              aria-current={isActive ? "step" : undefined}
              className={
                "px-3 py-1 rounded-full border transition-colors " +
                (isActive
                  ? "bg-slate-900 text-white border-slate-900"
                  : isPast
                    ? "bg-slate-200 text-slate-700 hover:bg-slate-300 cursor-pointer"
                    : "text-slate-400 border-dashed cursor-default")
              }
            >
              {i + 1}. {STEP_LABEL[s]}
            </button>
            {i < STEPS.length - 1 && (
              <span
                className={
                  "text-slate-400 select-none " +
                  (i < idx ? "text-slate-500" : "text-slate-300")
                }
                aria-hidden
              >
                →
              </span>
            )}
          </li>
        );
      })}
    </ol>
  );
}

function TopPhaseNav({
  sessionId,
  currentPhase,
  currentVersionId,
  state,
}: {
  sessionId: string;
  currentPhase: Phase;
  currentVersionId: string;
  state: Record<string, unknown>;
}) {
  const qc = useQueryClient();
  const back = navTargets(currentPhase).back;
  const advance = navTargets(currentPhase).advance;
  const ps = state as Record<string, unknown>;

  // Forward gating: only allow Advance when the user has the
  // prerequisites for the next phase (mirrors the bottom-nav guards
  // each phase component implements internally).
  const orgIds = (ps.selected_org_ids as number[] | undefined) ?? [];
  const entityMap = (ps.selected_entity_ids as
    | Partial<Record<string, number[]>>
    | undefined) ?? {};
  const totalEntities =
    (entityMap.document?.length ?? 0) +
    (entityMap.email_thread?.length ?? 0) +
    (entityMap.calendar_event?.length ?? 0) +
    (entityMap.slack_message_group?.length ?? 0);

  let advanceDisabled = false;
  if (advance) {
    if (currentPhase === "org_select" && orgIds.length === 0) advanceDisabled = true;
    if (currentPhase === "entity_select" && totalEntities === 0)
      advanceDisabled = true;
  }

  const go = (target: Step) =>
    navigateToStep({
      qc,
      sessionId,
      parentVersionId: currentVersionId,
      currentState: state,
      target,
    });

  // Render nothing for terminal phases with no nav (none today; both
  // ends have either a back or an advance).
  if (!back && !advance) return null;

  return (
    <div className="flex items-center justify-between mb-6 text-sm">
      <div>
        {back && (
          <button
            type="button"
            onClick={() => void go(back)}
            className="px-3 py-1.5 border border-slate-300 text-slate-700 rounded-md hover:bg-slate-50"
          >
            ← Back to {STEP_LABEL[back]}
          </button>
        )}
      </div>
      <div>
        {advance && (
          <button
            type="button"
            onClick={() => void go(advance)}
            disabled={advanceDisabled}
            className="px-3 py-1.5 bg-slate-900 text-white rounded-md disabled:opacity-40"
          >
            Advance to {STEP_LABEL[advance]} →
          </button>
        )}
      </div>
    </div>
  );
}

function navTargets(phase: Phase): { back: Step | null; advance: Step | null } {
  switch (phase) {
    case "org_select":
      return { back: null, advance: "entity_select" };
    case "entity_select":
      return { back: "org_select", advance: "historical_data_room" };
    case "data_room_setup":
      // Build button (centered below presets) replaces the top-nav
      // Advance for this phase -- nothing on the right.
      return { back: "entity_select", advance: null };
    case "data_room_view":
      // Per user: Back from view goes to entity_select (not setup).
      // The data room stays built; this just pivots the session state.
      return { back: "entity_select", advance: null };
  }
}

async function navigateToStep({
  qc,
  sessionId,
  parentVersionId,
  currentState,
  target,
}: {
  qc: ReturnType<typeof useQueryClient>;
  sessionId: string;
  parentVersionId: string;
  currentState: Record<string, unknown>;
  target: Step;
}): Promise<void> {
  // Build the next-phase state from what's available in the current
  // version. Each transition preserves what makes sense for the target
  // phase and resets the rest.
  let newPhase: Phase;
  let newState: Record<string, unknown>;
  const cs = currentState as Record<string, unknown>;
  const orgIds = (cs.selected_org_ids as number[] | undefined) ?? [];
  const entitiesMap =
    (cs.selected_entity_ids as Record<string, number[]> | undefined) ?? {};

  if (target === "org_select") {
    newPhase = "org_select";
    newState = {
      user_query: "",
      ai_candidates: [],
      selected_org_ids: orgIds,
    };
  } else if (target === "entity_select") {
    newPhase = "entity_select";
    newState = {
      inherits_from_version: parentVersionId,
      selected_org_ids: orgIds,
      selected_entity_ids: entitiesMap,
    };
  } else {
    // historical_data_room -> always enter as data_room_setup (the
    // pre-build view). If the user has already built, they can navigate
    // back here for now and it'll show setup; the underlying room
    // still exists in dealcloud.
    newPhase = "data_room_setup";
    let preset_question_ids: number[] = [];
    try {
      const defaults = await api.getPresetQuestions();
      preset_question_ids = defaults.map((q) => q.id);
    } catch {
      // Fall through with empty list; the cron treats empty as "all".
    }
    newState = {
      inherits_from_version: parentVersionId,
      selected_org_ids: orgIds,
      selected_entity_ids: entitiesMap,
      preset_question_ids,
      custom_questions: [],
      data_room_id: null,
    };
  }

  try {
    const data = await api.appendVersion(sessionId, {
      parent_id: parentVersionId,
      phase: newPhase,
      state: newState,
      summary:
        target === "org_select"
          ? "Back to org_select"
          : target === "entity_select"
            ? "Back to entity_select"
            : "Advance to historical_data_room",
    });
    qc.setQueryData(["session", sessionId], {
      session: data.session,
      current_version: data.version,
    });
  } catch (err) {
    console.error("phase nav failed", err);
    qc.invalidateQueries({ queryKey: ["session", sessionId] });
  }
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
  // Search mode toggle. Trigram is the default (matches prior behavior;
  // best for exact-name lookups). Hybrid blends trigram with semantic
  // via RRF -- exact matches stay top-ranked, descriptive matches get
  // added. Semantic-only is here mainly for diagnostic A/B.
  const [searchMode, setSearchMode] = useState<SearchMode>("trigram");
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQ(q.trim()), 300);
    return () => clearTimeout(t);
  }, [q]);

  const selected = (state.selected_org_ids as number[] | undefined) ?? [];

  const search = useQuery({
    queryKey: ["orgs", "search", debouncedQ, searchMode],
    queryFn: () => api.searchOrgs(debouncedQ, 15, searchMode),
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

  // Bulk select / deselect all current search matches. Operates on
  // search.data (the full match set), not searchVisible (which hides
  // already-selected). Lets the user one-click select every match for
  // a query, or undo that. Single bulk version-append so undo unwinds
  // it as one unit; mirrors the entity_select page-checkbox pattern.
  const [bulkPending, setBulkPending] = useState(false);
  const onToggleAllMatches = () => {
    const matchIds = (search.data ?? []).map((r) => r.org_id);
    if (matchIds.length === 0) return;
    const cached = qc.getQueryData<SessionWithCurrent>(["session", sessionId]);
    if (!cached) return;
    const cur =
      (cached.current_version.state.selected_org_ids as
        | number[]
        | undefined) ?? [];
    const matchSet = new Set(matchIds);
    const allSelected = matchIds.every((id) => cur.includes(id));
    let nextIds: number[];
    let summary: string;
    if (allSelected) {
      nextIds = cur.filter((id) => !matchSet.has(id));
      summary = `Deselect ${matchIds.length} matching orgs`;
    } else {
      const toAdd = matchIds.filter((id) => !cur.includes(id));
      nextIds = [...cur, ...toAdd];
      summary = `Select ${toAdd.length} matching orgs`;
    }

    // Optimistic patch.
    qc.setQueryData<SessionWithCurrent>(["session", sessionId], {
      ...cached,
      current_version: {
        ...cached.current_version,
        state: {
          ...cached.current_version.state,
          selected_org_ids: nextIds,
          user_query: q,
        },
      },
    });

    setBulkPending(true);
    queueRef.current = queueRef.current
      .catch(() => undefined)
      .then(async () => {
        const latest = qc.getQueryData<SessionWithCurrent>([
          "session",
          sessionId,
        ]);
        const parentId = latest?.current_version.id ?? parentVersionId;
        try {
          const data = await api.appendVersion(sessionId, {
            parent_id: parentId,
            phase: "org_select",
            state: { ...state, selected_org_ids: nextIds, user_query: q },
            summary,
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
              const hasLater =
                JSON.stringify(serverIds) !== JSON.stringify(cacheIds);
              return hasLater
                ? {
                    session: data.session,
                    current_version: {
                      ...data.version,
                      state: old.current_version.state,
                    },
                  }
                : { session: data.session, current_version: data.version };
            },
          );
        } catch (err) {
          console.error("bulk org toggle failed", err);
          qc.invalidateQueries({ queryKey: ["session", sessionId] });
        } finally {
          setBulkPending(false);
        }
      });
  };

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
          (overflow-y-auto in ResearchPage) so it's always reachable.
          Below the input, a 3-way mode toggle: trigram (default, exact
          name lookups), hybrid (RRF over trigram + semantic; best for
          mixed queries), semantic (cosine only; good for descriptive
          phrasing). UI state only -- not persisted to session. */}
      <div className="sticky top-0 bg-white pb-2 z-20 -mx-1 px-1">
        <input
          type="text"
          autoFocus
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder={
            searchMode === "trigram"
              ? "Type a company name..."
              : searchMode === "hybrid"
                ? "Name, or describe the company..."
                : "Describe the kind of company you're looking for..."
          }
          className="w-full border border-slate-300 rounded-md px-3 py-2"
        />
        <SearchModeToggle mode={searchMode} onChange={setSearchMode} />
      </div>

      {/* Selected section + search results in one scroll flow. Selected
          orgs are visually grouped at the top via the header, but they
          scroll with everything else so a long selection doesn't
          fill the viewport and hide search results. The select-all
          toggle for the current search lives above the Selected
          section so it's the first action the user sees after typing
          a query. */}
      <div className="space-y-2 mt-2">
        {(search.data?.length ?? 0) > 0 && (
          <SelectAllMatchesHeader
            matchIds={(search.data ?? []).map((r) => r.org_id)}
            selectedIds={selected}
            disabled={bulkPending}
            onToggle={onToggleAllMatches}
          />
        )}

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

        {searchVisible.length === 0 && (search.data?.length ?? 0) > 0 && (
          <div className="text-xs text-slate-500 pt-1 px-3">
            All {search.data!.length} matching{" "}
            {search.data!.length === 1 ? "result is" : "results are"} in your
            selection above.
          </div>
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
          Advance to Entity select →
        </button>
      </div>
    </div>
  );
}
