import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import {
  api,
  ENTITY_TYPES,
  type EntityFilter,
  type EntityOrgContextRow,
  type EntityType,
  type SessionWithCurrent,
} from "../lib/api";
import { useChat } from "../stores/chat";

const TAB_LABEL: Record<EntityType, string> = {
  document: "Documents",
  email_thread: "Emails",
  calendar_event: "Calendar",
  slack_message_group: "Slack",
};

const PAGE_SIZE = 50;

interface PhaseState {
  inherits_from_version?: string;
  selected_org_ids?: number[];
  selected_entity_ids?: Partial<Record<EntityType, number[]>>;
}

/**
 * Phase 2 -- entity_select. Tabs by source, paginated lists with
 * checkboxes, free-text + date range filter (local UI state).
 *
 * Toggle pattern mirrors Phase 1: optimistic patch on the cached
 * session, queued POST so quick clicks don't 409 on parent_id.
 */
export default function EntitySelectPhase({
  sessionId,
  parentVersionId,
  state,
}: {
  sessionId: string;
  parentVersionId: string;
  state: Record<string, unknown>;
}) {
  const qc = useQueryClient();
  const ps = state as PhaseState;
  const selected: Record<EntityType, number[]> = {
    document: ps.selected_entity_ids?.document ?? [],
    email_thread: ps.selected_entity_ids?.email_thread ?? [],
    calendar_event: ps.selected_entity_ids?.calendar_event ?? [],
    slack_message_group: ps.selected_entity_ids?.slack_message_group ?? [],
  };
  const totalSelected =
    selected.document.length +
    selected.email_thread.length +
    selected.calendar_event.length +
    selected.slack_message_group.length;

  const [activeTab, setActiveTab] = useState<EntityType>("document");
  // Filter form (local). Date inputs use the HTML date format
  // YYYY-MM-DD; we forward as ISO strings on read.
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [contains, setContains] = useState("");
  // Debounced applied filter so per-keystroke renders don't refetch
  // and re-render the list.
  const [appliedFilter, setAppliedFilter] = useState<EntityFilter>({});
  useEffect(() => {
    const t = setTimeout(() => {
      setAppliedFilter({
        date_from: dateFrom ? `${dateFrom}T00:00:00Z` : null,
        date_to: dateTo ? `${dateTo}T23:59:59Z` : null,
        contains: contains.trim() || null,
      });
      // Reset pagination to page 0 when filter changes.
      setOffset(0);
    }, 300);
    return () => clearTimeout(t);
  }, [dateFrom, dateTo, contains]);

  const [offset, setOffset] = useState(0);

  // One count per tab so the tab bar can show "Documents (2,528)"
  // badges. We unroll to 4 useQuery calls because hook order has to
  // be stable across renders -- can't call useQuery inside .map().
  const countDocs = useQuery({
    queryKey: ["entities", sessionId, "document", "count", appliedFilter],
    queryFn: () => api.countEntities(sessionId, "document", appliedFilter),
    staleTime: 30_000,
  });
  const countEmails = useQuery({
    queryKey: ["entities", sessionId, "email_thread", "count", appliedFilter],
    queryFn: () => api.countEntities(sessionId, "email_thread", appliedFilter),
    staleTime: 30_000,
  });
  const countCal = useQuery({
    queryKey: ["entities", sessionId, "calendar_event", "count", appliedFilter],
    queryFn: () => api.countEntities(sessionId, "calendar_event", appliedFilter),
    staleTime: 30_000,
  });
  const countSlack = useQuery({
    queryKey: ["entities", sessionId, "slack_message_group", "count", appliedFilter],
    queryFn: () =>
      api.countEntities(sessionId, "slack_message_group", appliedFilter),
    staleTime: 30_000,
  });
  const countByType: Record<EntityType, number> = {
    document: countDocs.data?.count ?? 0,
    email_thread: countEmails.data?.count ?? 0,
    calendar_event: countCal.data?.count ?? 0,
    slack_message_group: countSlack.data?.count ?? 0,
  };

  const list = useQuery({
    queryKey: [
      "entities",
      sessionId,
      activeTab,
      "list",
      appliedFilter,
      offset,
    ],
    queryFn: () =>
      api.listEntities(sessionId, activeTab, appliedFilter, PAGE_SIZE, offset),
    staleTime: 30_000,
  });

  // Per-row pending visual + queued POSTs (same shape as Phase 1).
  const [pendingKeys, setPendingKeys] = useState<Set<string>>(new Set());
  const queueRef = useRef<Promise<void>>(Promise.resolve());

  // Generic helper used by both single-row toggle and the page-level
   // "select / deselect all visible" affordance. Takes the next id list
   // for one entity_type and posts one append-version. Pending-keys
   // are managed by the caller so per-row vs bulk visuals can differ.
  const setSelectionForType = (
    entityType: EntityType,
    nextIds: number[],
    summary: string,
  ): Promise<void> => {
    const cached = qc.getQueryData<SessionWithCurrent>(["session", sessionId]);
    if (!cached) return Promise.resolve();
    const cur = cached.current_version.state as PhaseState;

    qc.setQueryData<SessionWithCurrent>(["session", sessionId], {
      ...cached,
      current_version: {
        ...cached.current_version,
        state: {
          ...(cached.current_version.state as Record<string, unknown>),
          selected_entity_ids: {
            ...(cur.selected_entity_ids ?? {}),
            [entityType]: nextIds,
          },
        },
      },
    });

    queueRef.current = queueRef.current
      .catch(() => undefined)
      .then(async () => {
        const latest = qc.getQueryData<SessionWithCurrent>([
          "session",
          sessionId,
        ]);
        const parentId = latest?.current_version.id ?? parentVersionId;
        const baseState =
          (latest?.current_version.state as PhaseState | undefined) ?? cur;
        try {
          const data = await api.appendVersion(sessionId, {
            parent_id: parentId,
            phase: "entity_select",
            state: {
              ...(baseState as Record<string, unknown>),
              selected_entity_ids: {
                ...(baseState.selected_entity_ids ?? {}),
                [entityType]: nextIds,
              },
            },
            summary,
          });
          qc.setQueryData<SessionWithCurrent | undefined>(
            ["session", sessionId],
            (old) => {
              if (!old) {
                return { session: data.session, current_version: data.version };
              }
              const serverIds =
                ((data.version.state as PhaseState).selected_entity_ids?.[
                  entityType
                ]) ?? [];
              const cacheIds =
                ((old.current_version.state as PhaseState)
                  .selected_entity_ids?.[entityType]) ?? [];
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
          console.error("entity selection update failed", err);
          qc.invalidateQueries({ queryKey: ["session", sessionId] });
        }
      });
    return queueRef.current;
  };

  const toggle = (entityType: EntityType, entityId: number) => {
    const cached = qc.getQueryData<SessionWithCurrent>(["session", sessionId]);
    if (!cached) return;
    const cur = cached.current_version.state as PhaseState;
    const curIds = cur.selected_entity_ids?.[entityType] ?? [];
    const isSelected = curIds.includes(entityId);
    const nextIds = isSelected
      ? curIds.filter((x) => x !== entityId)
      : [...curIds, entityId];

    const key = `${entityType}:${entityId}`;
    setPendingKeys((prev) => {
      const n = new Set(prev);
      n.add(key);
      return n;
    });

    void setSelectionForType(
      entityType,
      nextIds,
      isSelected ? `Removed ${entityType}` : `Added ${entityType}`,
    ).finally(() => {
      setPendingKeys((prev) => {
        const n = new Set(prev);
        n.delete(key);
        return n;
      });
    });
  };

  // "Select / deselect all visible" -- operates on the current page's
   // rows only. Behavior:
   //   * 0 of N selected on page -> click selects all on page
   //   * partial / all selected   -> click deselects all on page
   // Single bulk version-append rather than per-row, both for speed
   // and so undo unwinds the bulk action as one unit.
  const onToggleAllVisible = () => {
    const visibleIds = (list.data?.rows ?? []).map((r) => r.id as number);
    if (visibleIds.length === 0) return;
    const cached = qc.getQueryData<SessionWithCurrent>(["session", sessionId]);
    if (!cached) return;
    const cur = cached.current_version.state as PhaseState;
    const curIds = cur.selected_entity_ids?.[activeTab] ?? [];
    const visibleSet = new Set(visibleIds);
    const allSelected = visibleIds.every((id) => curIds.includes(id));

    let nextIds: number[];
    let summary: string;
    if (allSelected) {
      nextIds = curIds.filter((id) => !visibleSet.has(id));
      summary = `Deselect ${visibleIds.length} ${activeTab} on page`;
    } else {
      const toAdd = visibleIds.filter((id) => !curIds.includes(id));
      nextIds = [...curIds, ...toAdd];
      summary = `Select ${toAdd.length} ${activeTab} on page`;
    }

    const bulkKey = `__bulk:${activeTab}`;
    setPendingKeys((prev) => {
      const n = new Set(prev);
      n.add(bulkKey);
      return n;
    });
    void setSelectionForType(activeTab, nextIds, summary).finally(() => {
      setPendingKeys((prev) => {
        const n = new Set(prev);
        n.delete(bulkKey);
        return n;
      });
    });
  };

  const totalPages = Math.max(1, Math.ceil((list.data?.count ?? 0) / PAGE_SIZE));
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  // Publish per-turn UI snapshot so the chat orchestrator can render
  // a "Current UI state (entity_select)" block. Without this the
  // model sees the phase banner but not concrete counts.
  const setUIContext = useChat((s) => s.setUIContext);
  useEffect(() => {
    setUIContext(sessionId, {
      phase: "entity_select",
      selected_org_ids: ps.selected_org_ids ?? [],
      active_tab: activeTab,
      count_by_type: countByType,
      selected_counts: {
        document: selected.document.length,
        email_thread: selected.email_thread.length,
        calendar_event: selected.calendar_event.length,
        slack_message_group: selected.slack_message_group.length,
      },
    });
  }, [
    sessionId,
    setUIContext,
    activeTab,
    countByType.document,
    countByType.email_thread,
    countByType.calendar_event,
    countByType.slack_message_group,
    selected.document.length,
    selected.email_thread.length,
    selected.calendar_event.length,
    selected.slack_message_group.length,
    ps.selected_org_ids,
  ]);

  return (
    <div>
      <h2 className="text-lg font-semibold mb-1">
        Phase 2 — Select entities
      </h2>
      <p className="text-sm text-slate-500 mb-3">
        Browse documents, emails, calendar events, and slack threads
        attached to your selected orgs. Use the filter to narrow by
        date or keyword. Click rows to add or remove from your data
        room. {totalSelected} entities selected.
      </p>

      {/* Filter bar */}
      <div className="grid grid-cols-[1fr_120px_120px] gap-2 mb-3">
        <input
          type="text"
          placeholder="Filter by keyword..."
          value={contains}
          onChange={(e) => setContains(e.target.value)}
          className="border border-slate-300 rounded-md px-3 py-2 text-sm"
        />
        <input
          type="date"
          value={dateFrom}
          onChange={(e) => setDateFrom(e.target.value)}
          className="border border-slate-300 rounded-md px-3 py-2 text-sm"
        />
        <input
          type="date"
          value={dateTo}
          onChange={(e) => setDateTo(e.target.value)}
          className="border border-slate-300 rounded-md px-3 py-2 text-sm"
        />
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-slate-200 mb-3">
        {ENTITY_TYPES.map((t) => {
          const c = countByType[t];
          const sel = selected[t].length;
          const isActive = t === activeTab;
          return (
            <button
              key={t}
              type="button"
              onClick={() => {
                setActiveTab(t);
                setOffset(0);
              }}
              className={
                "px-3 py-2 text-sm border-b-2 -mb-px " +
                (isActive
                  ? "border-slate-900 text-slate-900 font-medium"
                  : "border-transparent text-slate-500 hover:text-slate-700")
              }
            >
              {TAB_LABEL[t]}{" "}
              <span className="text-xs text-slate-400 tabular-nums">
                ({c.toLocaleString()})
              </span>
              {sel > 0 && (
                <span className="ml-1 text-xs bg-slate-900 text-white rounded-full px-1.5 py-0.5">
                  {sel}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* List */}
      {list.isLoading && (
        <div className="text-slate-500 text-sm">Loading...</div>
      )}
      {list.error && (
        <div className="text-red-600 text-sm">
          {(list.error as Error).message}
        </div>
      )}

      {/* "Select all on page" header. Only shown when there are rows
          to act on. Indeterminate when some-but-not-all are selected
          (browser checkbox doesn't have an :indeterminate attribute,
          so we set it via a ref). For a more sweeping selection
          (cross-page, cross-filter) the AI chat's select_all_matching
          tool handles that. */}
      {(list.data?.rows.length ?? 0) > 0 && (
        <SelectAllVisibleHeader
          visibleIds={(list.data?.rows ?? []).map((r) => r.id as number)}
          selectedIds={selected[activeTab]}
          totalCount={list.data?.count ?? 0}
          pageOffset={offset}
          pageSize={PAGE_SIZE}
          disabled={pendingKeys.has(`__bulk:${activeTab}`)}
          onToggle={onToggleAllVisible}
        />
      )}

      <div className="space-y-1">
        {(list.data?.rows ?? []).map((row) => (
          <EntityRow
            key={String(row.id)}
            entityType={activeTab}
            row={row}
            sessionId={sessionId}
            selected={selected[activeTab].includes(row.id as number)}
            onToggle={() => toggle(activeTab, row.id as number)}
            disabled={pendingKeys.has(`${activeTab}:${row.id}`)}
          />
        ))}
        {list.data && list.data.rows.length === 0 && !list.isLoading && (
          <div className="text-sm text-slate-500 py-4">
            No matching {TAB_LABEL[activeTab].toLowerCase()}.
          </div>
        )}
      </div>

      {/* Pagination */}
      {(list.data?.count ?? 0) > PAGE_SIZE && (
        <div className="flex items-center justify-between mt-3 text-sm">
          <button
            type="button"
            onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
            disabled={offset === 0}
            className="px-3 py-1 border border-slate-300 rounded-md disabled:opacity-40"
          >
            ← Prev
          </button>
          <span className="text-slate-500">
            Page {currentPage} of {totalPages}
          </span>
          <button
            type="button"
            onClick={() => setOffset(offset + PAGE_SIZE)}
            disabled={offset + PAGE_SIZE >= (list.data?.count ?? 0)}
            className="px-3 py-1 border border-slate-300 rounded-md disabled:opacity-40"
          >
            Next →
          </button>
        </div>
      )}

      {/* Phase navigation -- back/advance via direct version append.
          Advance only enabled if at least one entity selected. */}
      <div className="flex items-center justify-between mt-6 pt-4 border-t border-slate-200">
        <PhaseNavButton
          label="← Back to Org select"
          onClick={() =>
            navPhase(qc, sessionId, parentVersionId, state, "org_select")
          }
        />
        <PhaseNavButton
          label="Advance to historical data room →"
          disabled={totalSelected === 0}
          primary
          onClick={() =>
            navPhase(
              qc,
              sessionId,
              parentVersionId,
              state,
              "data_room_setup",
            )
          }
        />
      </div>
    </div>
  );
}

function SelectAllVisibleHeader({
  visibleIds,
  selectedIds,
  totalCount,
  pageOffset,
  pageSize,
  disabled,
  onToggle,
}: {
  visibleIds: number[];
  selectedIds: number[];
  totalCount: number;
  pageOffset: number;
  pageSize: number;
  disabled?: boolean;
  onToggle: () => void;
}) {
  const ref = useRef<HTMLInputElement | null>(null);
  const selectedSet = new Set(selectedIds);
  const onPage = visibleIds.filter((id) => selectedSet.has(id)).length;
  const allChecked = onPage === visibleIds.length && visibleIds.length > 0;
  const someChecked = onPage > 0 && !allChecked;

  // Indeterminate is a property, not an HTML attribute; sync via ref
  // whenever the selection on the page changes.
  useEffect(() => {
    if (ref.current) ref.current.indeterminate = someChecked;
  }, [someChecked]);

  const pageStart = pageOffset + 1;
  const pageEnd = Math.min(pageOffset + pageSize, totalCount);
  const label = allChecked
    ? `Deselect all ${visibleIds.length} on page`
    : someChecked
    ? `Select remaining ${visibleIds.length - onPage} on page`
    : `Select all ${visibleIds.length} on page`;

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
      <span className="ml-auto text-xs text-slate-400 tabular-nums">
        {pageStart.toLocaleString()}–{pageEnd.toLocaleString()} of{" "}
        {totalCount.toLocaleString()}
      </span>
    </div>
  );
}

function EntityRow({
  entityType,
  row,
  selected,
  onToggle,
  disabled,
  sessionId,
}: {
  entityType: EntityType;
  row: Record<string, unknown>;
  selected: boolean;
  onToggle: () => void;
  disabled?: boolean;
  sessionId: string;
}) {
  const r = row as any; // shape varies by entity_type
  const [expanded, setExpanded] = useState(false);
  let title: string;
  let subtitle: string;
  let date: string | null;

  switch (entityType) {
    case "document":
      title = r.name ?? `document #${r.id}`;
      subtitle = r.path ?? "";
      date = r.modified_at ?? null;
      break;
    case "email_thread":
      title = r.subject ?? `thread #${r.id}`;
      subtitle = `${r.message_count ?? 0} messages${r.category ? " · " + r.category : ""}`;
      date = r.last_message_at ?? null;
      break;
    case "calendar_event":
      title = r.subject ?? `event #${r.id}`;
      subtitle = [
        r.organizer_name || r.organizer_email,
        r.location,
        r.is_online ? "online" : null,
      ]
        .filter(Boolean)
        .join(" · ");
      date = r.start_time ?? null;
      break;
    case "slack_message_group": {
      const channel = r.channel_name
        ? `#${r.channel_name}`
        : `slack group #${r.id}`;
      const firstTs = r.thread_ts
        ? new Date(parseInt(r.thread_ts.split(".")[0], 10) * 1000)
        : null;
      const lastTs = r.last_ts
        ? new Date(parseInt(r.last_ts.split(".")[0], 10) * 1000)
        : null;
      const fmt = (d: Date) => d.toISOString().slice(0, 10);
      let span = "";
      if (firstTs && lastTs) {
        span =
          fmt(firstTs) === fmt(lastTs)
            ? fmt(lastTs)
            : `${fmt(firstTs)} → ${fmt(lastTs)}`;
      } else if (lastTs) {
        span = fmt(lastTs);
      }
      title = span ? `${channel} · ${span}` : channel;
      subtitle = `${r.message_count ?? 0} messages`;
      date = lastTs ? lastTs.toISOString() : null;
      break;
    }
  }

  return (
    <div
      className={
        "border rounded-md transition-colors " +
        (disabled ? "opacity-50 pointer-events-none " : "") +
        (selected
          ? "border-slate-900 bg-slate-50"
          : "border-slate-200 bg-white hover:bg-slate-50")
      }
    >
      <div
        onClick={() => setExpanded((v) => !v)}
        role="button"
        aria-expanded={expanded}
        className="px-3 py-2 cursor-pointer"
      >
        <div className="flex items-start gap-2">
          <span
            onClick={(e) => {
              e.stopPropagation();
              if (!disabled) onToggle();
            }}
            onMouseDown={(e) => e.stopPropagation()}
            className="p-1 -m-1"
          >
            <input
              type="checkbox"
              readOnly
              checked={selected}
              disabled={disabled}
              className="mt-0.5 cursor-pointer"
            />
          </span>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium truncate">{title}</div>
            {subtitle && (
              <div className="text-xs text-slate-500 truncate">{subtitle}</div>
            )}
            {r.summary && (
              <div className="text-xs text-slate-500 mt-0.5 line-clamp-2">
                {r.summary}
              </div>
            )}
          </div>
          <div className="text-xs text-slate-400 shrink-0 tabular-nums">
            {date ? formatDate(date) : "—"}
          </div>
          <RowChevron open={expanded} />
        </div>
      </div>
      {expanded && (
        <div className="border-t border-slate-200 px-3 py-3 text-xs text-slate-700 space-y-3">
          <EntityDetail entityType={entityType} row={r} />
          <OrgContextBlock
            sessionId={sessionId}
            entityType={entityType}
            entityId={r.id as number}
          />
        </div>
      )}
    </div>
  );
}

function OrgContextBlock({
  sessionId,
  entityType,
  entityId,
}: {
  sessionId: string;
  entityType: EntityType;
  entityId: number;
}) {
  const ctx = useQuery({
    queryKey: ["entity-org-context", entityType, entityId, sessionId],
    queryFn: () =>
      api.getEntityOrgContext(sessionId, entityType, entityId),
    // Org-link metadata barely ever changes after the entity is
    // indexed; keep it cached across re-expands.
    staleTime: 5 * 60_000,
  });

  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-slate-500 font-semibold mb-1">
        Org link
      </div>
      {ctx.isLoading && (
        <div className="text-slate-400">Loading org context…</div>
      )}
      {ctx.error && (
        <div className="text-red-600">
          {(ctx.error as Error).message}
        </div>
      )}
      {ctx.data && ctx.data.rows.length === 0 && (
        <div className="text-slate-400">
          No `{entityType}_organization` rows for the selected orgs.
        </div>
      )}
      {ctx.data && ctx.data.rows.length > 0 && (
        <ul className="space-y-2">
          {ctx.data.rows.map((r, i) => (
            <OrgContextRowView key={i} row={r} entityType={entityType} />
          ))}
        </ul>
      )}
    </div>
  );
}

function OrgContextRowView({
  row,
  entityType,
}: {
  row: EntityOrgContextRow;
  entityType: EntityType;
}) {
  const meta: string[] = [];
  if (row.relationship_type) meta.push(row.relationship_type);
  if (row.alias_text && row.alias_text !== row.org_name)
    meta.push(`alias: ${row.alias_text}`);
  if (entityType === "document" && row.match_method)
    meta.push(`via ${row.match_method}`);
  if (entityType === "email_thread" && row.confidence != null)
    meta.push(`conf ${row.confidence.toFixed(2)}`);
  if (row.model) meta.push(row.model);

  return (
    <li className="border border-slate-200 rounded-md px-2 py-1.5 bg-slate-50">
      <div className="flex items-center gap-2">
        <span className="font-medium text-slate-800 truncate">
          {row.org_name}
        </span>
        <code className="text-[10px] text-slate-400 shrink-0">
          #{row.org_id}
        </code>
        {row.is_confirmed && (
          <span className="ml-auto text-[10px] uppercase tracking-wide bg-emerald-100 text-emerald-800 px-1.5 py-0.5 rounded shrink-0">
            confirmed
          </span>
        )}
      </div>
      {meta.length > 0 && (
        <div className="mt-0.5 text-[11px] text-slate-500">
          {meta.join(" · ")}
        </div>
      )}
      {row.context && (
        <div className="mt-1 text-slate-700 leading-relaxed whitespace-pre-wrap">
          {row.context}
        </div>
      )}
      {row.notes && (
        <div className="mt-1 text-slate-500 italic">{row.notes}</div>
      )}
    </li>
  );
}

function EntityDetail({
  entityType,
  row,
}: {
  entityType: EntityType;
  row: any;
}) {
  switch (entityType) {
    case "document":
      return (
        <div className="space-y-2">
          {row.summary && <Summary text={row.summary} />}
          <KVRow label="Path" value={row.path} />
          <KVRow label="Modified" value={formatTs(row.modified_at)} />
          <KVRow label="MIME" value={row.mime_type} />
          <KVRow
            label="Size"
            value={
              row.size_bytes ? formatBytes(row.size_bytes as number) : null
            }
          />
          {row.web_url && (
            <div>
              <a
                href={row.web_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                Open in SharePoint ↗
              </a>
            </div>
          )}
        </div>
      );
    case "email_thread":
      return (
        <div className="space-y-2">
          {row.summary && <Summary text={row.summary} />}
          <KVRow
            label="Span"
            value={
              row.first_message_at && row.last_message_at
                ? `${formatTs(row.first_message_at)} → ${formatTs(row.last_message_at)}`
                : formatTs(row.last_message_at)
            }
          />
          <KVRow label="Messages" value={String(row.message_count ?? 0)} />
          <KVRow label="Category" value={row.category} />
          <KVRow
            label="Internal / External"
            value={
              row.internal_count != null && row.external_count != null
                ? `${row.internal_count} / ${row.external_count}`
                : null
            }
          />
        </div>
      );
    case "calendar_event":
      return (
        <div className="space-y-2">
          {row.summary && <Summary text={row.summary} />}
          <KVRow
            label="Start"
            value={formatTs(row.start_time)}
          />
          <KVRow label="End" value={formatTs(row.end_time)} />
          <KVRow
            label="Organizer"
            value={row.organizer_name || row.organizer_email}
          />
          <KVRow label="Location" value={row.location} />
          <KVRow
            label="Online"
            value={row.is_online ? "yes" : null}
          />
          <KVRow
            label="External attendees"
            value={row.has_external ? "yes" : null}
          />
        </div>
      );
    case "slack_message_group":
      return (
        <div className="space-y-2">
          {row.summary && <Summary text={row.summary} />}
          <KVRow
            label="Channel"
            value={row.channel_name ? `#${row.channel_name}` : null}
          />
          <KVRow
            label="Span"
            value={(() => {
              const a = row.thread_ts ? formatSlackTs(row.thread_ts) : null;
              const b = row.last_ts ? formatSlackTs(row.last_ts) : null;
              if (a && b && a !== b) return `${a} → ${b}`;
              return b ?? a ?? null;
            })()}
          />
          <KVRow label="Messages" value={String(row.message_count ?? 0)} />
          {row.permalink && (
            <div>
              <a
                href={row.permalink}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                Open in Slack ↗
              </a>
            </div>
          )}
        </div>
      );
  }
}

function Summary({ text }: { text: string }) {
  return (
    <div className="leading-relaxed text-slate-800 whitespace-pre-wrap">
      {text}
    </div>
  );
}

function KVRow({
  label,
  value,
}: {
  label: string;
  value: string | null | undefined;
}) {
  if (!value) return null;
  return (
    <div className="flex gap-2">
      <span className="text-[10px] uppercase tracking-wide text-slate-500 font-semibold w-24 shrink-0 mt-0.5">
        {label}
      </span>
      <span className="text-slate-800 break-words">{value}</span>
    </div>
  );
}

function RowChevron({ open }: { open: boolean }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={
        "shrink-0 text-slate-400 mt-1 transition-transform " +
        (open ? "rotate-180" : "")
      }
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

function formatTs(ts: string | null | undefined): string | null {
  if (!ts) return null;
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts.slice(0, 10);
  return d.toISOString().replace("T", " ").slice(0, 16) + "Z";
}

function formatSlackTs(ts: string): string {
  const sec = parseInt(ts.split(".")[0], 10);
  if (Number.isNaN(sec)) return ts;
  return new Date(sec * 1000).toISOString().replace("T", " ").slice(0, 16) + "Z";
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function PhaseNavButton({
  label,
  onClick,
  disabled,
  primary,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  primary?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={
        "px-3 py-2 rounded-md text-sm disabled:opacity-40 " +
        (primary
          ? "bg-slate-900 text-white"
          : "border border-slate-300 text-slate-700")
      }
    >
      {label}
    </button>
  );
}

async function navPhase(
  qc: ReturnType<typeof useQueryClient>,
  sessionId: string,
  parentVersionId: string,
  state: Record<string, unknown>,
  newPhase: "org_select" | "data_room_setup",
) {
  // When advancing into data_room_setup, pre-populate preset_question_ids
  // with all default question ids so the user lands with everything
  // already checked. Mirrors the AI tool path
  // (advance_to_data_room_setup), which also seeds defaults.
  let nextState: Record<string, unknown> = { ...state };
  if (newPhase === "data_room_setup") {
    try {
      const defaults = await api.getPresetQuestions();
      nextState = {
        ...nextState,
        preset_question_ids: defaults.map((q) => q.id),
        custom_questions: [],
        data_room_id: null,
      };
    } catch (err) {
      console.warn(
        "failed to seed default preset_question_ids; advancing without them",
        err,
      );
      // Fall through with the prior state. The backend cron's "empty
      // list = all defaults" fallback still kicks in if the user
      // builds without picking any questions.
    }
  }

  // Direct version append; no optimistic UI since the phase change
  // is rare and the resulting layout shift would be misleading.
  try {
    const data = await api.appendVersion(sessionId, {
      parent_id: parentVersionId,
      phase: newPhase,
      state: nextState,
      summary:
        newPhase === "org_select"
          ? "Back to org_select"
          : "Advance to data_room_setup",
    });
    qc.setQueryData(["session", sessionId], {
      session: data.session,
      current_version: data.version,
    });
  } catch (err) {
    console.error("phase navigation failed", err);
    qc.invalidateQueries({ queryKey: ["session", sessionId] });
  }
}

function formatDate(ts: string): string {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts.slice(0, 10);
  return d.toISOString().slice(0, 10);
}
