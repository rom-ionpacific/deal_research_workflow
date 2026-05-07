import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  api,
  type PresetQuestion,
  type SessionWithCurrent,
} from "../lib/api";
import { useUI } from "../stores/ui";

interface PhaseState {
  selected_org_ids?: number[];
  selected_entity_ids?: Record<string, number[]>;
  preset_question_ids?: number[];
  data_room_id?: number | null;
}

/**
 * Phase 3 -- data_room_setup. Pick which preset questions the AI
 * should ask, then click Build to materialise a dealcloud data room
 * row. The data-room-builder cron in deal_cloud_enhancer picks it up
 * within ~2 min, uploads the entities to ToltIQ, runs the playlist,
 * and saves answers. We transition to data_room_view on build (which
 * is currently a placeholder until Phase 4 ships).
 *
 * Custom questions: stored in the same dealcloud.data_room_preset_question
 * table with grouping=NULL and originator=<user.email>. Edits create a
 * new row rather than UPDATE-ing in place, so historical data rooms
 * keep their original wording. The session's preset_question_ids
 * carries the union of default + custom ids.
 */
export default function DataRoomSetupPhase({
  sessionId,
  parentVersionId,
  state,
}: {
  sessionId: string;
  parentVersionId: string;
  state: Record<string, unknown>;
}) {
  const qc = useQueryClient();
  const userEmail = useUI((s) => s.userEmail);
  const ps = state as PhaseState;
  const selectedQuestions = ps.preset_question_ids ?? [];
  const entityMap = ps.selected_entity_ids ?? {};
  const totalEntities =
    (entityMap.document?.length ?? 0) +
    (entityMap.email_thread?.length ?? 0) +
    (entityMap.calendar_event?.length ?? 0) +
    (entityMap.slack_message_group?.length ?? 0);

  const presets = useQuery({
    queryKey: ["preset-questions"],
    queryFn: api.getPresetQuestions,
    staleTime: 5 * 60_000, // preset list rarely changes
  });

  const defaultIds = useMemo(
    () => new Set((presets.data ?? []).map((q) => q.id)),
    [presets.data],
  );

  // Customs are whatever's selected but not in the defaults list. We
  // sort to keep the cache key stable across renders so the
  // by-ids query doesn't refetch on cosmetic re-renders.
  const customIds = useMemo(
    () =>
      selectedQuestions
        .filter((id) => !defaultIds.has(id))
        .slice()
        .sort((a, b) => a - b),
    [selectedQuestions, defaultIds],
  );

  const customs = useQuery({
    queryKey: ["preset-questions-by-ids", customIds.join(",")],
    queryFn: () => api.getPresetQuestionsByIds(customIds),
    enabled: customIds.length > 0,
    staleTime: Infinity, // a row's content never changes (edits make new rows)
  });

  const customsById = useMemo(() => {
    const m = new Map<number, PresetQuestion>();
    for (const r of customs.data ?? []) m.set(r.id, r);
    return m;
  }, [customs.data]);

  // Render customs in the order they appear in selectedQuestions, so
  // edit ordering is preserved across renders even after id swaps.
  const customsInOrder = useMemo(
    () =>
      selectedQuestions
        .filter((id) => !defaultIds.has(id))
        .map((id) => customsById.get(id))
        .filter((q): q is PresetQuestion => Boolean(q)),
    [selectedQuestions, defaultIds, customsById],
  );

  const [pendingIds, setPendingIds] = useState<Set<number>>(new Set());
  const queueRef = useRef<Promise<void>>(Promise.resolve());

  // Append a new session_version that swaps preset_question_ids for
  // `nextIds`. Used by toggle, create-custom, and edit-custom paths.
  // Optimistically patches the session cache, then chains the POST so
  // concurrent ops serialise via parent_id.
  const queueQuestionMutation = (
    nextIds: number[],
    summary: string,
  ): Promise<void> => {
    const cached = qc.getQueryData<SessionWithCurrent>(["session", sessionId]);
    if (!cached) return Promise.resolve();
    qc.setQueryData<SessionWithCurrent>(["session", sessionId], {
      ...cached,
      current_version: {
        ...cached.current_version,
        state: {
          ...(cached.current_version.state as Record<string, unknown>),
          preset_question_ids: nextIds,
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
          (latest?.current_version.state as Record<string, unknown>) ?? {};
        try {
          const data = await api.appendVersion(sessionId, {
            parent_id: parentId,
            phase: "data_room_setup",
            state: { ...baseState, preset_question_ids: nextIds },
            summary,
          });
          qc.setQueryData<SessionWithCurrent | undefined>(
            ["session", sessionId],
            (old) => {
              if (!old) {
                return { session: data.session, current_version: data.version };
              }
              const serverIds =
                (data.version.state as PhaseState).preset_question_ids ?? [];
              const cacheIds =
                (old.current_version.state as PhaseState)
                  .preset_question_ids ?? [];
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
          console.error("question mutation failed", err);
          qc.invalidateQueries({ queryKey: ["session", sessionId] });
        }
      });

    return queueRef.current;
  };

  // Bulk select / deselect all preset (default-grouping) questions.
  // Operates only on defaults; customs are managed individually since
  // they're typically small in number and per-row edit/remove is more
  // useful than a master toggle.
  const [bulkPending, setBulkPending] = useState(false);
  const onToggleAllPresets = () => {
    const presetIds = (presets.data ?? []).map((q) => q.id);
    if (presetIds.length === 0) return;
    const cached = qc.getQueryData<SessionWithCurrent>(["session", sessionId]);
    if (!cached) return;
    const cur = cached.current_version.state as PhaseState;
    const curIds = cur.preset_question_ids ?? [];
    const presetSet = new Set(presetIds);
    const allSelected = presetIds.every((id) => curIds.includes(id));
    let nextIds: number[];
    let summary: string;
    if (allSelected) {
      nextIds = curIds.filter((id) => !presetSet.has(id));
      summary = `Deselect all ${presetIds.length} preset questions`;
    } else {
      const toAdd = presetIds.filter((id) => !curIds.includes(id));
      nextIds = [...curIds, ...toAdd];
      summary = `Select all ${presetIds.length} preset questions`;
    }
    setBulkPending(true);
    void queueQuestionMutation(nextIds, summary).finally(() =>
      setBulkPending(false),
    );
  };

  const toggle = (questionId: number) => {
    const cached = qc.getQueryData<SessionWithCurrent>(["session", sessionId]);
    if (!cached) return;
    const cur = cached.current_version.state as PhaseState;
    const curIds = cur.preset_question_ids ?? [];
    const isSelected = curIds.includes(questionId);
    const nextIds = isSelected
      ? curIds.filter((x) => x !== questionId)
      : [...curIds, questionId];
    setPendingIds((prev) => new Set(prev).add(questionId));
    void queueQuestionMutation(
      nextIds,
      isSelected
        ? `Removed question ${questionId}`
        : `Added question ${questionId}`,
    ).finally(() => {
      setPendingIds((prev) => {
        const n = new Set(prev);
        n.delete(questionId);
        return n;
      });
    });
  };

  // ---- Custom questions: add ----------------------------------------------
  const [addingNew, setAddingNew] = useState(false);
  const [newLabel, setNewLabel] = useState("");
  const [newText, setNewText] = useState("");
  const [addError, setAddError] = useState<string | null>(null);
  const [savingNew, setSavingNew] = useState(false);

  const cancelAdd = () => {
    setAddingNew(false);
    setNewLabel("");
    setNewText("");
    setAddError(null);
  };

  const saveNewCustom = async () => {
    const label = newLabel.trim();
    const question_text = newText.trim();
    if (!label || !question_text) {
      setAddError("Both label and question text are required.");
      return;
    }
    setSavingNew(true);
    setAddError(null);
    try {
      const created = await api.createPresetQuestion(label, question_text);
      // Pre-warm the by-ids cache so the row renders immediately.
      qc.setQueryData<PresetQuestion[]>(
        [
          "preset-questions-by-ids",
          [...customIds, created.id].sort((a, b) => a - b).join(","),
        ],
        [...(customs.data ?? []), created],
      );
      const cached = qc.getQueryData<SessionWithCurrent>([
        "session",
        sessionId,
      ]);
      const curIds =
        (cached?.current_version.state as PhaseState | undefined)
          ?.preset_question_ids ?? selectedQuestions;
      const nextIds = curIds.includes(created.id)
        ? curIds
        : [...curIds, created.id];
      cancelAdd();
      await queueQuestionMutation(
        nextIds,
        `Add custom question ${created.id} (${label})`,
      );
    } catch (err) {
      setAddError(err instanceof Error ? err.message : String(err));
    } finally {
      setSavingNew(false);
    }
  };

  // ---- Custom questions: edit ---------------------------------------------
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editLabel, setEditLabel] = useState("");
  const [editText, setEditText] = useState("");
  const [editError, setEditError] = useState<string | null>(null);
  const [savingEdit, setSavingEdit] = useState(false);

  const startEdit = (q: PresetQuestion) => {
    setEditingId(q.id);
    setEditLabel(q.label);
    setEditText(q.question_text);
    setEditError(null);
  };
  const cancelEdit = () => {
    setEditingId(null);
    setEditLabel("");
    setEditText("");
    setEditError(null);
  };

  const saveEdit = async (oldQ: PresetQuestion) => {
    const label = editLabel.trim();
    const question_text = editText.trim();
    if (!label || !question_text) {
      setEditError("Both label and question text are required.");
      return;
    }
    if (label === oldQ.label && question_text === oldQ.question_text) {
      cancelEdit();
      return;
    }
    setSavingEdit(true);
    setEditError(null);
    try {
      const created = await api.createPresetQuestion(label, question_text);
      const cached = qc.getQueryData<SessionWithCurrent>([
        "session",
        sessionId,
      ]);
      const curIds =
        (cached?.current_version.state as PhaseState | undefined)
          ?.preset_question_ids ?? selectedQuestions;
      // Swap old id -> new id in place to preserve ordering.
      const nextIds = curIds.includes(oldQ.id)
        ? curIds.map((x) => (x === oldQ.id ? created.id : x))
        : [...curIds, created.id];
      // Pre-warm the by-ids cache for the new key so the row renders
      // immediately without a refetch.
      const nextCustomIds = nextIds
        .filter((id) => !defaultIds.has(id))
        .slice()
        .sort((a, b) => a - b);
      qc.setQueryData<PresetQuestion[]>(
        ["preset-questions-by-ids", nextCustomIds.join(",")],
        [
          ...(customs.data ?? []).filter((q) => q.id !== oldQ.id),
          created,
        ],
      );
      cancelEdit();
      await queueQuestionMutation(
        nextIds,
        `Edit custom question ${oldQ.id} -> ${created.id}`,
      );
    } catch (err) {
      setEditError(err instanceof Error ? err.message : String(err));
    } finally {
      setSavingEdit(false);
    }
  };

  // ---- Build / back -------------------------------------------------------
  const [building, setBuilding] = useState(false);
  const [buildError, setBuildError] = useState<string | null>(null);

  const onBuild = async () => {
    setBuilding(true);
    setBuildError(null);
    try {
      const resp = await api.buildDataRoom(sessionId);
      await qc.invalidateQueries({ queryKey: ["session", sessionId] });
      console.info("Built data room:", resp);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setBuildError(msg);
    } finally {
      setBuilding(false);
    }
  };

  const onBack = async () => {
    try {
      const data = await api.appendVersion(sessionId, {
        parent_id: parentVersionId,
        phase: "entity_select",
        state: {
          ...(ps as Record<string, unknown>),
        },
        summary: "Back to entity_select",
      });
      qc.setQueryData(["session", sessionId], {
        session: data.session,
        current_version: data.version,
      });
    } catch (err) {
      console.error("back failed", err);
      qc.invalidateQueries({ queryKey: ["session", sessionId] });
    }
  };

  return (
    <div>
      <h2 className="text-lg font-semibold mb-1">
        Phase 3 — Data room setup
      </h2>
      <p className="text-sm text-slate-500 mb-3">
        Pick which questions the AI should answer using your selection.{" "}
        {totalEntities.toLocaleString()} entities scoped from Phase 2.
        Click Build to ship the data room to ToltIQ — the builder cron
        will upload your entities, run the playlist, and save answers
        (typically 10–15 min).
      </p>

      {presets.isLoading && (
        <div className="text-sm text-slate-500">
          Loading preset questions...
        </div>
      )}
      {presets.error && (
        <div className="text-sm text-red-600">
          {(presets.error as Error).message}
        </div>
      )}

      {/* Defaults section */}
      {(presets.data?.length ?? 0) > 0 && (
        <SelectAllPresetsHeader
          presetIds={(presets.data ?? []).map((q) => q.id)}
          selectedIds={selectedQuestions}
          disabled={bulkPending}
          onToggle={onToggleAllPresets}
        />
      )}
      <div className="space-y-2">
        {(presets.data ?? []).map((q: PresetQuestion) => (
          <PresetQuestionRow
            key={q.id}
            q={q}
            selected={selectedQuestions.includes(q.id)}
            onToggle={() => toggle(q.id)}
            disabled={pendingIds.has(q.id)}
          />
        ))}
      </div>

      {/* Customs section */}
      <div className="mt-6">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold text-slate-700">
            Custom questions
          </h3>
          {!addingNew && (
            <button
              type="button"
              onClick={() => setAddingNew(true)}
              className="text-xs px-2 py-1 border border-slate-300 rounded-md hover:bg-slate-50"
            >
              + Add custom question
            </button>
          )}
        </div>

        {customsInOrder.length === 0 && !addingNew && (
          <div className="text-xs text-slate-500">
            None yet. Add one if a question you want answered isn't
            covered by the defaults above.
          </div>
        )}

        <div className="space-y-2">
          {customsInOrder.map((q) => {
            const owned =
              !!q.originator && !!userEmail && q.originator === userEmail;
            if (editingId === q.id) {
              return (
                <CustomQuestionEditor
                  key={q.id}
                  label={editLabel}
                  text={editText}
                  setLabel={setEditLabel}
                  setText={setEditText}
                  saving={savingEdit}
                  error={editError}
                  onCancel={cancelEdit}
                  onSave={() => void saveEdit(q)}
                  saveLabel="Save"
                />
              );
            }
            return (
              <CustomQuestionRow
                key={q.id}
                q={q}
                selected={selectedQuestions.includes(q.id)}
                onToggle={() => toggle(q.id)}
                onEdit={owned ? () => startEdit(q) : undefined}
                disabled={pendingIds.has(q.id)}
              />
            );
          })}

          {addingNew && (
            <CustomQuestionEditor
              label={newLabel}
              text={newText}
              setLabel={setNewLabel}
              setText={setNewText}
              saving={savingNew}
              error={addError}
              onCancel={cancelAdd}
              onSave={() => void saveNewCustom()}
              saveLabel="Add to plan"
            />
          )}
        </div>
      </div>

      {selectedQuestions.length === 0 && (presets.data?.length ?? 0) > 0 && (
        <div className="text-xs text-slate-500 mt-3">
          Tip: with no questions selected the cron falls back to all
          default questions.
        </div>
      )}

      {buildError && (
        <div className="mt-4 text-sm text-red-600 border border-red-200 bg-red-50 rounded-md px-3 py-2">
          Build failed: {buildError}
        </div>
      )}

      <div className="flex items-center justify-between mt-6 pt-4 border-t border-slate-200">
        <button
          type="button"
          onClick={onBack}
          disabled={building}
          className="px-3 py-2 border border-slate-300 text-slate-700 text-sm rounded-md disabled:opacity-40"
        >
          ← Back to entity_select
        </button>
        <button
          type="button"
          onClick={onBuild}
          disabled={building || totalEntities === 0}
          className="px-4 py-2 bg-slate-900 text-white text-sm rounded-md disabled:opacity-40"
        >
          {building ? "Building..." : "Build data room →"}
        </button>
      </div>
    </div>
  );
}

function SelectAllPresetsHeader({
  presetIds,
  selectedIds,
  disabled,
  onToggle,
}: {
  presetIds: number[];
  selectedIds: number[];
  disabled?: boolean;
  onToggle: () => void;
}) {
  const ref = useRef<HTMLInputElement | null>(null);
  const selectedSet = new Set(selectedIds);
  const onCount = presetIds.filter((id) => selectedSet.has(id)).length;
  const allChecked = onCount === presetIds.length && presetIds.length > 0;
  const someChecked = onCount > 0 && !allChecked;

  useEffect(() => {
    if (ref.current) ref.current.indeterminate = someChecked;
  }, [someChecked]);

  const label = allChecked
    ? `Deselect all ${presetIds.length} preset questions`
    : someChecked
    ? `Select remaining ${presetIds.length - onCount} preset questions`
    : `Select all ${presetIds.length} preset questions`;

  return (
    <div className="flex items-center gap-2 px-3 py-2 mb-2 bg-slate-50 border border-slate-200 rounded-md">
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

function PresetQuestionRow({
  q,
  selected,
  onToggle,
  disabled,
}: {
  q: PresetQuestion;
  selected: boolean;
  onToggle: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      disabled={disabled}
      className={
        "w-full text-left border rounded-md px-3 py-2 hover:bg-slate-50 " +
        "disabled:opacity-50 transition-colors " +
        (selected
          ? "border-slate-900 bg-slate-50"
          : "border-slate-200 bg-white")
      }
    >
      <div className="flex items-start gap-2">
        <input type="checkbox" readOnly checked={selected} className="mt-0.5" />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium">{q.label}</div>
          <div className="text-xs text-slate-500 mt-0.5">{q.question_text}</div>
        </div>
        <code className="text-xs text-slate-400 shrink-0">#{q.id}</code>
      </div>
    </button>
  );
}

function CustomQuestionRow({
  q,
  selected,
  onToggle,
  onEdit,
  disabled,
}: {
  q: PresetQuestion;
  selected: boolean;
  onToggle: () => void;
  onEdit?: () => void;
  disabled?: boolean;
}) {
  return (
    <div
      className={
        "border rounded-md px-3 py-2 " +
        (selected
          ? "border-slate-900 bg-slate-50"
          : "border-slate-200 bg-white")
      }
    >
      <div className="flex items-start gap-2">
        <button
          type="button"
          onClick={onToggle}
          disabled={disabled}
          className="mt-0.5 disabled:opacity-50"
          aria-label={selected ? "Remove from plan" : "Add to plan"}
        >
          <input type="checkbox" readOnly checked={selected} />
        </button>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-wide bg-violet-100 text-violet-800 px-1.5 py-0.5 rounded">
              {q.originator ? `${q.originator}` : "user"}
            </span>
            <div className="text-sm font-medium">{q.label}</div>
          </div>
          <div className="text-xs text-slate-500 mt-0.5">{q.question_text}</div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {onEdit && (
            <button
              type="button"
              onClick={onEdit}
              className="text-xs px-2 py-0.5 border border-slate-300 rounded hover:bg-slate-100"
            >
              Edit
            </button>
          )}
          <code className="text-xs text-slate-400">#{q.id}</code>
        </div>
      </div>
    </div>
  );
}

function CustomQuestionEditor({
  label,
  text,
  setLabel,
  setText,
  saving,
  error,
  onCancel,
  onSave,
  saveLabel,
}: {
  label: string;
  text: string;
  setLabel: (v: string) => void;
  setText: (v: string) => void;
  saving: boolean;
  error: string | null;
  onCancel: () => void;
  onSave: () => void;
  saveLabel: string;
}) {
  return (
    <div className="border border-violet-300 bg-violet-50/40 rounded-md px-3 py-2">
      <input
        type="text"
        value={label}
        onChange={(e) => setLabel(e.target.value)}
        placeholder="Label (short description, max 200 chars)"
        maxLength={200}
        className="w-full text-sm border border-slate-300 rounded px-2 py-1 mb-2"
        disabled={saving}
      />
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Full question text — what should the LLM answer?"
        maxLength={2000}
        rows={3}
        className="w-full text-sm border border-slate-300 rounded px-2 py-1"
        disabled={saving}
      />
      {error && <div className="text-xs text-red-600 mt-1">{error}</div>}
      <div className="flex items-center justify-end gap-2 mt-2">
        <button
          type="button"
          onClick={onCancel}
          disabled={saving}
          className="text-xs px-2 py-1 border border-slate-300 rounded hover:bg-slate-100 disabled:opacity-50"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={onSave}
          disabled={saving}
          className="text-xs px-3 py-1 bg-slate-900 text-white rounded disabled:opacity-50"
        >
          {saving ? "Saving..." : saveLabel}
        </button>
      </div>
    </div>
  );
}
