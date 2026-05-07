import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";

import {
  api,
  type PresetQuestion,
  type SessionWithCurrent,
} from "../lib/api";

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
 * V0 supports preset questions only. Custom questions need a schema
 * change in deal_cloud_enhancer (historical_data_room_question only
 * has preset_question_id today) -- deferred.
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

  const [pendingIds, setPendingIds] = useState<Set<number>>(new Set());
  const queueRef = useRef<Promise<void>>(Promise.resolve());

  const toggle = (questionId: number) => {
    const cached = qc.getQueryData<SessionWithCurrent>(["session", sessionId]);
    if (!cached) return;
    const cur = cached.current_version.state as PhaseState;
    const curIds = cur.preset_question_ids ?? [];
    const isSelected = curIds.includes(questionId);
    const nextIds = isSelected
      ? curIds.filter((x) => x !== questionId)
      : [...curIds, questionId];

    // Optimistic patch.
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

    setPendingIds((prev) => new Set(prev).add(questionId));

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
            phase: "data_room_setup",
            state: {
              ...(cur as Record<string, unknown>),
              preset_question_ids: nextIds,
            },
            summary: isSelected
              ? `Removed preset question ${questionId}`
              : `Added preset question ${questionId}`,
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
                (old.current_version.state as PhaseState).preset_question_ids ??
                [];
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
          console.error("question toggle failed", err);
          qc.invalidateQueries({ queryKey: ["session", sessionId] });
        } finally {
          setPendingIds((prev) => {
            const n = new Set(prev);
            n.delete(questionId);
            return n;
          });
        }
      });
  };

  const [building, setBuilding] = useState(false);
  const [buildError, setBuildError] = useState<string | null>(null);

  const onBuild = async () => {
    setBuilding(true);
    setBuildError(null);
    try {
      const resp = await api.buildDataRoom(sessionId);
      // Build advances the session to data_room_view server-side; we
      // refetch the session so the page re-renders with the new state.
      await qc.invalidateQueries({ queryKey: ["session", sessionId] });
      // Light feedback in case the placeholder view doesn't catch the
      // transition yet.
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
        Pick which preset questions the AI should answer using your
        selection. {totalEntities.toLocaleString()} entities scoped from
        Phase 2. Click Build to ship the data room to ToltIQ — the
        builder cron will upload your entities, run the playlist, and
        save answers (typically 10-15 min).
      </p>

      {presets.isLoading && (
        <div className="text-sm text-slate-500">Loading preset questions...</div>
      )}
      {presets.error && (
        <div className="text-sm text-red-600">
          {(presets.error as Error).message}
        </div>
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

      {/* Phase nav. Build is the primary action; Back returns to
          entity_select with selection preserved. */}
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
