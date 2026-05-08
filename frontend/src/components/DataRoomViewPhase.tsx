import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import {
  api,
  type DataRoomDetail,
  type FollowupQA,
  type PresetQA,
} from "../lib/api";
import { useChat } from "../stores/chat";

const TERMINAL_STATUSES = new Set(["complete", "failed"]);
const PROGRESS_LABELS: Record<string, string> = {
  pending: "Queued",
  uploading: "Uploading entities to ToltIQ",
  extracting: "Waiting for ToltIQ to ingest documents",
  querying: "Running the question playlist",
  complete: "Build complete",
  failed: "Build failed",
};

/**
 * Phase 4 -- data_room_view. Top section: collapsible per-question
 * accordion of preset Q&A (and any ad-hoc follow-ups). Whole section
 * is also collapsible. While the room is still building, the section
 * is replaced by a spinner + status copy. The chat panel (sibling on
 * the page) is always available and answers from local sources
 * pre-build, with ToltIQ tools added post-build.
 */
export default function DataRoomViewPhase({
  sessionId,
  state,
}: {
  sessionId: string;
  state: Record<string, unknown>;
}) {
  const dataRoomId =
    typeof state.data_room_id === "number"
      ? (state.data_room_id as number)
      : null;

  const room = useQuery({
    queryKey: ["data-room", dataRoomId],
    queryFn: () => api.getDataRoom(dataRoomId!),
    enabled: dataRoomId != null,
    // Poll cadence:
    //   * Build still in flight: 15s (entity progress changes slowly).
    //   * Build done but a follow-up is running: 5s (ToltIQ workflows
    //     finish in 30-90s; we want the new answer to land soon).
    //   * Otherwise: stop polling -- nothing will change.
    refetchInterval: (q) => {
      const data = q.state.data as DataRoomDetail | undefined;
      if (!data) return false;
      const status = data.status;
      if (!TERMINAL_STATUSES.has(status)) return 15_000;
      const hasRunningFollowup = data.followup_questions.some(
        (f) => f.status === "running" || f.status === "pending",
      );
      if (hasRunningFollowup) return 5_000;
      return false;
    },
  });

  // Publish UI context for the Phase 4 chat. The orchestrator surfaces
  // this in the system prompt so the model knows whether ToltIQ is
  // available, how many preset Qs exist, etc.
  const setUIContext = useChat((s) => s.setUIContext);
  useEffect(() => {
    if (!room.data) {
      setUIContext(sessionId, null);
      return;
    }
    setUIContext(sessionId, {
      phase: "data_room_view",
      data_room_id: room.data.id,
      status: room.data.status,
      selected_org_ids: [room.data.main_organization_id],
      entity_progress: room.data.entity_progress,
      preset_question_count: room.data.preset_questions.length,
      followup_question_count: room.data.followup_questions.length,
    });
  }, [sessionId, setUIContext, room.data]);

  if (dataRoomId == null) {
    return (
      <div className="mt-4 text-sm text-slate-500">
        No data_room_id on this session. (Did the build step complete?)
      </div>
    );
  }
  if (room.isLoading) {
    return <div className="mt-4 text-sm text-slate-500">Loading data room…</div>;
  }
  if (room.error) {
    return (
      <div className="mt-4 text-sm text-red-600">
        {(room.error as Error).message}
      </div>
    );
  }
  if (!room.data) return null;

  const isBuilding = !TERMINAL_STATUSES.has(room.data.status);
  const isFailed = room.data.status === "failed";

  return (
    <div>
      <h2 className="text-lg font-semibold mb-1">Phase 4 — Data room view</h2>
      <p className="text-sm text-slate-500 mb-3">
        {room.data.name}. Status:{" "}
        <code className="text-xs bg-slate-100 px-1 rounded">
          {room.data.status}
        </code>
      </p>

      {isBuilding ? (
        <BuildingSpinner room={room.data} />
      ) : isFailed ? (
        <BuildFailedNotice room={room.data} />
      ) : (
        <>
          <PresetAnswersSection
            presets={room.data.preset_questions}
            followups={room.data.followup_questions}
          />
          {/* Direct ToltIQ chat: posts straight to the deal, the
              answer lands in the followups list above when ready. */}
          <DirectToltIQChat roomId={room.data.id} />
        </>
      )}

      {/* The AI Assistant on the right is also useful: it can search
          the org dossier, read full document summaries, and (post-
          build) decide whether a question needs ToltIQ at all. */}
      <div className="mt-6 pt-4 border-t border-slate-200 text-xs text-slate-500">
        Tip: the assistant on the right
        {isBuilding
          ? " can answer from local document summaries and the org dossier while the room finishes building."
          : " can also reason over the local dossier and decide when ToltIQ is the right call."}
      </div>
    </div>
  );
}

function DirectToltIQChat({ roomId }: { roomId: number }) {
  const qc = useQueryClient();
  const [draft, setDraft] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    const text = draft.trim();
    if (!text || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const resp = await api.askDataRoom(roomId, text);
      // Optimistically add a 'running' follow-up so the user sees the
      // question immediately instead of waiting for the next poll
      // tick. The polling refetch will overwrite this with canonical
      // data shortly.
      qc.setQueryData<DataRoomDetail | undefined>(
        ["data-room", roomId],
        (old) => {
          if (!old) return old;
          return {
            ...old,
            followup_questions: [
              ...old.followup_questions,
              {
                answer_id: resp.answer_id,
                question_text: text,
                status: "running",
                answer_text: null,
                attachments: null,
                error_message: null,
                created_at: new Date().toISOString(),
                completed_at: null,
              },
            ],
          };
        },
      );
      setDraft("");
      // Trigger a near-immediate refetch so the running -> complete
      // transition is picked up promptly.
      void qc.invalidateQueries({ queryKey: ["data-room", roomId] });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className="mt-4 border border-slate-200 rounded-md">
      <div className="px-4 py-2 bg-slate-50 border-b border-slate-200 text-sm font-semibold text-slate-700">
        Ask the data room (ToltIQ)
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
        className="p-3"
      >
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              void submit();
            }
          }}
          placeholder="Ask anything about this room — the answer is generated by ToltIQ over the uploaded entities. Cmd/Ctrl+Enter to send."
          maxLength={2000}
          rows={3}
          disabled={submitting}
          className="w-full text-sm border border-slate-300 rounded-md px-3 py-2 focus:outline-none focus:border-slate-500"
        />
        <div className="flex items-center justify-between mt-2">
          <div className="text-xs text-slate-500">
            ToltIQ workflows take 30-90s. The answer will appear above
            once it's ready.
          </div>
          <button
            type="submit"
            disabled={submitting || !draft.trim()}
            className="px-3 py-1.5 bg-slate-900 text-white text-sm rounded-md disabled:opacity-50"
          >
            {submitting ? "Sending..." : "Ask ToltIQ"}
          </button>
        </div>
        {error && (
          <div className="mt-2 text-xs text-red-600">
            Failed to send: {error}
          </div>
        )}
      </form>
    </section>
  );
}

function BuildingSpinner({ room }: { room: DataRoomDetail }) {
  const status = room.status;
  const label = PROGRESS_LABELS[status] ?? status;
  const ent = room.entity_progress;
  const totalEntities = Object.values(ent).reduce((a, b) => a + b, 0);
  const uploaded = ent.uploaded ?? 0;
  const failed = ent.failed ?? 0;

  return (
    <div className="mt-4 border border-slate-200 rounded-md bg-slate-50 p-6 flex flex-col items-center text-center">
      <Spinner />
      <div className="mt-3 text-sm font-medium text-slate-800">{label}</div>
      <div className="mt-1 text-xs text-slate-500">
        Hang tight — typically 10–15 minutes end to end. The page will refresh
        automatically when the build completes.
      </div>
      {totalEntities > 0 && (
        <div className="mt-4 w-full max-w-sm">
          <div className="flex justify-between text-xs text-slate-600 mb-1">
            <span>
              Entities uploaded: {uploaded.toLocaleString()} /{" "}
              {totalEntities.toLocaleString()}
            </span>
            {failed > 0 && (
              <span className="text-red-600">{failed} failed</span>
            )}
          </div>
          <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-slate-900 transition-all"
              style={{
                width: `${Math.min(100, (uploaded / totalEntities) * 100)}%`,
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function Spinner() {
  return (
    <svg
      width="32"
      height="32"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      className="text-slate-700 animate-spin"
    >
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
  );
}

function BuildFailedNotice({ room }: { room: DataRoomDetail }) {
  return (
    <div className="mt-4 border border-red-200 rounded-md bg-red-50 p-4">
      <div className="text-sm font-medium text-red-700">
        Build failed
      </div>
      {room.error_message && (
        <pre className="mt-2 text-xs text-red-700 whitespace-pre-wrap font-mono">
          {room.error_message}
        </pre>
      )}
      <div className="mt-2 text-xs text-red-700">
        You can rebuild from Phase 3 with a different selection or
        question plan, or ask the assistant on the right for help
        triaging.
      </div>
    </div>
  );
}

function PresetAnswersSection({
  presets,
  followups,
}: {
  presets: PresetQA[];
  followups: FollowupQA[];
}) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <section className="mt-2 border border-slate-200 rounded-md">
      <button
        type="button"
        onClick={() => setCollapsed((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-2 bg-slate-50 hover:bg-slate-100 rounded-t-md"
      >
        <div className="text-sm font-semibold text-slate-700">
          Preset questions ({presets.length})
          {followups.length > 0 && (
            <span className="ml-2 text-xs font-normal text-slate-500">
              + {followups.length} follow-up
              {followups.length === 1 ? "" : "s"}
            </span>
          )}
        </div>
        <Chevron open={!collapsed} />
      </button>
      {!collapsed && (
        <div className="divide-y divide-slate-200">
          {presets.map((q) => (
            <PresetRow key={q.preset_question_id} q={q} />
          ))}
          {followups.map((f) => (
            <FollowupRow key={f.answer_id} f={f} />
          ))}
        </div>
      )}
    </section>
  );
}

function PresetRow({ q }: { q: PresetQA }) {
  // Default open if there's an answer; default collapsed for pending/
  // failed so the user isn't pre-overwhelmed.
  const [open, setOpen] = useState(q.answer_status === "complete");

  const statusBadge =
    q.answer_status === "complete" ? null : (
      <span
        className={
          "ml-2 text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded " +
          (q.answer_status === "failed"
            ? "bg-red-100 text-red-700"
            : "bg-amber-100 text-amber-700")
        }
      >
        {q.answer_status}
      </span>
    );

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-start justify-between gap-3 px-4 py-3 hover:bg-slate-50 text-left"
      >
        <div className="min-w-0">
          <div className="text-sm font-medium truncate">
            {q.label}
            {statusBadge}
          </div>
          <div className="text-xs text-slate-500 truncate">
            {q.question_text}
          </div>
        </div>
        <Chevron open={open} />
      </button>
      {open && (
        <div className="px-4 pb-4 -mt-1">
          {q.answer_status === "complete" && q.answer_text && (
            <div className="text-sm text-slate-800 whitespace-pre-wrap leading-relaxed">
              {q.answer_text}
            </div>
          )}
          {q.answer_status === "failed" && (
            <div className="text-sm text-red-700 whitespace-pre-wrap">
              {q.answer_error || "ToltIQ returned an error for this question."}
            </div>
          )}
          {(q.answer_status === "pending" || q.answer_status === "running") && (
            <div className="text-sm text-slate-500 italic">
              Waiting for ToltIQ…
            </div>
          )}
          <AttachmentsList attachments={q.attachments} />
        </div>
      )}
    </div>
  );
}

function FollowupRow({ f }: { f: FollowupQA }) {
  // Initial: complete means open, anything else means collapsed.
  // BUT: the row first appears as 'running' (just-asked question) and
  // later transitions to 'complete' on the next poll. useState only
  // captures the initial value, so without the effect below the row
  // stays collapsed and the user thinks the answer never arrived.
  const [open, setOpen] = useState(f.status === "complete");
  useEffect(() => {
    if (f.status === "complete") setOpen(true);
  }, [f.status]);
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-start justify-between gap-3 px-4 py-3 hover:bg-slate-50 text-left"
      >
        <div className="min-w-0">
          <div className="text-sm font-medium truncate">
            <span className="text-[10px] uppercase tracking-wide bg-violet-100 text-violet-800 px-1.5 py-0.5 rounded mr-2">
              follow-up
            </span>
            {f.question_text}
          </div>
          <div className="text-xs text-slate-500">
            {new Date(f.created_at).toLocaleString()}
            {" · "}
            <code className="text-[11px]">{f.status}</code>
          </div>
        </div>
        <Chevron open={open} />
      </button>
      {open && (
        <div className="px-4 pb-4 -mt-1">
          {f.status === "complete" && f.answer_text && (
            <div className="text-sm text-slate-800 whitespace-pre-wrap leading-relaxed">
              {f.answer_text}
            </div>
          )}
          {f.status === "failed" && (
            <div className="text-sm text-red-700 whitespace-pre-wrap">
              {f.error_message || "ToltIQ returned an error."}
            </div>
          )}
          {(f.status === "pending" || f.status === "running") && (
            <div className="text-sm text-slate-500 italic">
              Waiting for ToltIQ…
            </div>
          )}
          <AttachmentsList attachments={f.attachments} />
        </div>
      )}
    </div>
  );
}

function AttachmentsList({ attachments }: { attachments: unknown }) {
  if (!attachments || !Array.isArray(attachments) || attachments.length === 0) {
    return null;
  }
  return (
    <div className="mt-2 text-xs text-slate-500">
      <div className="font-semibold mb-1">Attachments:</div>
      <ul className="list-disc list-inside space-y-0.5">
        {(attachments as Array<Record<string, unknown>>).map((a, i) => (
          <li key={i}>
            {(a.name as string | undefined) ?? `attachment ${i + 1}`}
          </li>
        ))}
      </ul>
    </div>
  );
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={
        "shrink-0 text-slate-400 transition-transform " +
        (open ? "rotate-180" : "")
      }
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}
