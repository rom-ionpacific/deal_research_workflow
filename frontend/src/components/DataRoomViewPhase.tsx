import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import {
  api,
  type DataRoomDetail,
  type FollowupQA,
  type PresetAnswer,
  type PresetQA,
} from "../lib/api";
import { useChat } from "../stores/chat";
import Markdown from "./Markdown";

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
      // room_provider lets ChatPanel decide whether to surface the
      // Both / Claude / ToltIQ toggle (only meaningful on 'both' rooms).
      room_provider: room.data.provider,
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
  // When the user adds the OTHER provider to an existing room (or
  // when a fresh 'both' build has progressed far enough that Claude
  // answers have started landing), there's already content worth
  // showing. Detect that by looking for ANY complete answer; if
  // we find one, switch from the full-page BuildingSpinner to a
  // small inline indicator and keep the preset Q&A + chat box live.
  const hasAnyCompleteAnswer = room.data.preset_questions.some((q) =>
    q.answers.some((a) => a.answer_status === "complete"),
  );
  const showInlineBuild = isBuilding && hasAnyCompleteAnswer;
  const showFullSpinner = isBuilding && !hasAnyCompleteAnswer;

  return (
    <div>
      <h2 className="text-lg font-semibold mb-1">
        Historical data room — answers
      </h2>
      <p className="text-sm text-slate-500 mb-3">
        {room.data.name}. Status:{" "}
        <code className="text-xs bg-slate-100 px-1 rounded">
          {room.data.status}
        </code>
        {room.data.toltiq_deal_id && (
          <>
            {" · "}
            <a
              href={`https://ui.diligentiq.io/deals/${room.data.toltiq_deal_id}/intelligence`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-blue-600 hover:underline"
            >
              Open in ToltIQ ↗
            </a>
          </>
        )}
      </p>

      {showInlineBuild && <InlineBuildIndicator room={room.data} />}

      {/* Add-the-other-provider call-to-action. Lives ABOVE the
          build conditional so it's reachable even during a fresh
          toltiq-only build (the most common state where a user
          realises mid-build they wanted Claude too) -- Claude
          doesn't need ToltIQ ingest, it queries pgvector, so the
          add-claude path works the moment the room is created. */}
      {(room.data.provider === "toltiq" ||
        room.data.provider === "claude") && (
        <AddProviderBanner
          roomId={room.data.id}
          currentProvider={room.data.provider}
        />
      )}

      {showFullSpinner ? (
        <BuildingSpinner room={room.data} />
      ) : isFailed ? (
        <BuildFailedNotice room={room.data} />
      ) : (
        <>
          <PresetAnswersSection
            roomId={room.data.id}
            presets={room.data.preset_questions}
          />
          {/* Direct ToltIQ chat: posts straight to the deal. The
              followups list (the "chat history") is rendered inside
              this section so it reads like a conversation rather than
              getting mixed in with the preset Q&A above. */}
          <DirectToltIQChat
            roomId={room.data.id}
            provider={room.data.provider}
            roomStatus={room.data.status}
            followups={room.data.followup_questions}
          />
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

function DirectToltIQChat({
  roomId,
  provider,
  roomStatus,
  followups,
}: {
  roomId: number;
  provider: "toltiq" | "claude" | "both";
  roomStatus: string;
  followups: FollowupQA[];
}) {
  const qc = useQueryClient();
  const [draft, setDraft] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Which ask buttons make sense: matches what the server accepts.
  // ToltIQ-only rooms have no Claude path; Claude-only rooms have no
  // ToltIQ deal. 'both' rooms expose both buttons.
  const showToltiq = provider === "toltiq" || provider === "both";
  const showClaude = provider === "claude" || provider === "both";
  // ToltIQ's ad-hoc /ask requires the room to be 'complete' (deal_id
  // populated + entities ingested). Claude works regardless of room
  // status since it queries our pgvector retrieval directly. Disable
  // the ToltIQ button while the ToltIQ pipeline is still running so
  // the user doesn't burn a click on a guaranteed 409.
  const toltiqReady = roomStatus === "complete";

  const submitToltiq = async () => {
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
                provider: "toltiq",
              },
            ],
          };
        },
      );
      setDraft("");
      void qc.invalidateQueries({ queryKey: ["data-room", roomId] });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const submitClaude = async () => {
    const text = draft.trim();
    if (!text || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      // Optimistic 'running' row keyed by a fresh negative id so the
      // optimistic-vs-canonical comparison is unambiguous (real
      // answer_ids are positive autoincrementing ints). The synchronous
      // /ask-claude returns the full answer inline; we invalidate to
      // pick up the canonical row.
      const placeholderId = -Date.now();
      qc.setQueryData<DataRoomDetail | undefined>(
        ["data-room", roomId],
        (old) => {
          if (!old) return old;
          return {
            ...old,
            followup_questions: [
              ...old.followup_questions,
              {
                answer_id: placeholderId,
                question_text: text,
                status: "running",
                answer_text: null,
                attachments: null,
                error_message: null,
                created_at: new Date().toISOString(),
                completed_at: null,
                provider: "claude",
              },
            ],
          };
        },
      );
      const submittedDraft = text;
      setDraft("");
      try {
        await api.askDataRoomClaude(roomId, submittedDraft);
      } finally {
        // Either success or error: refetch so canonical state wins. On
        // error the placeholder is dropped (the canonical row was
        // marked 'failed' server-side or never created if route 4xx'd).
        void qc.invalidateQueries({ queryKey: ["data-room", roomId] });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className="mt-4 border border-slate-200 rounded-md">
      <div className="px-4 py-2 bg-slate-50 border-b border-slate-200 text-sm font-semibold text-slate-700">
        Ask the data room
        {followups.length > 0 && (
          <span className="ml-2 text-xs font-normal text-slate-500">
            ({followups.length} question{followups.length === 1 ? "" : "s"})
          </span>
        )}
      </div>

      {/* Conversation history -- past ad-hoc questions and their
          answers. Empty on a fresh room; grows as the user asks. */}
      {followups.length > 0 && (
        <div className="divide-y divide-slate-200">
          {followups.map((f) => (
            <FollowupRow key={f.answer_id} f={f} roomId={roomId} />
          ))}
        </div>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          // Enter submits to whichever provider the room was built
          // for. ToltIQ takes precedence on 'both' rooms (preserves
          // the prior default keyboard behavior). Alt/Shift+Enter on
          // 'both' rooms routes to Claude.
          if (showToltiq) void submitToltiq();
          else if (showClaude) void submitClaude();
        }}
        className={
          "p-3 " +
          (followups.length > 0 ? "border-t border-slate-200" : "")
        }
      >
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              if (showToltiq) void submitToltiq();
              else if (showClaude) void submitClaude();
            } else if (e.key === "Enter" && (e.altKey || e.shiftKey)) {
              e.preventDefault();
              if (showClaude) void submitClaude();
              else if (showToltiq) void submitToltiq();
            }
          }}
          placeholder={
            provider === "both"
              ? "Ask anything. Send to ToltIQ (~30-90s, page-cited) or Claude (~5s, doc-cited)."
              : provider === "claude"
                ? "Ask anything. Claude answers in ~5s with inline doc-id citations."
                : "Ask anything. ToltIQ answers in ~30-90s with page-cited responses."
          }
          maxLength={2000}
          rows={3}
          disabled={submitting}
          className="w-full text-sm border border-slate-300 rounded-md px-3 py-2 focus:outline-none focus:border-slate-500"
        />
        <div className="flex items-center justify-between mt-2 gap-2">
          <div className="text-xs text-slate-500">
            {provider === "both"
              ? "ToltIQ ~30–90s · Claude ~5s. Both answer from the same curated docs."
              : provider === "claude"
                ? "Claude over our local pgvector retrieval. ~5s per question."
                : "ToltIQ workflow over the uploaded entities. ~30–90s per question."}
          </div>
          <div className="flex gap-2">
            {showClaude && (
              <button
                type="button"
                onClick={() => void submitClaude()}
                disabled={submitting || !draft.trim()}
                title="Faster answer via Claude over local retrieval"
                className="px-3 py-1.5 bg-violet-700 text-white text-sm rounded-md disabled:opacity-50 hover:bg-violet-800"
              >
                {submitting ? "..." : "Ask Claude"}
              </button>
            )}
            {showToltiq && (
              <button
                type="submit"
                disabled={submitting || !draft.trim() || !toltiqReady}
                title={
                  toltiqReady
                    ? undefined
                    : "ToltIQ side is still building; ask Claude or wait."
                }
                className="px-3 py-1.5 bg-slate-900 text-white text-sm rounded-md disabled:opacity-50"
              >
                {submitting
                  ? "Sending..."
                  : toltiqReady
                    ? "Ask ToltIQ"
                    : "ToltIQ building…"}
              </button>
            )}
          </div>
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

function InlineBuildIndicator({ room }: { room: DataRoomDetail }) {
  // Used when the room has SOME complete answers (typically the
  // already-built provider's column) and the OTHER provider's build
  // is still running. Compact banner at the top of the content area;
  // leaves the preset Q&A and chat box live underneath.
  const status = room.status;
  const label = PROGRESS_LABELS[status] ?? status;
  const ent = room.entity_progress;
  const total = Object.values(ent).reduce((a, b) => a + b, 0);
  const uploaded = ent.uploaded ?? 0;
  // Status drives the ToltIQ pipeline today; if it's non-terminal we
  // know ToltIQ is the side that's still building (claude finishes
  // before status flips). Phrase the message accordingly.
  return (
    <div className="mt-3 mb-2 flex items-center gap-3 px-3 py-2 bg-amber-50 border border-amber-200 rounded-md">
      <SmallSpinner />
      <div className="flex-1 text-xs text-amber-900">
        <span className="font-medium">Building ToltIQ side: {label}.</span>
        {total > 0 && status === "uploading" && (
          <>
            {" "}
            Entities uploaded: {uploaded.toLocaleString()} /{" "}
            {total.toLocaleString()}.
          </>
        )}
        {" "}
        You can keep querying the existing answers below in the meantime.
      </div>
    </div>
  );
}

function SmallSpinner() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      className="text-amber-700 animate-spin shrink-0"
    >
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
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
  const qc = useQueryClient();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const retry = async () => {
    if (submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      await api.retryDataRoomBuild(room.id);
      // Optimistic: flip status to 'pending' so the failure UI swaps
      // out for the building spinner immediately. Polling resumes via
      // refetchInterval and catches up to canonical state.
      qc.setQueryData<DataRoomDetail | undefined>(
        ["data-room", room.id],
        (old) =>
          old
            ? {
                ...old,
                status: "pending",
                error_message: null,
                started_at: null,
                completed_at: null,
              }
            : old,
      );
      void qc.invalidateQueries({ queryKey: ["data-room", room.id] });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="mt-4 border border-red-200 rounded-md bg-red-50 p-4">
      <div className="text-sm font-medium text-red-700">Build failed</div>
      {room.error_message && (
        <pre className="mt-2 text-xs text-red-700 whitespace-pre-wrap font-mono">
          {room.error_message}
        </pre>
      )}
      <div className="mt-3 flex items-center gap-3">
        <button
          type="button"
          onClick={retry}
          disabled={submitting}
          className="text-xs px-3 py-1.5 bg-red-700 text-white rounded-md hover:bg-red-800 disabled:opacity-50"
        >
          {submitting ? "Retrying…" : "Retry build"}
        </button>
        <div className="text-xs text-red-700">
          Retry re-claims this room without changing the entity or
          question selection. Already-uploaded entities skip. If it
          keeps failing, rebuild from Phase 3 or ask the assistant for
          help triaging.
        </div>
      </div>
      {error && (
        <div className="mt-2 text-xs text-red-700">
          Retry request failed: {error}
        </div>
      )}
    </div>
  );
}

function AddProviderBanner({
  roomId,
  currentProvider,
}: {
  roomId: number;
  currentProvider: "toltiq" | "claude";
}) {
  const qc = useQueryClient();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const adding = currentProvider === "toltiq" ? "claude" : "toltiq";
  const addingLabel = adding === "claude" ? "Claude" : "ToltIQ";
  const currentLabel = currentProvider === "claude" ? "Claude" : "ToltIQ";

  const onClick = async () => {
    if (submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      await api.addDataRoomProvider(roomId, adding);
      // Optimistic flip: room.provider becomes 'both' so the
      // PresetAnswersSection injects pending placeholders for the
      // new provider's column, and the chat box's button-gate
      // unhides the now-available Ask button. Polling takes over
      // from here -- ToltIQ-add resets status to 'pending' so the
      // BuildingSpinner replaces this section until the cron
      // finishes; Claude-add stays on 'complete' status and the
      // BackgroundTask fills in the new answer rows row-by-row.
      qc.setQueryData<DataRoomDetail | undefined>(
        ["data-room", roomId],
        (old) => {
          if (!old) return old;
          return {
            ...old,
            provider: "both",
            // For ToltIQ-add the server reset status to pending;
            // for Claude-add we keep the existing status.
            status: adding === "toltiq" ? "pending" : old.status,
          };
        },
      );
      void qc.invalidateQueries({ queryKey: ["data-room", roomId] });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="mt-3 border border-violet-200 bg-violet-50/50 rounded-md px-4 py-3">
      <div className="flex items-start gap-3">
        <div className="flex-1 text-sm text-slate-700">
          <span className="font-medium">
            Built with {currentLabel} only.
          </span>{" "}
          {adding === "claude" ? (
            <>
              Add {addingLabel} answers (~1–2 min) to get side-by-side
              answers for the same preset questions. Existing
              {" "}{currentLabel} answers stay intact.
            </>
          ) : (
            <>
              Add {addingLabel} answers (~10–15 min). The cron will
              upload the room's entities to ToltIQ and run the preset
              playlist. Existing {currentLabel} answers stay intact.
            </>
          )}
        </div>
        <button
          type="button"
          onClick={onClick}
          disabled={submitting}
          className="shrink-0 px-3 py-1.5 bg-violet-700 text-white text-sm rounded-md hover:bg-violet-800 disabled:opacity-50"
        >
          {submitting ? "Starting…" : `+ Add ${addingLabel}`}
        </button>
      </div>
      {error && (
        <div className="mt-2 text-xs text-red-600">
          Couldn't add {addingLabel}: {error}
        </div>
      )}
    </div>
  );
}

function PresetAnswersSection({
  roomId,
  presets,
}: {
  roomId: number;
  presets: PresetQA[];
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
        </div>
        <Chevron open={!collapsed} />
      </button>
      {!collapsed && (
        <div className="divide-y divide-slate-200">
          {presets.map((q) => (
            <PresetRow key={q.preset_question_id} q={q} roomId={roomId} />
          ))}
        </div>
      )}
    </section>
  );
}

function PresetRow({ q, roomId }: { q: PresetQA; roomId: number }) {
  // Aggregate status for the header row: 'complete' iff any answer
  // has landed (gives the user something useful to click open even
  // mid-build); 'failed' iff all are failed; otherwise 'pending'.
  const anyComplete = q.answers.some((a) => a.answer_status === "complete");
  const allFailed =
    q.answers.length > 0 && q.answers.every((a) => a.answer_status === "failed");
  const aggregateStatus = anyComplete
    ? "complete"
    : allFailed
      ? "failed"
      : "pending";

  const [open, setOpen] = useState(aggregateStatus === "complete");
  useEffect(() => {
    if (aggregateStatus === "complete") setOpen(true);
  }, [aggregateStatus]);

  const statusBadge =
    aggregateStatus === "complete" ? null : (
      <span
        className={
          "ml-2 text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded " +
          (aggregateStatus === "failed"
            ? "bg-red-100 text-red-700"
            : "bg-amber-100 text-amber-700")
        }
      >
        {aggregateStatus}
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
          {q.answers.length > 1 ? (
            <PresetAnswerTabs answers={q.answers} roomId={roomId} />
          ) : q.answers.length === 1 ? (
            <PresetAnswerView
              answer={q.answers[0]}
              roomId={roomId}
              singleProvider={true}
            />
          ) : null}
        </div>
      )}
    </div>
  );
}

function PresetAnswerTabs({
  answers,
  roomId,
}: {
  answers: PresetAnswer[];
  roomId: number;
}) {
  // Default tab: prefer a complete answer; if multiple complete, prefer
  // ToltIQ for continuity with the established baseline. If none
  // complete, fall back to the first answer in the list.
  const defaultProvider = useMemo(() => {
    const completeT = answers.find(
      (a) => a.provider === "toltiq" && a.answer_status === "complete",
    );
    if (completeT) return "toltiq" as const;
    const completeC = answers.find(
      (a) => a.provider === "claude" && a.answer_status === "complete",
    );
    if (completeC) return "claude" as const;
    return answers[0].provider;
  }, [answers]);

  const [active, setActive] = useState<"toltiq" | "claude">(defaultProvider);

  // If a new answer transitions to 'complete' (e.g. the user clicked
  // expand mid-build and Claude just landed), prefer auto-switching
  // the active tab to the freshly-arrived provider. We only do this
  // if the currently active tab is NOT complete, so it doesn't yank
  // the user out of an answer they're reading.
  useEffect(() => {
    const current = answers.find((a) => a.provider === active);
    if (current && current.answer_status === "complete") return;
    const fresh = answers.find((a) => a.answer_status === "complete");
    if (fresh && fresh.provider !== active) setActive(fresh.provider);
  }, [answers, active]);

  // Render the tabs in a stable order (toltiq first, then claude) so
  // the tab strip doesn't reflow when answers arrive in different
  // orders across renders.
  const ordered = useMemo(() => {
    const out: PresetAnswer[] = [];
    const t = answers.find((a) => a.provider === "toltiq");
    const c = answers.find((a) => a.provider === "claude");
    if (t) out.push(t);
    if (c) out.push(c);
    return out;
  }, [answers]);

  const activeAnswer =
    ordered.find((a) => a.provider === active) ?? ordered[0];

  return (
    <div>
      <div className="flex items-center gap-1 border-b border-slate-200 mb-3">
        {ordered.map((a) => {
          const isActive = a.provider === active;
          const label = a.provider === "claude" ? "Claude" : "ToltIQ";
          const statusGlyph =
            a.answer_status === "complete"
              ? "✓"
              : a.answer_status === "failed"
                ? "✗"
                : "…";
          const statusColor =
            a.answer_status === "complete"
              ? "text-emerald-600"
              : a.answer_status === "failed"
                ? "text-red-600"
                : "text-amber-600";
          return (
            <button
              key={a.provider}
              type="button"
              onClick={() => setActive(a.provider)}
              className={
                "px-3 py-1.5 text-xs border-b-2 -mb-px transition-colors " +
                (isActive
                  ? "border-slate-900 text-slate-900 font-medium"
                  : "border-transparent text-slate-500 hover:text-slate-700")
              }
            >
              {label}{" "}
              <span className={statusColor + " ml-0.5"}>{statusGlyph}</span>
            </button>
          );
        })}
      </div>
      <PresetAnswerView
        key={`${activeAnswer.provider}:${activeAnswer.answer_id ?? "pending"}`}
        answer={activeAnswer}
        roomId={roomId}
        singleProvider={true /* hide the provider chip inside; tab strip already shows it */}
      />
    </div>
  );
}

function PresetAnswerView({
  answer,
  roomId,
  singleProvider,
}: {
  answer: PresetAnswer;
  roomId: number;
  singleProvider: boolean;
}) {
  const providerLabel =
    answer.provider === "claude" ? "Claude" : "ToltIQ";
  const providerColor =
    answer.provider === "claude"
      ? "bg-indigo-100 text-indigo-800"
      : "bg-emerald-100 text-emerald-800";

  return (
    <div
      className={
        singleProvider
          ? "" // visually identical to the prior single-answer layout
          : "border border-slate-200 rounded-md p-3 bg-slate-50"
      }
    >
      {!singleProvider && (
        <div className="flex items-center gap-2 mb-2">
          <span
            className={
              "text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded " +
              providerColor
            }
          >
            {providerLabel}
          </span>
          {answer.answer_status !== "complete" && (
            <span
              className={
                "text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded " +
                (answer.answer_status === "failed"
                  ? "bg-red-100 text-red-700"
                  : "bg-amber-100 text-amber-700")
              }
            >
              {answer.answer_status}
            </span>
          )}
        </div>
      )}
      {answer.answer_status === "complete" && answer.answer_text && (
        <div className="text-sm text-slate-800">
          <Markdown>{answer.answer_text}</Markdown>
        </div>
      )}
      {answer.answer_status === "failed" && (
        <div>
          <div className="text-sm text-red-700 whitespace-pre-wrap">
            {answer.answer_error ||
              `${providerLabel} returned an error for this question.`}
          </div>
          {answer.answer_id != null && (
            <RetryButton roomId={roomId} answerId={answer.answer_id} />
          )}
        </div>
      )}
      {(answer.answer_status === "pending" ||
        answer.answer_status === "running") && (
        <div className="text-sm text-slate-500 italic">
          Waiting for {providerLabel}…
        </div>
      )}
      <AttachmentsList attachments={answer.attachments} />
    </div>
  );
}

function FollowupRow({ f, roomId }: { f: FollowupQA; roomId: number }) {
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
            <span className="text-[10px] uppercase tracking-wide bg-violet-100 text-violet-800 px-1.5 py-0.5 rounded mr-1">
              follow-up
            </span>
            <span
              className={
                "text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded mr-2 " +
                (f.provider === "claude"
                  ? "bg-indigo-100 text-indigo-800"
                  : "bg-emerald-100 text-emerald-800")
              }
            >
              {f.provider === "claude" ? "claude" : "toltiq"}
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
            <div className="text-sm text-slate-800">
              <Markdown>{f.answer_text}</Markdown>
            </div>
          )}
          {f.status === "failed" && (
            <div>
              <div className="text-sm text-red-700 whitespace-pre-wrap">
                {f.error_message ||
                  `${f.provider === "claude" ? "Claude" : "ToltIQ"} returned an error.`}
              </div>
              <RetryButton roomId={roomId} answerId={f.answer_id} />
            </div>
          )}
          {(f.status === "pending" || f.status === "running") && (
            <div className="text-sm text-slate-500 italic">
              Waiting for {f.provider === "claude" ? "Claude" : "ToltIQ"}…
            </div>
          )}
          <AttachmentsList attachments={f.attachments} />
        </div>
      )}
    </div>
  );
}

function RetryButton({
  roomId,
  answerId,
}: {
  roomId: number;
  answerId: number;
}) {
  const qc = useQueryClient();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const retry = async () => {
    if (submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      await api.retryDataRoomAnswer(roomId, answerId);
      // Optimistic: flip the row's status to 'running' locally so the
      // failure UI disappears immediately; canonical state will catch
      // up on the next poll tick.
      qc.setQueryData<DataRoomDetail | undefined>(
        ["data-room", roomId],
        (old) => {
          if (!old) return old;
          return {
            ...old,
            preset_questions: old.preset_questions.map((p) => ({
              ...p,
              answers: p.answers.map((a) =>
                a.answer_id === answerId
                  ? { ...a, answer_status: "running", answer_error: null }
                  : a,
              ),
            })),
            followup_questions: old.followup_questions.map((f) =>
              f.answer_id === answerId
                ? {
                    ...f,
                    status: "running",
                    error_message: null,
                    answer_text: null,
                  }
                : f,
            ),
          };
        },
      );
      void qc.invalidateQueries({ queryKey: ["data-room", roomId] });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={retry}
        disabled={submitting}
        className="text-xs px-2 py-1 border border-slate-300 rounded-md bg-white hover:bg-slate-50 disabled:opacity-50"
      >
        {submitting ? "Retrying…" : "Retry"}
      </button>
      {error && (
        <div className="mt-1 text-xs text-red-600">Retry failed: {error}</div>
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
