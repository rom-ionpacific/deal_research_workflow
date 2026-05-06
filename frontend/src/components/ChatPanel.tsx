import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import {
  api,
  streamChat,
  type ChatMessage,
  type Phase,
} from "../lib/api";
import { useChat, type InFlightTurn } from "../stores/chat";

/**
 * Streaming chat panel for one session. Rendered inside ResearchPage as
 * a sidebar. Owns its own query for the persisted message log and its
 * own SSE stream for the in-flight turn.
 *
 * Persistence-vs-streaming model:
 *   - history = TanStack Query on /messages (oldest first)
 *   - in-flight = Zustand store, populated from SSE events
 * The panel renders [...history, optional in-flight turn]. When a turn
 * ends, we invalidate the history query; once the refetch resolves with
 * the new server-side messages, we clear in-flight via resetTurn() so
 * the on-screen messages don't double-render.
 *
 * On version_created, we also invalidate the parent ['session', id]
 * query so the orgs panel re-renders with the new selection.
 */
export default function ChatPanel({
  sessionId,
  phase,
  parentVersionId,
}: {
  sessionId: string;
  phase: Phase;
  parentVersionId: string;
}) {
  const qc = useQueryClient();

  const messages = useQuery({
    queryKey: ["messages", sessionId],
    queryFn: () => api.listMessages(sessionId),
    // Cheap to refetch when we know the turn just landed; otherwise
    // keep stale-while-revalidate behavior on its own.
    staleTime: 30_000,
  });

  const draft = useChat((s) => s.drafts[sessionId] ?? "");
  const setDraft = useChat((s) => s.setDraft);
  const inFlight = useChat((s) => s.inFlight[sessionId] ?? null);
  const streaming = useChat((s) => s.streaming[sessionId] ?? false);
  const startTurn = useChat((s) => s.startTurn);
  const endTurn = useChat((s) => s.endTurn);
  const resetTurn = useChat((s) => s.resetTurn);
  const patchInFlight = useChat((s) => s.patchInFlight);

  // After a turn ends, watch for the messages refetch to bring in the
  // new canonical rows. Once message count exceeds the pre-turn count
  // (i.e. the new turn's rows are visible), clear in-flight.
  const lastSeenCountRef = useRef<number>(0);
  useEffect(() => {
    const n = messages.data?.length ?? 0;
    if (!streaming && inFlight && n > lastSeenCountRef.current) {
      resetTurn(sessionId);
    }
    lastSeenCountRef.current = n;
  }, [messages.data, streaming, inFlight, sessionId, resetTurn]);

  const submit = async () => {
    if (!draft.trim() || streaming) return;
    const text = draft.trim();
    startTurn(sessionId);

    try {
      await streamChat({
        sessionId,
        phase,
        message: text,
        parentId: parentVersionId,
        onEvent: (ev) => {
          switch (ev.type) {
            case "turn_start":
              patchInFlight(sessionId, (p) => ({
                ...p,
                user_message_id: ev.user_message_id,
              }));
              break;
            case "text_delta":
              patchInFlight(sessionId, (p) => ({
                ...p,
                assistantText: p.assistantText + ev.text,
              }));
              break;
            case "thinking_delta":
              patchInFlight(sessionId, (p) => ({
                ...p,
                thinkingText: p.thinkingText + ev.text,
              }));
              break;
            case "assistant_message":
              if (ev.ai_message_id) {
                patchInFlight(sessionId, (p) => ({
                  ...p,
                  ai_message_id: ev.ai_message_id,
                }));
              }
              break;
            case "tool_call":
              patchInFlight(sessionId, (p) => ({
                ...p,
                toolCalls: [
                  ...p.toolCalls,
                  {
                    tool_use_id: ev.tool_use_id,
                    name: ev.name,
                    input: ev.input,
                  },
                ],
              }));
              break;
            case "tool_result":
              patchInFlight(sessionId, (p) => ({
                ...p,
                toolCalls: p.toolCalls.map((tc) =>
                  tc.tool_use_id === ev.tool_use_id
                    ? {
                        ...tc,
                        output: ev.output,
                        is_error: ev.is_error,
                        mutates_state: ev.mutates_state,
                      }
                    : tc
                ),
              }));
              break;
            case "version_created":
              patchInFlight(sessionId, (p) => ({
                ...p,
                toolCalls: p.toolCalls.map((tc) =>
                  // The most recent tool call without a version_id is
                  // the one that produced this -- there's no explicit
                  // pairing in the event, but tools always emit
                  // tool_result before version_created.
                  tc.version_id || tc.is_error
                    ? tc
                    : { ...tc, version_id: ev.version_id }
                ),
              }));
              // Refetch session so the orgs panel re-renders with the
              // mutated selection.
              qc.invalidateQueries({ queryKey: ["session", sessionId] });
              break;
            case "turn_done":
              endTurn(sessionId);
              qc.invalidateQueries({ queryKey: ["messages", sessionId] });
              break;
            case "turn_failed":
            case "error":
              patchInFlight(sessionId, (p) => ({
                ...p,
                error:
                  ev.type === "error"
                    ? ev.message
                    : `Turn failed: ${ev.reason}`,
              }));
              break;
          }
        },
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      patchInFlight(sessionId, (p) => ({ ...p, error: msg }));
      endTurn(sessionId);
    }
  };

  return (
    <div className="flex flex-col h-full border-l border-slate-200 bg-slate-50">
      <header className="px-4 py-2 border-b border-slate-200 bg-white">
        <h3 className="text-sm font-semibold text-slate-700">AI assistant</h3>
        <p className="text-xs text-slate-500">
          Phase: <code>{phase}</code>
        </p>
      </header>

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {messages.isLoading && (
          <div className="text-xs text-slate-500">Loading conversation...</div>
        )}
        {messages.error && (
          <div className="text-xs text-red-600">
            Could not load history: {(messages.error as Error).message}
          </div>
        )}
        {messages.data?.map((m) => (
          <PersistedMessage key={m.id} m={m} />
        ))}
        {inFlight && <InFlightView turn={inFlight} streaming={streaming} />}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
        className="p-3 border-t border-slate-200 bg-white"
      >
        <div className="flex gap-2">
          <input
            type="text"
            value={draft}
            onChange={(e) => setDraft(sessionId, e.target.value)}
            placeholder="Ask the assistant..."
            disabled={streaming}
            className="flex-1 border border-slate-300 rounded-md px-3 py-2 text-sm disabled:bg-slate-100"
          />
          <button
            type="submit"
            disabled={streaming || !draft.trim()}
            className="px-3 py-2 bg-slate-900 text-white text-sm rounded-md disabled:opacity-50"
          >
            {streaming ? "..." : "Send"}
          </button>
        </div>
      </form>
    </div>
  );
}

function PersistedMessage({ m }: { m: ChatMessage }) {
  if (m.role === "user") {
    const text = (m.content.text as string | undefined) ?? "";
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] bg-slate-900 text-white text-sm px-3 py-2 rounded-lg">
          {text}
        </div>
      </div>
    );
  }
  if (m.role === "assistant") {
    const blocks = (m.content.blocks as Array<Record<string, unknown>>) ?? [];
    const text = blocks
      .filter((b) => b.type === "text")
      .map((b) => b.text as string)
      .join("");
    const tool_uses = blocks.filter((b) => b.type === "tool_use");
    return (
      <div className="text-sm">
        {text && <div className="whitespace-pre-wrap">{text}</div>}
        {tool_uses.length > 0 && (
          <div className="mt-1 text-xs text-slate-500">
            {tool_uses.map((b) => (
              <div key={b.id as string}>
                → <code>{b.name as string}</code>(
                {summariseInput(b.input as Record<string, unknown>)})
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }
  // tool
  const isErr = Boolean(m.content.is_error);
  const out = (m.content.output as string) ?? "";
  return (
    <div
      className={
        "text-xs font-mono px-2 py-1 rounded border " +
        (isErr
          ? "bg-red-50 border-red-200 text-red-700"
          : "bg-slate-100 border-slate-200 text-slate-600")
      }
    >
      {truncate(out, 240)}
    </div>
  );
}

function InFlightView({
  turn,
  streaming,
}: {
  turn: InFlightTurn;
  streaming: boolean;
}) {
  return (
    <div className="space-y-1">
      {turn.thinkingText && (
        <div className="text-xs italic text-slate-500 whitespace-pre-wrap">
          {turn.thinkingText}
        </div>
      )}
      {turn.assistantText && (
        <div className="text-sm whitespace-pre-wrap">{turn.assistantText}</div>
      )}
      {turn.toolCalls.map((tc) => (
        <div key={tc.tool_use_id} className="text-xs text-slate-500">
          → <code>{tc.name}</code>({summariseInput(tc.input)})
          {tc.output !== undefined && (
            <span className="ml-1 text-slate-400">
              ✓ {tc.is_error ? "error" : truncate(tc.output, 80)}
            </span>
          )}
        </div>
      ))}
      {streaming && (
        <div className="text-xs text-slate-400">
          <span className="inline-block w-2 h-2 bg-slate-400 rounded-full animate-pulse" />
        </div>
      )}
      {turn.error && (
        <div className="text-xs text-red-600">{turn.error}</div>
      )}
    </div>
  );
}

function summariseInput(input: Record<string, unknown>): string {
  return Object.entries(input)
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join(", ");
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}
