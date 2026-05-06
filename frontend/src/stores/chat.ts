import { create } from "zustand";

/**
 * In-flight chat state for the streaming turn.
 *
 * Persisted history (user messages, finalised assistant turns, tool
 * results) lives in the server and is fetched via TanStack Query. This
 * store only holds the bits that don't have a server identity yet: the
 * input draft, the partially-streamed assistant text, and the
 * tool-call sequence as it arrives.
 *
 * Keyed by sessionId because the user can have multiple research
 * sessions open in different tabs. We don't preserve drafts across
 * reloads -- if you wanted that, lift the input draft into ui.ts /
 * localStorage instead.
 */

export interface InFlightToolCall {
  tool_use_id: string;
  name: string;
  input: Record<string, unknown>;
  output?: string;
  is_error?: boolean;
  mutates_state?: boolean;
  version_id?: string; // populated when the tool emits version_created
}

export interface InFlightTurn {
  // Anchor identifiers from turn_start, useful for debugging and for
  // wiring the typing indicator to a specific assistant message_id once
  // it's assigned.
  user_message_id?: string;
  ai_message_id?: string;

  // Streamed assistant text accumulated from text_delta events.
  assistantText: string;
  // Optional thinking text -- shown only if the orchestrator forwards
  // thinking_delta events (off by default).
  thinkingText: string;

  // Tool calls in arrival order. Each call's output gets filled in by
  // the matching tool_result event.
  toolCalls: InFlightToolCall[];

  // Set on `error` events (transport, Anthropic 4xx, max_iters).
  // Surfaced inline in the panel; doesn't clear until the next turn.
  error?: string;
}

interface ChatState {
  drafts: Record<string, string>; // sessionId -> input draft
  inFlight: Record<string, InFlightTurn | null>;
  streaming: Record<string, boolean>;

  setDraft: (sessionId: string, value: string) => void;
  startTurn: (sessionId: string) => void;
  endTurn: (sessionId: string) => void;
  resetTurn: (sessionId: string) => void;
  patchInFlight: (
    sessionId: string,
    fn: (prev: InFlightTurn) => InFlightTurn
  ) => void;
}

const blankTurn = (): InFlightTurn => ({
  assistantText: "",
  thinkingText: "",
  toolCalls: [],
});

export const useChat = create<ChatState>((set, get) => ({
  drafts: {},
  inFlight: {},
  streaming: {},

  setDraft: (sessionId, value) =>
    set((s) => ({ drafts: { ...s.drafts, [sessionId]: value } })),

  startTurn: (sessionId) =>
    set((s) => ({
      inFlight: { ...s.inFlight, [sessionId]: blankTurn() },
      streaming: { ...s.streaming, [sessionId]: true },
      drafts: { ...s.drafts, [sessionId]: "" },
    })),

  endTurn: (sessionId) =>
    set((s) => ({
      // Keep inFlight visible until the messages query refetches and
      // shows the canonical version -- the ChatPanel clears via
      // resetTurn() once the new server-side messages arrive. This
      // avoids a flash of empty state between turn_done and the
      // refetch resolving.
      streaming: { ...s.streaming, [sessionId]: false },
    })),

  resetTurn: (sessionId) =>
    set((s) => ({
      inFlight: { ...s.inFlight, [sessionId]: null },
      streaming: { ...s.streaming, [sessionId]: false },
    })),

  patchInFlight: (sessionId, fn) => {
    const prev = get().inFlight[sessionId];
    if (!prev) return; // safety: should only be called between startTurn and resetTurn
    set((s) => ({
      inFlight: { ...s.inFlight, [sessionId]: fn(prev) },
    }));
  },
}));
