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
  // Latest UI-context snapshot per session. Phase components publish to
  // this when their view state changes; ChatPanel reads it on submit
  // and forwards to the orchestrator. Keyed by sessionId so multiple
  // tabs with different sessions don't trample each other.
  uiContexts: Record<string, Record<string, unknown> | null>;
  // Sticky per-session toggle for external web sources. Off by default
  // (privacy-safe). ChatPanel renders a segmented button; flipping it
  // sets this value, which then rides on the next turn's request body.
  // Not persisted across reloads in v1 -- session-bound is the right
  // granularity for now since each research session has its own
  // intent.
  webSearchEnabled: Record<string, boolean>;
  // Phase 4 data-room A/B knob. Default 'both' -- the model sees
  // both ask_toltiq + ask_claude_room. 'claude' / 'toltiq' strip
  // the other tool so the model has only one path. UI surfaces the
  // toggle only when room.provider === 'both'.
  chatProviderMode: Record<string, "both" | "claude" | "toltiq">;

  setDraft: (sessionId: string, value: string) => void;
  startTurn: (sessionId: string) => void;
  endTurn: (sessionId: string) => void;
  resetTurn: (sessionId: string) => void;
  patchInFlight: (
    sessionId: string,
    fn: (prev: InFlightTurn) => InFlightTurn
  ) => void;
  setUIContext: (
    sessionId: string,
    ctx: Record<string, unknown> | null,
  ) => void;
  setWebSearchEnabled: (sessionId: string, enabled: boolean) => void;
  setChatProviderMode: (
    sessionId: string,
    mode: "both" | "claude" | "toltiq",
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
  uiContexts: {},
  webSearchEnabled: {},
  chatProviderMode: {},

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

  setUIContext: (sessionId, ctx) =>
    set((s) => ({
      uiContexts: { ...s.uiContexts, [sessionId]: ctx },
    })),

  setWebSearchEnabled: (sessionId, enabled) =>
    set((s) => ({
      webSearchEnabled: { ...s.webSearchEnabled, [sessionId]: enabled },
    })),

  setChatProviderMode: (sessionId, mode) =>
    set((s) => ({
      chatProviderMode: { ...s.chatProviderMode, [sessionId]: mode },
    })),
}));
