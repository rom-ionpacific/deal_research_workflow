import { useUI } from "../stores/ui";

const BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "";

/** Thin fetch wrapper that injects the V0 stub auth header. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const email = useUI.getState().userEmail;
  const headers = new Headers(init?.headers);
  headers.set("Content-Type", "application/json");
  if (email) headers.set("X-User-Email", email);
  const res = await fetch(`${BASE}${path}`, { ...init, headers });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${body || path}`);
  }
  return res.json() as Promise<T>;
}

// ----- types (mirror the FastAPI Pydantic shapes) -----
export type Phase =
  | "org_select"
  | "entity_select"
  | "data_room_setup"
  | "data_room_view";

export interface Session {
  id: string;
  originator_email: string;
  title: string | null;
  current_version_id: string;
  redo_version_id: string | null;
  forked_from_version_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface Version {
  id: string;
  session_id: string;
  parent_id: string | null;
  undo_unit_id: string;
  phase: Phase;
  state: Record<string, unknown>;
  source: string;
  ai_message_id: string | null;
  summary: string | null;
  created_at: string;
}

export interface SessionWithCurrent {
  session: Session;
  current_version: Version;
}

export interface OrgContact {
  email: string;
  name: string | null;
}

export interface OrgSearchResult {
  org_id: number;
  name: string;
  // score / why_match are present on /orgs/search hits but not on
  // /orgs/by-ids batch fetches.
  score: number | null;
  why_match: string | null;
  sample_evidence: string[];
  // enriched fields from dealcloud.organization_summary; refreshed
  // nightly. Counts default to 0 when the org isn't in the summary
  // table; the contact fields are nullable.
  document_count: number;
  communication_count: number;
  latest_update_at: string | null;
  main_contact: OrgContact | null;
  main_ion_contact: OrgContact | null;
}

// ----- entities (phase 2) -----

export type EntityType =
  | "document"
  | "email_thread"
  | "calendar_event"
  | "slack_message_group";

export const ENTITY_TYPES: EntityType[] = [
  "document",
  "email_thread",
  "calendar_event",
  "slack_message_group",
];

export interface EntityFilter {
  // ISO timestamps. Empty string / null = no constraint.
  date_from?: string | null;
  date_to?: string | null;
  contains?: string | null;
}

export interface EntityCountResp {
  entity_type: EntityType;
  count: number;
}

export interface EntityListResp {
  entity_type: EntityType;
  count: number;
  rows: Array<Record<string, unknown>>;
  limit: number;
  offset: number;
}

function entityFilterToQuery(
  filter: EntityFilter,
  extra: Record<string, string | number> = {},
): string {
  const params = new URLSearchParams();
  if (filter.date_from) params.set("date_from", filter.date_from);
  if (filter.date_to) params.set("date_to", filter.date_to);
  if (filter.contains && filter.contains.trim()) {
    params.set("contains", filter.contains.trim());
  }
  for (const [k, v] of Object.entries(extra)) {
    params.set(k, String(v));
  }
  const s = params.toString();
  return s ? `?${s}` : "";
}

// ----- data rooms (phase 3) -----

export interface PresetQuestion {
  id: number;
  label: string;
  question_text: string;
  sort_order: number | null;
  grouping: string | null;
}

export interface BuildDataRoomResp {
  data_room_id: number;
  name: string;
  entity_count: number;
  question_count: number;
  new_version_id: string;
  created_at: string;
}

// ----- chat -----

export interface ChatMessage {
  id: string;
  session_id: string;
  phase: string;
  role: "user" | "assistant" | "tool";
  content: Record<string, unknown>;
  pre_version_id: string | null;
  post_version_id: string | null;
  parent_message_id: string | null;
  model_id: string | null;
  tokens_in: number | null;
  tokens_out: number | null;
  latency_ms: number | null;
  error: string | null;
  created_at: string;
}

// SSE events the backend emits. See backend/app/services/chat_research/
// orchestrator.py for the producer side.
export type ChatEvent =
  | { type: "turn_start"; session_id: string; phase: string;
      user_message_id: string; current_version_id: string; undo_unit_id: string }
  | { type: "assistant_start" }
  | { type: "text_delta"; text: string }
  | { type: "thinking_delta"; text: string }
  | { type: "assistant_message"; message_id: string; stop_reason: string;
      ai_message_id?: string;
      content: Array<Record<string, unknown>>;
      usage: { input_tokens: number; output_tokens: number;
        cache_creation_input_tokens: number; cache_read_input_tokens: number } }
  | { type: "tool_call"; tool_use_id: string; name: string;
      input: Record<string, unknown>; assistant_message_id: string }
  | { type: "tool_result"; tool_use_id: string; name: string;
      output: string; is_error: boolean; tool_message_id?: string;
      mutates_state?: boolean }
  | { type: "version_created"; version_id: string; phase: string;
      summary: string }
  | { type: "turn_complete"; stop_reason: string }
  | { type: "turn_done"; session_id: string; current_version_id: string }
  | { type: "turn_failed"; reason: string; last_stop_reason?: string }
  | { type: "error"; message: string };

// ----- endpoints -----
export const api = {
  whoami: () => request<{ email: string }>("/api/v1/me"),

  listSessions: () => request<Session[]>("/api/v1/sessions"),
  createSession: () =>
    request<SessionWithCurrent>("/api/v1/sessions", {
      method: "POST",
      body: JSON.stringify({}),
    }),
  getSession: (id: string) =>
    request<SessionWithCurrent>(`/api/v1/sessions/${id}`),

  appendVersion: (
    sessionId: string,
    body: {
      parent_id: string;
      phase: Phase;
      state: Record<string, unknown>;
      summary?: string;
    }
  ) =>
    request<{ version: Version; session: Session }>(
      `/api/v1/sessions/${sessionId}/versions`,
      { method: "POST", body: JSON.stringify(body) }
    ),

  searchOrgs: (q: string, limit = 10) =>
    request<OrgSearchResult[]>(
      `/api/v1/orgs/search?q=${encodeURIComponent(q)}&limit=${limit}`
    ),

  getOrgsByIds: (ids: number[]) =>
    request<OrgSearchResult[]>(
      `/api/v1/orgs/by-ids?ids=${ids.join(",")}`
    ),

  countEntities: (
    sessionId: string,
    entityType: EntityType,
    filter: EntityFilter,
  ) => {
    const qs = entityFilterToQuery(filter);
    return request<EntityCountResp>(
      `/api/v1/sessions/${sessionId}/entities/${entityType}/count${qs}`,
    );
  },

  listEntities: (
    sessionId: string,
    entityType: EntityType,
    filter: EntityFilter,
    limit: number,
    offset: number,
  ) => {
    const qs = entityFilterToQuery(filter, { limit, offset });
    return request<EntityListResp>(
      `/api/v1/sessions/${sessionId}/entities/${entityType}/list${qs}`,
    );
  },

  // ---- phase 3: data-rooms ----
  getPresetQuestions: () =>
    request<PresetQuestion[]>("/api/v1/data-rooms/preset-questions"),

  buildDataRoom: (sessionId: string) =>
    request<BuildDataRoomResp>(
      `/api/v1/sessions/${sessionId}/data-rooms`,
      { method: "POST" },
    ),

  listMessages: (sessionId: string, limit = 200) =>
    request<ChatMessage[]>(
      `/api/v1/sessions/${sessionId}/messages?limit=${limit}`
    ),
};

// -- SSE chat stream --------------------------------------------------------

// Browser EventSource can only do GET and can't set request headers, so
// it's a non-starter for our X-User-Email-authenticated POST. Roll a
// thin SSE parser over fetch + ReadableStream instead.
//
// Frame format (matches the backend's _sse_format):
//   event: <type>\ndata: <json>\n\n
//
// Caller passes onEvent for typed dispatch. The returned promise resolves
// when the stream closes cleanly; reject on network/parse errors. Pass an
// AbortSignal to cancel mid-stream (the backend's StreamingResponse
// observes the disconnect and tears the loop down via the orchestrator's
// finally block).

export interface ChatStreamRequest {
  sessionId: string;
  phase: Phase;
  message: string;
  parentId?: string;
  signal?: AbortSignal;
  onEvent: (ev: ChatEvent) => void;
}

export async function streamChat(req: ChatStreamRequest): Promise<void> {
  const email = useUI.getState().userEmail;
  const headers = new Headers();
  headers.set("Content-Type", "application/json");
  headers.set("Accept", "text/event-stream");
  if (email) headers.set("X-User-Email", email);

  const res = await fetch(`${BASE}/api/v1/sessions/${req.sessionId}/chat`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      phase: req.phase,
      message: req.message,
      parent_id: req.parentId,
    }),
    signal: req.signal,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${body}`);
  }
  if (!res.body) {
    throw new Error("Response has no body — SSE not supported by this fetch?");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      // Each SSE frame is delimited by a blank line. Walk forward,
      // dispatching every complete frame and keeping the trailing
      // partial in the buffer for the next iteration.
      let sep: number;
      while ((sep = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        dispatchFrame(frame, req.onEvent);
      }
    }
    // Flush any trailing frame (shouldn't normally happen since the
    // backend always ends frames with \n\n, but be defensive).
    if (buf.trim()) dispatchFrame(buf, req.onEvent);
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // Best-effort; aborted reads can throw on releaseLock.
    }
  }
}

function dispatchFrame(frame: string, onEvent: (ev: ChatEvent) => void): void {
  // Read both the `event:` line and the `data:` line. The SSE spec
  // makes `event:` the authoritative type; we use it as a fallback
  // when the JSON body lacks one. Earlier orchestrator code emitted
  // turn_start / turn_done with payloads that didn't include `type`,
  // and the frontend's switch silently no-op'd, leaving the input
  // disabled forever. This makes either source of truth work.
  let dataStr = "";
  let eventType = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("data:")) dataStr += line.slice(5).trim();
    else if (line.startsWith("event:")) eventType = line.slice(6).trim();
  }
  if (!dataStr) return;
  try {
    const parsed = JSON.parse(dataStr) as Partial<ChatEvent> & {
      type?: string;
    };
    if (!parsed.type && eventType) {
      (parsed as { type: string }).type = eventType;
    }
    onEvent(parsed as ChatEvent);
  } catch (err) {
    console.warn("SSE: malformed JSON in data:", dataStr, err);
  }
}
