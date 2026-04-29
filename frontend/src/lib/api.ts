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

export interface OrgSearchResult {
  org_id: number;
  name: string;
  score: number;
  why_match: string;
  sample_evidence: string[];
}

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
};
