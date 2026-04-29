import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";

import { api, type Session } from "../lib/api";
import { useUI } from "../stores/ui";

export default function SessionsListPage() {
  const userEmail = useUI((s) => s.userEmail);
  const setUserEmail = useUI((s) => s.setUserEmail);
  const navigate = useNavigate();
  const qc = useQueryClient();

  const sessions = useQuery({
    queryKey: ["sessions"],
    queryFn: api.listSessions,
    enabled: Boolean(userEmail),
  });

  const create = useMutation({
    mutationFn: api.createSession,
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["sessions"] });
      navigate(`/research/${data.session.id}`);
    },
  });

  if (!userEmail) {
    return (
      <div className="max-w-md mx-auto mt-24 p-6 border rounded-md bg-white shadow-sm">
        <h1 className="text-lg font-semibold mb-2">V0 dev sign-in</h1>
        <p className="text-sm text-slate-600 mb-4">
          Auth is stubbed until Entra ID is wired. Enter your email to identify
          your sessions.
        </p>
        <input
          type="email"
          autoFocus
          placeholder="you@ionpacific.com"
          className="w-full border border-slate-300 rounded-md px-3 py-2"
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              const v = (e.target as HTMLInputElement).value.trim();
              if (v) setUserEmail(v);
            }
          }}
        />
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold">Your research sessions</h1>
        <button
          className="bg-slate-900 text-white px-3 py-2 rounded-md text-sm disabled:opacity-50"
          disabled={create.isPending}
          onClick={() => create.mutate()}
        >
          {create.isPending ? "Creating..." : "+ New session"}
        </button>
      </div>

      {sessions.isLoading && <div className="text-slate-500">Loading...</div>}
      {sessions.error && (
        <div className="text-red-600 text-sm">
          {(sessions.error as Error).message}
        </div>
      )}
      {sessions.data && sessions.data.length === 0 && (
        <div className="text-slate-500">
          No sessions yet. Create one to get started.
        </div>
      )}

      <ul className="divide-y border rounded-md">
        {sessions.data?.map((s: Session) => (
          <li key={s.id}>
            <Link
              to={`/research/${s.id}`}
              className="block px-4 py-3 hover:bg-slate-50"
            >
              <div className="font-medium">{s.title ?? "Untitled session"}</div>
              <div className="text-xs text-slate-500">
                Updated {new Date(s.updated_at).toLocaleString()}
              </div>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
