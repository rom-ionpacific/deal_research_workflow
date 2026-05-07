import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { api, type Session } from "../lib/api";
import { useUI } from "../stores/ui";

const RECENT_DEFAULT = 5;

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

  // Server returns starred-first then by updated_at DESC. Split here
  // for the two visual sections.
  const [starred, recent] = useMemo(() => {
    const data = sessions.data ?? [];
    return [data.filter((s) => s.is_starred), data.filter((s) => !s.is_starred)];
  }, [sessions.data]);

  const [historyExpanded, setHistoryExpanded] = useState(false);
  const visibleRecent = historyExpanded
    ? recent
    : recent.slice(0, RECENT_DEFAULT);
  const hiddenCount = Math.max(0, recent.length - RECENT_DEFAULT);

  if (!userEmail) {
    return (
      <div className="h-full overflow-y-auto">
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
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto">
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

        {starred.length > 0 && (
          <section className="mb-6">
            <h2 className="text-xs uppercase tracking-wide text-slate-500 mb-2">
              Starred
            </h2>
            <SessionList sessions={starred} />
          </section>
        )}

        {recent.length > 0 && (
          <section>
            <h2 className="text-xs uppercase tracking-wide text-slate-500 mb-2">
              {starred.length > 0 ? "Other sessions" : "Recent sessions"}
            </h2>
            <SessionList sessions={visibleRecent} />
            {hiddenCount > 0 && !historyExpanded && (
              <button
                type="button"
                onClick={() => setHistoryExpanded(true)}
                className="mt-3 text-sm text-slate-600 hover:text-slate-900"
              >
                Session history... ({hiddenCount} more)
              </button>
            )}
            {historyExpanded && hiddenCount > 0 && (
              <button
                type="button"
                onClick={() => setHistoryExpanded(false)}
                className="mt-3 text-sm text-slate-600 hover:text-slate-900"
              >
                Show less
              </button>
            )}
          </section>
        )}
      </div>
    </div>
  );
}

function SessionList({ sessions }: { sessions: Session[] }) {
  return (
    <ul className="divide-y border rounded-md">
      {sessions.map((s) => (
        <li key={s.id}>
          <Link
            to={`/research/${s.id}`}
            className="flex items-center gap-2 px-4 py-3 hover:bg-slate-50"
          >
            {s.is_starred && (
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="#facc15"
                stroke="#ca8a04"
                strokeWidth="2"
                strokeLinejoin="round"
                className="shrink-0"
              >
                <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
              </svg>
            )}
            <div className="flex-1 min-w-0">
              <div className="font-medium truncate">
                {s.title ?? "Untitled session"}
              </div>
              <div className="text-xs text-slate-500">
                Updated {new Date(s.updated_at).toLocaleString()}
              </div>
            </div>
          </Link>
        </li>
      ))}
    </ul>
  );
}
