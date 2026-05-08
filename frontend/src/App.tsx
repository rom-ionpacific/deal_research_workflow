import { useQueryClient } from "@tanstack/react-query";
import { Navigate, Route, Routes, Link, useNavigate } from "react-router-dom";

import ResearchPage from "./routes/ResearchPage";
import SessionsListPage from "./routes/SessionsListPage";
import { useUI } from "./stores/ui";

export default function App() {
  return (
    <div className="h-full flex flex-col">
      <header className="border-b border-slate-200 px-6 py-3 flex items-center justify-between gap-4">
        <Link to="/" className="font-semibold text-slate-800 shrink-0">
          deal_research_workflow
        </Link>
        <UserStrip />
      </header>
      {/* min-h-0 + overflow-hidden lets nested scrollers (the chat
          panel's message list, the org search list) own scrolling
          independently. Without this, main grows to fit content and
          the whole page scrolls as one. */}
      <main className="flex-1 min-h-0 overflow-hidden">
        <Routes>
          <Route path="/" element={<SessionsListPage />} />
          <Route path="/research/:sessionId" element={<ResearchPage />} />
          <Route
            path="/research/:sessionId/v/:versionId"
            element={<ResearchPage />}
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}

/** Top-right strip: "Logged in as <email>" + Log out, or just the v0
 * tag when no user is signed in (the SessionsListPage shows the V0
 * sign-in form in that case). Logout clears the persisted email and
 * wipes the TanStack Query cache so the next user doesn't see prior
 * sessions / messages. */
function UserStrip() {
  const userEmail = useUI((s) => s.userEmail);
  const setUserEmail = useUI((s) => s.setUserEmail);
  const qc = useQueryClient();
  const navigate = useNavigate();

  if (!userEmail) {
    return <span className="text-xs text-slate-500">v0</span>;
  }

  const onLogout = () => {
    setUserEmail("");
    qc.clear();
    navigate("/");
  };

  return (
    <div className="flex items-center gap-3 text-sm min-w-0">
      <span className="text-slate-600 truncate">
        Logged in as <span className="font-medium text-slate-800">{userEmail}</span>
      </span>
      <button
        type="button"
        onClick={onLogout}
        className="shrink-0 text-xs px-2 py-1 border border-slate-300 rounded-md text-slate-600 hover:bg-slate-50"
      >
        Log out
      </button>
    </div>
  );
}
