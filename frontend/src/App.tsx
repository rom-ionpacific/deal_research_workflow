import { Navigate, Route, Routes, Link } from "react-router-dom";

import ResearchPage from "./routes/ResearchPage";
import SessionsListPage from "./routes/SessionsListPage";

export default function App() {
  return (
    <div className="h-full flex flex-col">
      <header className="border-b border-slate-200 px-6 py-3 flex items-center justify-between">
        <Link to="/" className="font-semibold text-slate-800">
          deal_research_workflow
        </Link>
        <span className="text-xs text-slate-500">v0</span>
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
