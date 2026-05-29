import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";

import Markdown from "../components/Markdown";
import { api, type DealOnePagerResp } from "../lib/api";
import { useUI } from "../stores/ui";

/** One deal's one-pager: renders the stored sections (standard markdown)
 * and a Refresh/Create button that triggers a rebuild in dce. While a
 * build is in flight the page polls until the fresh one-pager lands;
 * the previous content (if any) stays visible, dimmed. */
export default function DealOnePagerPage() {
  const params = useParams();
  const dealId = Number(params.dealId);
  const userEmail = useUI((s) => s.userEmail);
  const qc = useQueryClient();

  // True from the moment we click build until a newer one-pager lands
  // (or the build goes stale). Drives polling + button state, and
  // covers the race where dce returns 202 before its 'running' row is
  // inserted (so build.state is briefly still 'idle').
  const [buildPending, setBuildPending] = useState(false);
  const prevGeneratedAt = useRef<string | null>(null);

  const query = useQuery({
    queryKey: ["deal-one-pager", dealId],
    queryFn: () => api.getDealOnePager(dealId),
    enabled: Boolean(userEmail) && Number.isFinite(dealId),
    refetchInterval: (q) => {
      const d = q.state.data as DealOnePagerResp | undefined;
      if (buildPending) return 4000;
      if (d?.build.state === "running") return 4000;
      return false;
    },
  });

  const data = query.data;

  // Clear the pending flag once the rebuild finishes (a one-pager with a
  // newer generated_at appears) or the build is reported stalled.
  useEffect(() => {
    if (!buildPending || !data) return;
    if (data.build.state === "stale") {
      setBuildPending(false);
      return;
    }
    const gen = data.one_pager?.generated_at ?? null;
    if (gen && gen !== prevGeneratedAt.current) {
      setBuildPending(false);
    }
  }, [buildPending, data]);

  const build = useMutation({
    mutationFn: (force: boolean) => api.buildDealOnePager(dealId, force),
    onMutate: () => {
      prevGeneratedAt.current = data?.one_pager?.generated_at ?? null;
      setBuildPending(true);
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["deal-one-pager", dealId] });
    },
    onError: () => {
      setBuildPending(false);
    },
  });

  if (!userEmail) {
    return (
      <CenteredNote>
        Sign in on the{" "}
        <Link to="/" className="text-blue-600 hover:underline">
          home page
        </Link>{" "}
        first.
      </CenteredNote>
    );
  }
  if (query.isLoading) return <CenteredNote>Loading...</CenteredNote>;
  if (query.error) {
    return (
      <CenteredNote>
        <span className="text-red-600">{(query.error as Error).message}</span>
      </CenteredNote>
    );
  }
  if (!data) return null;

  const { deal, one_pager: onePager, build: buildState } = data;
  const isBuilding = buildPending || buildState.state === "running";
  const isStale = !isBuilding && buildState.state === "stale";
  const hasPager = Boolean(onePager);

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-3xl mx-auto p-6">
        <Link
          to="/one-pagers"
          className="text-sm text-slate-500 hover:text-slate-800"
        >
          &larr; All deals
        </Link>

        <div className="flex items-start justify-between gap-4 mt-2 mb-4">
          <div className="min-w-0">
            <h1 className="text-xl font-semibold truncate">{deal.name}</h1>
            <div className="text-sm text-slate-500 truncate">
              {deal.company ? `${deal.company} · ` : ""}
              {deal.status}
            </div>
            {onePager?.generated_at && (
              <div className="text-xs text-slate-400 mt-1">
                One-pager generated{" "}
                {new Date(onePager.generated_at).toLocaleString()}
                {onePager.status === "partial" ? " (partial)" : ""}
              </div>
            )}
          </div>
          <button
            type="button"
            disabled={isBuilding || build.isPending}
            onClick={() => build.mutate(false)}
            className="shrink-0 bg-slate-900 text-white px-3 py-2 rounded-md text-sm disabled:opacity-50"
          >
            {isBuilding
              ? "Building..."
              : hasPager
                ? "Refresh"
                : "Create one-pager"}
          </button>
        </div>

        {isBuilding && (
          <div className="mb-4 text-sm rounded-md border border-blue-200 bg-blue-50 px-3 py-2 text-blue-800">
            Building this one-pager from the latest data — this takes about
            two minutes. The page updates automatically when it's ready.
          </div>
        )}

        {isStale && (
          <div className="mb-4 text-sm rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-amber-800 flex items-center justify-between gap-3">
            <span>
              A build started but seems to have stalled (it may have been
              interrupted).
            </span>
            <button
              type="button"
              onClick={() => build.mutate(true)}
              className="shrink-0 text-xs px-2 py-1 border border-amber-300 rounded-md hover:bg-amber-100"
            >
              Retry
            </button>
          </div>
        )}

        {build.error && (
          <div className="mb-4 text-sm rounded-md border border-red-200 bg-red-50 px-3 py-2 text-red-700">
            {(build.error as Error).message}
          </div>
        )}

        {!hasPager && !isBuilding ? (
          <div className="text-slate-500 text-sm border rounded-md px-4 py-8 text-center">
            No one-pager has been built for this deal yet. Click{" "}
            <span className="font-medium">Create one-pager</span> to generate
            one.
          </div>
        ) : (
          <div>
            {onePager?.sections.map((s) => (
              <section key={s.section_key} className="mb-6">
                <h2 className="text-base font-semibold border-b border-slate-200 pb-1 mb-2">
                  {s.title}
                </h2>
                {s.content_markdown.trim() ? (
                  <div className="text-sm text-slate-700">
                    <Markdown>{s.content_markdown}</Markdown>
                  </div>
                ) : (
                  <div className="text-sm text-slate-400 italic">
                    ({s.status})
                  </div>
                )}
              </section>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function CenteredNote({ children }: { children: React.ReactNode }) {
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-md mx-auto mt-24 p-6 border rounded-md bg-white shadow-sm text-sm text-slate-600">
        {children}
      </div>
    </div>
  );
}
