import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { api, type DealListItem } from "../lib/api";
import { useUI } from "../stores/ui";

/** Landing page for the deal one-pager web view. Default-lists the
 * live-pipeline deals (the weekly-baked set) with a one-pager status
 * badge each; a search box finds any deal by name or company. */
export default function OnePagersListPage() {
  const userEmail = useUI((s) => s.userEmail);

  const [q, setQ] = useState("");
  const [debouncedQ, setDebouncedQ] = useState("");
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQ(q.trim()), 250);
    return () => clearTimeout(t);
  }, [q]);

  const deals = useQuery({
    queryKey: ["deals", debouncedQ],
    queryFn: () => api.listDeals(debouncedQ || undefined),
    enabled: Boolean(userEmail),
  });

  if (!userEmail) {
    return (
      <div className="h-full overflow-y-auto">
        <div className="max-w-md mx-auto mt-24 p-6 border rounded-md bg-white shadow-sm text-sm text-slate-600">
          Sign in on the{" "}
          <Link to="/" className="text-blue-600 hover:underline">
            home page
          </Link>{" "}
          first to view deal one-pagers.
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-3xl mx-auto p-6">
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold">Deal one-pagers</h1>
            <p className="text-sm text-slate-500 mt-1">
              {debouncedQ
                ? "Search results across all deals."
                : "Live-pipeline deals. Search to find any deal."}
            </p>
          </div>
          <DealcloudSyncButton />
        </div>

        <input
          type="text"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search by deal or company name..."
          className="w-full border border-slate-300 rounded-md px-3 py-2 mb-4"
        />

        {deals.isLoading && <div className="text-slate-500">Loading...</div>}
        {deals.error && (
          <div className="text-red-600 text-sm">
            {(deals.error as Error).message}
          </div>
        )}
        {deals.data && deals.data.length === 0 && (
          <div className="text-slate-500 text-sm">
            {debouncedQ ? "No deals match that search." : "No pipeline deals found."}
          </div>
        )}

        {deals.data && deals.data.length > 0 && (
          <ul className="divide-y border rounded-md">
            {deals.data.map((d) => (
              <DealRow key={d.deal_id} deal={d} />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function DealRow({ deal }: { deal: DealListItem }) {
  return (
    <li>
      <Link
        to={`/one-pagers/${deal.deal_id}`}
        className="flex items-center gap-3 px-4 py-3 hover:bg-slate-50"
      >
        <div className="flex-1 min-w-0">
          <div className="font-medium truncate">{deal.name}</div>
          <div className="text-xs text-slate-500 truncate">
            {deal.company ? `${deal.company} · ` : ""}
            {deal.status}
          </div>
        </div>
        <OnePagerBadge deal={deal} />
      </Link>
    </li>
  );
}

/** Manual "Sync DealCloud data" trigger. A new deal created in DealCloud
 * doesn't otherwise show up here until the next scheduled sync (daily,
 * full on Sundays) -- this lets an analyst pull it in immediately. Polls
 * dce's sync status while a run is in flight so the button reflects
 * progress even across a page refresh. */
function DealcloudSyncButton() {
  // True from the moment we click sync until the polled status leaves
  // 'running' -- covers the race where dce returns 202 before its
  // in-memory state flips to 'running'.
  const [pending, setPending] = useState(false);

  const statusQuery = useQuery({
    queryKey: ["dealcloud-sync-status"],
    queryFn: () => api.getDealcloudSyncStatus(),
    refetchInterval: (q) => {
      if (pending) return 4000;
      const d = q.state.data;
      return d?.status === "running" ? 4000 : false;
    },
  });

  const sync = useMutation({
    mutationFn: () => api.triggerDealcloudSync(),
    onMutate: () => setPending(true),
    onSuccess: () => statusQuery.refetch(),
    onError: () => setPending(false),
  });

  const remoteStatus = statusQuery.data?.status;
  useEffect(() => {
    if (!pending) return;
    if (remoteStatus === "running" || remoteStatus === undefined) return;
    setPending(false);
  }, [pending, remoteStatus]);

  const isRunning = pending || remoteStatus === "running";

  return (
    <div className="shrink-0 flex flex-col items-end gap-1">
      <button
        type="button"
        disabled={isRunning}
        onClick={() => sync.mutate()}
        className="text-sm px-3 py-2 rounded-md border border-slate-300 text-slate-700 hover:bg-slate-50 disabled:opacity-50"
      >
        {isRunning ? "Syncing..." : "Sync DealCloud data"}
      </button>
      {!isRunning && remoteStatus === "complete" && (
        <span className="text-xs text-emerald-600">Synced</span>
      )}
      {!isRunning && (remoteStatus === "failed" || sync.isError) && (
        <span className="text-xs text-red-600">Sync failed</span>
      )}
    </div>
  );
}

function OnePagerBadge({ deal }: { deal: DealListItem }) {
  if (!deal.has_one_pager) {
    return (
      <span className="shrink-0 text-xs px-2 py-1 rounded-full bg-amber-50 text-amber-700 border border-amber-200">
        No one-pager
      </span>
    );
  }
  const when = deal.generated_at
    ? new Date(deal.generated_at).toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
      })
    : "";
  const partial = deal.one_pager_status === "partial";
  return (
    <span
      className={
        "shrink-0 text-xs px-2 py-1 rounded-full border " +
        (partial
          ? "bg-yellow-50 text-yellow-700 border-yellow-200"
          : "bg-emerald-50 text-emerald-700 border-emerald-200")
      }
    >
      {partial ? "Partial" : "Built"}
      {when ? ` · ${when}` : ""}
    </span>
  );
}
