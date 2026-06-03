import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";

import Markdown from "../components/Markdown";
import {
  api,
  type DealOnePagerResp,
  type PortfolioDirectPosition,
  type PortfolioFund,
  type PortfolioRelationshipContent,
} from "../lib/api";
import { useUI } from "../stores/ui";

const PORTFOLIO_SECTION_KEY = "portfolio_relationship";

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
            <PortfolioBanner
              section={onePager?.sections.find(
                (s) => s.section_key === PORTFOLIO_SECTION_KEY
              )}
            />
            {onePager?.sections
              .filter((s) => s.section_key !== PORTFOLIO_SECTION_KEY)
              .map((s) => (
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

/** Banner-style render of the portfolio_relationship section. We
 * special-case this above the regular sections loop so the user sees
 * the company's standing immediately under the page header rather than
 * buried in section 6 of 7. Typed content from the section is the
 * source of truth; the section's stored content_markdown is the Slack
 * / Todd / fallback render. */
function PortfolioBanner({
  section,
}: {
  section: { content: unknown } | undefined;
}) {
  if (!section) return null;
  const content = section.content as PortfolioRelationshipContent | null;
  if (!content) return null;

  if (!content.in_portfolio) {
    return (
      <div className="mb-5">
        <span className="inline-flex items-center px-3 py-1.5 rounded-full bg-slate-100 border border-slate-200 text-sm text-slate-600">
          Not in portfolio
        </span>
      </div>
    );
  }

  return (
    <div className="mb-5 rounded-md border border-emerald-200 bg-emerald-50 px-4 py-3">
      <div className="flex items-center justify-between gap-3 mb-2">
        <span className="text-sm font-semibold text-emerald-900">
          In Portfolio
        </span>
        {content.summary && (
          <span className="text-xs text-emerald-800/80 text-right">
            {content.summary}
          </span>
        )}
      </div>
      <ul className="space-y-2 text-sm text-slate-800">
        {content.direct_positions.map((p, i) => (
          <li key={`d-${i}`}>
            <div>
              <span className="font-semibold">{p.deal_name}</span>
              {p.is_co_invest && (
                <span className="ml-2 text-[10px] uppercase tracking-wide rounded px-1.5 py-0.5 bg-amber-100 text-amber-800 border border-amber-200">
                  Co-invest
                </span>
              )}
              <PositionMoneyTail p={p} />
            </div>
            <FundList funds={p.funds ?? []} />
          </li>
        ))}
        {content.indirect_positions.map((p, i) => (
          <li key={`i-${i}`}>
            <span className="font-medium">
              Indirect via {p.via_org_name ?? "(unknown GP)"}
            </span>
            <span className="text-slate-600">
              {" "}— through Ion fund{" "}
              <em>{p.ion_fund_name ?? "(unallocated)"}</em>{" "}
              <span className="text-slate-500">(deal: {p.deal_name})</span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function FundList({ funds }: { funds: PortfolioFund[] }) {
  if (!funds || funds.length === 0) {
    return (
      <div className="ml-3 mt-0.5 text-xs text-slate-500 italic">
        No fund vehicle on record.
      </div>
    );
  }
  return (
    <div className="ml-3 mt-0.5 flex flex-wrap gap-1.5 items-center text-xs">
      <span className="text-slate-500">Funds:</span>
      {funds.map((f, i) => (
        <FundChip key={`${f.fund_id}-${i}`} fund={f} />
      ))}
    </div>
  );
}

function FundChip({ fund }: { fund: PortfolioFund }) {
  const label = fund.fund_type_label || "Other";
  // SPV (deal-specific vehicle) vs Blind Pool (commingled) vs SMA vs
  // Other. Distinct colors so a glance separates the "is this our
  // single-deal SPV?" vs "is this the parent commingled fund?" cases.
  const styles =
    label === "SPV"
      ? "bg-purple-50 border-purple-200 text-purple-800"
      : label === "Blind Pool"
        ? "bg-blue-50 border-blue-200 text-blue-800"
        : label === "SMA"
          ? "bg-teal-50 border-teal-200 text-teal-800"
          : "bg-slate-100 border-slate-200 text-slate-700";
  return (
    <span className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 bg-white">
      <span className="font-medium text-slate-800">
        {fund.fund_name ?? `#${fund.fund_id}`}
      </span>
      <span
        className={`text-[10px] uppercase tracking-wide rounded px-1.5 py-0.5 border ${styles}`}
      >
        {label}
      </span>
    </span>
  );
}

function PositionMoneyTail({ p }: { p: PortfolioDirectPosition }) {
  const bits: string[] = [];
  if (p.invested_capital != null && p.invested_capital !== 0)
    bits.push(`invested ${fmtUSD(p.invested_capital)}`);
  if (p.fair_value != null && p.fair_value !== 0)
    bits.push(`fair value ${fmtUSD(p.fair_value)}`);
  if (p.total_value_to_invested != null)
    bits.push(`${p.total_value_to_invested.toFixed(2)}x TVI`);
  if (bits.length === 0) return null;
  return <span className="text-slate-600"> — {bits.join(" · ")}</span>;
}

function fmtUSD(n: number): string {
  const a = Math.abs(n);
  if (a >= 1e9) return `$${(n / 1e9).toFixed(1)}B`;
  if (a >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `$${(n / 1e3).toFixed(0)}K`;
  return `$${n.toFixed(0)}`;
}
