import type { OrgSearchResult } from "../lib/api";

/**
 * Org card used in both the sticky-selected panel and the search
 * results list. Renders the org's enriched metrics from
 * dealcloud.organization_summary (refreshed nightly):
 *
 *   - document count (SharePoint + Slack files combined)
 *   - communication count (email threads + calendar events combined)
 *   - latest update timestamp across those four entity types
 *   - main contact (top external participant)
 *   - main Ion contact (top Ion-internal participant)
 *
 * Counts always render as numbers (zero shown explicitly so a "0
 * documents" card is visibly distinct from "data not yet loaded").
 * Contacts render as "—" when null.
 */
export default function OrgCard({
  org,
  selected,
  onToggle,
  disabled,
}: {
  org: OrgSearchResult;
  selected: boolean;
  onToggle: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      disabled={disabled}
      className={
        "w-full text-left border rounded-md px-4 py-3 hover:bg-slate-50 " +
        "disabled:opacity-50 transition-colors " +
        (selected
          ? "border-slate-900 bg-slate-50"
          : "border-slate-200 bg-white")
      }
    >
      <div className="flex items-start gap-3">
        <input
          type="checkbox"
          readOnly
          checked={selected}
          className="mt-1"
        />
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline justify-between gap-2">
            <div className="font-medium truncate">{org.name}</div>
            <code className="text-xs text-slate-400 shrink-0">
              #{org.org_id}
            </code>
          </div>
          {org.why_match && (
            <div className="text-xs text-slate-500 mt-0.5">
              {org.why_match}
              {org.score != null && ` · score ${org.score.toFixed(2)}`}
            </div>
          )}
          <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
            <Detail
              label="Documents"
              value={String(org.document_count)}
            />
            <Detail
              label="Communications"
              value={String(org.communication_count)}
            />
            <Detail
              label="Latest update"
              value={formatLatest(org.latest_update_at)}
              className="col-span-2"
            />
            <Detail
              label="Main contact"
              value={formatContact(org.main_contact)}
              className="col-span-2"
            />
            <Detail
              label="Main Ion contact"
              value={formatContact(org.main_ion_contact)}
              className="col-span-2"
            />
          </dl>
        </div>
      </div>
    </button>
  );
}

function Detail({
  label,
  value,
  className,
}: {
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div className={"flex items-baseline gap-2 " + (className ?? "")}>
      <dt className="text-slate-500 shrink-0 w-32">{label}</dt>
      <dd className="text-slate-700 truncate">{value}</dd>
    </div>
  );
}

function formatLatest(ts: string | null): string {
  if (!ts) return "—";
  // Render as YYYY-MM-DD; the timestamp's specific time is rarely useful
  // in this view and the day is what people anchor on for recency.
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toISOString().slice(0, 10);
}

function formatContact(c: { email: string; name: string | null } | null): string {
  if (!c) return "—";
  if (c.name && c.name.trim()) return `${c.name} <${c.email}>`;
  return c.email;
}
