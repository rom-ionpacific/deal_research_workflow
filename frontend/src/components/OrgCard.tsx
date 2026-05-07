import type { OrgSearchResult } from "../lib/api";

/**
 * Compact org card. Two metadata lines:
 *
 *   [☐] Name #id · 2,613 docs · 3,146 comms
 *   Updated 2026-05-05 · Ion: Lailo Hung · Cherry Yau
 *   why_match · score (only on search results)
 *
 * Counts always render (zero shown explicitly so it reads visibly
 * different from "still loading"). Contact segments are dropped from
 * line 2 when null so the line collapses cleanly. why_match line is
 * omitted on the sticky-selected panel where score isn't meaningful.
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
  const docs = numberFmt.format(org.document_count);
  const comms = numberFmt.format(org.communication_count);
  const updated = formatLatest(org.latest_update_at);

  const ionName = pickName(org.main_ion_contact);
  const contactName = pickName(org.main_contact);

  const line2Parts = [
    updated && `Updated ${updated}`,
    ionName && `Ion: ${ionName}`,
    contactName && contactName,
  ].filter(Boolean);

  return (
    <button
      type="button"
      onClick={onToggle}
      disabled={disabled}
      className={
        "w-full text-left border rounded-md px-3 py-2 hover:bg-slate-50 " +
        "disabled:opacity-50 transition-colors " +
        (selected
          ? "border-slate-900 bg-slate-50"
          : "border-slate-200 bg-white")
      }
    >
      <div className="flex items-center gap-2">
        <input type="checkbox" readOnly checked={selected} />
        <div className="font-medium truncate">{org.name}</div>
        <code className="text-xs text-slate-400 shrink-0">#{org.org_id}</code>
        <span className="ml-auto text-xs text-slate-500 shrink-0 tabular-nums">
          {docs} docs · {comms} comms
        </span>
      </div>
      {line2Parts.length > 0 && (
        <div className="mt-0.5 text-xs text-slate-500 truncate pl-6">
          {line2Parts.join(" · ")}
        </div>
      )}
      {org.why_match && (
        <div className="text-[10px] uppercase tracking-wide text-slate-400 pl-6 mt-0.5">
          {org.why_match}
          {org.score != null && ` · ${org.score.toFixed(2)}`}
        </div>
      )}
    </button>
  );
}

const numberFmt = new Intl.NumberFormat("en-US");

function formatLatest(ts: string | null): string {
  if (!ts) return "";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return "";
  return d.toISOString().slice(0, 10);
}

function pickName(c: { email: string; name: string | null } | null): string {
  if (!c) return "";
  if (c.name && c.name.trim()) return c.name;
  // Fall back to the local-part of the email so we never show a bare
  // address with @ symbols cluttering the line.
  const at = c.email.indexOf("@");
  return at > 0 ? c.email.slice(0, at) : c.email;
}
