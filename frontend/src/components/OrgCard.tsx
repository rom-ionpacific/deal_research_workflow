import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { api, type OrgContactRecent, type OrgSearchResult } from "../lib/api";

/**
 * Org card. Compact header is two lines + an optional why_match line;
 * click anywhere except the checkbox toggles the expand panel, which
 * lazy-fetches the dossier (counts per entity, top-5 Ion / counterpart
 * contacts, org description). Clicking the checkbox area toggles
 * selection without expanding.
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
  const [expanded, setExpanded] = useState(false);

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

  // Lazy: only fetch the dossier once expanded, then keep it cached.
  // Stale-time is generous since dossier data changes slowly.
  const dossier = useQuery({
    queryKey: ["org-dossier", org.org_id],
    queryFn: () => api.getOrgDossier(org.org_id),
    enabled: expanded,
    staleTime: 5 * 60_000,
  });

  return (
    <div
      className={
        "border rounded-md transition-colors " +
        (selected
          ? "border-slate-900 bg-slate-50"
          : "border-slate-200 bg-white hover:bg-slate-50")
      }
    >
      {/* Header row -- the whole header (excluding the checkbox area
          itself) is the expand toggle. The checkbox area handles the
          select/deselect via stopPropagation so the card doesn't
          expand at the same time. */}
      <div
        className={
          "px-3 py-2 cursor-pointer " +
          (disabled ? "opacity-50 pointer-events-none" : "")
        }
        onClick={() => setExpanded((v) => !v)}
        role="button"
        aria-expanded={expanded}
      >
        <div className="flex items-center gap-2">
          <span
            // Stop both click + mousedown from bubbling so the parent
            // expand handler doesn't fire when the user is toggling
            // selection. mousedown is needed because some browsers
            // fire it before the synthetic click bubbling.
            onClick={(e) => {
              e.stopPropagation();
              if (!disabled) onToggle();
            }}
            onMouseDown={(e) => e.stopPropagation()}
            className="p-1 -m-1"
          >
            <input
              type="checkbox"
              readOnly
              checked={selected}
              disabled={disabled}
              className="cursor-pointer"
            />
          </span>
          <div className="font-medium truncate">{org.name}</div>
          <code className="text-xs text-slate-400 shrink-0">#{org.org_id}</code>
          <span className="ml-auto text-xs text-slate-500 shrink-0 tabular-nums">
            {docs} docs · {comms} comms
          </span>
          <Chevron open={expanded} />
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
      </div>

      {expanded && (
        <div className="border-t border-slate-200 px-3 py-3 text-xs text-slate-700 space-y-3">
          {dossier.isLoading && (
            <div className="text-slate-500">Loading dossier…</div>
          )}
          {dossier.error && (
            <div className="text-red-600">
              {(dossier.error as Error).message}
            </div>
          )}
          {dossier.data && <DossierBody dossier={dossier.data} />}
        </div>
      )}
    </div>
  );
}

function DossierBody({
  dossier,
}: {
  dossier: NonNullable<ReturnType<typeof api.getOrgDossier> extends Promise<infer T> ? T : never>;
}) {
  return (
    <>
      {dossier.description && (
        <div>
          <SectionLabel>Description</SectionLabel>
          <div className="mt-1 leading-relaxed text-slate-800">
            {dossier.description}
          </div>
        </div>
      )}

      <div>
        <SectionLabel>Counts</SectionLabel>
        <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-0.5 sm:grid-cols-5 tabular-nums">
          <CountCell label="Documents" value={dossier.counts.documents} />
          <CountCell label="Emails" value={dossier.counts.email_threads} />
          <CountCell label="Calendar" value={dossier.counts.calendar_events} />
          <CountCell label="Slack" value={dossier.counts.slack_groups} />
          <CountCell
            label="Communications"
            value={dossier.counts.communications}
          />
        </div>
      </div>

      <ContactBlock
        label="Top Ion Pacific contacts"
        contacts={dossier.top_ion_contacts}
        emailKey="ion_email"
        nameKey="ion_name"
      />
      <ContactBlock
        label="Top company contacts"
        contacts={dossier.top_their_contacts}
        emailKey="email"
        nameKey="name"
      />
    </>
  );
}

function ContactBlock({
  label,
  contacts,
  emailKey,
  nameKey,
}: {
  label: string;
  contacts: OrgContactRecent[];
  emailKey: "email" | "ion_email";
  nameKey: "name" | "ion_name";
}) {
  if (!contacts || contacts.length === 0) {
    return (
      <div>
        <SectionLabel>{label}</SectionLabel>
        <div className="mt-1 text-slate-400">No comms data yet.</div>
      </div>
    );
  }
  return (
    <div>
      <SectionLabel>{label}</SectionLabel>
      <ul className="mt-1 space-y-0.5">
        {contacts.slice(0, 5).map((c, i) => {
          const email = (c as Record<string, unknown>)[emailKey] as
            | string
            | null
            | undefined;
          const name = (c as Record<string, unknown>)[nameKey] as
            | string
            | null
            | undefined;
          const display = name || email || "(unknown)";
          const active = c.active_touches ?? 0;
          const passive = c.passive_touches ?? 0;
          const total = active + passive;
          return (
            <li key={i} className="flex items-center gap-2">
              <span className="truncate text-slate-800">{display}</span>
              {email && name && (
                <span className="text-slate-400 truncate">{email}</span>
              )}
              <span className="ml-auto shrink-0 tabular-nums text-slate-500">
                {total} comm{total === 1 ? "" : "s"}
                {active > 0 && passive > 0 && (
                  <span className="text-slate-400">
                    {" "}
                    ({active}↗ / {passive}↙)
                  </span>
                )}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function CountCell({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-baseline justify-between">
      <span className="text-slate-500">{label}</span>
      <span className="font-medium text-slate-800">
        {numberFmt.format(value)}
      </span>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[10px] uppercase tracking-wide text-slate-500 font-semibold">
      {children}
    </div>
  );
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={
        "shrink-0 text-slate-400 transition-transform " +
        (open ? "rotate-180" : "")
      }
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

const numberFmt = new Intl.NumberFormat("en-US");

function formatLatest(ts: string | null | undefined): string {
  if (!ts) return "";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return "";
  return d.toISOString().slice(0, 10);
}

function pickName(
  c: { email: string; name: string | null } | null | undefined,
): string {
  if (!c) return "";
  if (c.name && c.name.trim()) return c.name;
  const at = c.email.indexOf("@");
  return at > 0 ? c.email.slice(0, at) : c.email;
}
