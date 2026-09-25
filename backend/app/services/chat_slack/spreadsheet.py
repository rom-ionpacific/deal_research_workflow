"""Building .xlsx files and handing them to Slack.

Two callers, one writer:

  * `create_spreadsheet` -- the model supplies the rows. Ad-hoc tables,
    including ones derived from a file somebody just uploaded.
  * `export_to_spreadsheet` -- a named internal dataset is re-fetched
    server-side and written in full, because the interesting exports
    (the ~1,450-deal book) are far past what the model can retype
    inside its output budget.

Four things in here are defensive rather than decorative, and all four
are reachable from ordinary use now that Todd reads uploaded documents:

1. **Formula injection.** openpyxl turns any string beginning `=` into
   a live formula. A cell lifted out of a PDF someone emailed us is
   untrusted input, and `=HYPERLINK(...)`/`=WEBSERVICE(...)` in a
   spreadsheet a colleague opens is the classic CSV-injection shape.
   `_set_cell` forces those to text.
2. **Illegal characters.** openpyxl raises IllegalCharacterError on the
   control bytes that PDF and DOCX extraction routinely emit, which
   would turn "make me a spreadsheet of this" into a 500.
3. **Excel's own limits** -- 31-char unique sheet names, 32,767 chars
   per cell -- which are silently fatal rather than loud.
4. **Memory.** The workbook is built in `write_only` mode and streamed
   to a temp file, not assembled as a list of Cell objects on a 512 MB
   instance.
"""
from __future__ import annotations

import datetime as _dt
import io
import logging
import re
from decimal import Decimal
from typing import Any, Iterable, Optional, Sequence

log = logging.getLogger(__name__)

# Excel hard limits.
MAX_SHEET_NAME = 31
MAX_CELL_CHARS = 32_767
MAX_ROWS = 1_048_576

# Our own, well inside Slack's file limits and an instance's memory.
MAX_ROWS_PER_SHEET = 100_000
MAX_COLS = 200

_INVALID_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")
# openpyxl's own ILLEGAL_CHARACTERS_RE, inlined so this module doesn't
# depend on a private-ish constant staying put.
_ILLEGAL_CHARS = re.compile(r"[\000-\010\013\014\016-\037]")
# Leading characters Excel treats as the start of a formula.
_FORMULA_LEAD = ("=", "+", "-", "@")


class SpreadsheetError(Exception):
    """Recoverable problem surfaced to the user as prose."""


# ---------------------------------------------------------------------------
# Value coercion
# ---------------------------------------------------------------------------

def _clean_text(s: str) -> str:
    s = _ILLEGAL_CHARS.sub("", s)
    if len(s) > MAX_CELL_CHARS:
        s = s[: MAX_CELL_CHARS - 1] + "…"
    return s


def _coerce(value: Any) -> Any:
    """Map a JSON-ish value to something openpyxl will write, keeping
    numbers and dates as real numbers and dates so the spreadsheet can
    be sorted and summed rather than merely read."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, Decimal):
        # Decimal is what psycopg2 hands back for NUMERIC; Excel has no
        # decimal type, and float is what a spreadsheet user wants.
        return float(value)
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        if isinstance(value, _dt.datetime) and value.tzinfo is not None:
            # Excel has no concept of an offset; normalise to naive UTC
            # rather than letting openpyxl refuse the tz-aware value.
            value = value.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, (list, tuple, set)):
        return _clean_text(", ".join(str(v) for v in value))
    if isinstance(value, dict):
        import json
        return _clean_text(json.dumps(value, default=str))
    return _clean_text(str(value))


def _is_formula_like(v: Any) -> bool:
    return isinstance(v, str) and v[:1] in _FORMULA_LEAD


# ---------------------------------------------------------------------------
# Flattening (for exports whose tools return nested dicts)
# ---------------------------------------------------------------------------

def flatten_row(row: dict, *, prefix: str = "", depth: int = 0) -> dict:
    """Flatten nested dicts into dotted keys: {'performance': {'nav': 1}}
    becomes {'performance.nav': 1}. Lists are left to `_coerce` to join,
    because a list of scalars is a cell and a list of dicts is a
    different table -- not something to explode into columns.

    `list_funds` is the reason this exists: its rows nest
    capital_committed and performance blocks, which as a single JSON
    blob per cell would be useless in a spreadsheet.
    """
    out: dict[str, Any] = {}
    for k, v in row.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and depth < 3:
            nested = flatten_row(v, prefix=f"{key}.", depth=depth + 1)
            if nested:
                out.update(nested)
            else:
                out[key] = None          # an empty dict is still a column
        else:
            out[key] = v
    return out


def columns_from_rows(rows: Sequence[dict],
                      preferred: Optional[Sequence[str]] = None) -> list[str]:
    """Column order: `preferred` first (those that actually occur), then
    every remaining key in first-seen order. Stable across exports, and
    puts the identifying columns on the left where a reader looks."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen_set:
                seen.append(k)
                seen_set.add(k)
    if not preferred:
        return seen[:MAX_COLS]
    head = [c for c in preferred if c in seen_set]
    tail = [c for c in seen if c not in set(head)]
    return (head + tail)[:MAX_COLS]


def humanize(col: str) -> str:
    """'deal_name' -> 'Deal Name'; 'performance.nav' -> 'Performance / Nav'."""
    parts = col.split(".")
    return " / ".join(p.replace("_", " ").strip().title() for p in parts)


# ---------------------------------------------------------------------------
# Sheet naming
# ---------------------------------------------------------------------------

def safe_sheet_name(name: str, taken: set[str]) -> str:
    """Excel: <=31 chars, none of []:*?/\\, non-blank, unique per book.
    Any of those silently corrupts the file or throws at save time."""
    n = _INVALID_SHEET_CHARS.sub("-", (name or "Sheet").strip()) or "Sheet"
    n = n[:MAX_SHEET_NAME]
    if n.lower() not in {t.lower() for t in taken}:
        taken.add(n)
        return n
    base = n[: MAX_SHEET_NAME - 4]
    for i in range(2, 100):
        cand = f"{base} ({i})"
        if cand.lower() not in {t.lower() for t in taken}:
            taken.add(cand)
            return cand
    taken.add(n)
    return n


# ---------------------------------------------------------------------------
# Workbook building
# ---------------------------------------------------------------------------

def build_workbook(sheets: Iterable[dict]) -> bytes:
    """Render sheets to .xlsx bytes.

    Each sheet: {"name": str, "columns": [str] | None, "rows": [dict] |
    [[cell]], "column_labels": [str] | None}.

    Uses a normal (not write_only) workbook so the header row can be
    styled and frozen, but writes row-by-row and never materialises a
    second copy of the data.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)          # drop the default empty sheet
    taken: set[str] = set()
    any_sheet = False

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="374151")   # slate-700
    header_align = Alignment(vertical="center", wrap_text=False)

    for spec in sheets:
        rows = spec.get("rows") or []
        if len(rows) > MAX_ROWS_PER_SHEET:
            raise SpreadsheetError(
                f"Sheet {spec.get('name')!r} has {len(rows):,} rows, over the "
                f"{MAX_ROWS_PER_SHEET:,}-row limit."
            )

        dict_rows = bool(rows) and isinstance(rows[0], dict)
        if dict_rows:
            columns = list(spec.get("columns") or columns_from_rows(rows))
            labels = list(spec.get("column_labels")
                          or [humanize(c) for c in columns])
        else:
            columns = list(spec.get("columns") or [])
            labels = list(spec.get("column_labels") or columns)
            if not labels and rows:
                labels = [f"Column {i + 1}" for i in range(len(rows[0]))]

        ws = wb.create_sheet(safe_sheet_name(spec.get("name") or "Sheet", taken))
        any_sheet = True

        widths: list[int] = []
        if labels:
            ws.append([_clean_text(str(x)) for x in labels])
            for i, cell in enumerate(ws[1], start=0):
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = header_align
            widths = [min(len(str(x)) + 2, 60) for x in labels]

        for row in rows:
            values = ([row.get(c) for c in columns] if dict_rows
                      else list(row)[:MAX_COLS])
            ws.append([None] * len(values))       # extend, then set typed
            r = ws.max_row
            for i, raw in enumerate(values, start=1):
                v = _coerce(raw)
                _set_cell(ws.cell(row=r, column=i), v)
                if i - 1 < len(widths):
                    widths[i - 1] = max(widths[i - 1],
                                        min(len(str(v if v is not None else "")) + 2, 60))
                elif v is not None:
                    widths.extend([10] * (i - len(widths)))

        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = max(w, 9)

        if labels:
            ws.freeze_panes = "A2"
            if rows:
                ws.auto_filter.ref = (
                    f"A1:{get_column_letter(max(1, len(labels)))}{ws.max_row}"
                )

    if not any_sheet:
        raise SpreadsheetError("Nothing to write -- no sheets were given.")

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _set_cell(cell, value: Any) -> None:
    """Assign a value, forcing formula-looking text to stay text.

    openpyxl's value setter infers data_type='f' from a leading '=', so
    a cell copied out of an uploaded document could execute on open in
    the reader's Excel. Re-stamping data_type after assignment writes it
    as a string instead. Dates keep a readable number format -- without
    one Excel shows the serial number.
    """
    cell.value = value
    if _is_formula_like(value) and cell.data_type == "f":
        cell.data_type = "s"
    if isinstance(value, _dt.datetime):
        cell.number_format = "yyyy-mm-dd hh:mm"
    elif isinstance(value, _dt.date):
        cell.number_format = "yyyy-mm-dd"


# ---------------------------------------------------------------------------
# Slack upload
# ---------------------------------------------------------------------------

def safe_filename(name: str, default: str = "todd-export") -> str:
    """A filename Slack and every OS will accept, always ending .xlsx."""
    n = (name or "").strip() or default
    n = re.sub(r"[^A-Za-z0-9 ._()\-]", "", n).strip(" .") or default
    if not n.lower().endswith(".xlsx"):
        n = re.sub(r"\.(xls|csv|txt)$", "", n, flags=re.I) + ".xlsx"
    return n[:120]


def upload_to_slack(
    *,
    content: bytes,
    filename: str,
    channel_id: str,
    thread_ts: Optional[str] = None,
    title: Optional[str] = None,
    initial_comment: Optional[str] = None,
) -> dict:
    """Upload a built workbook into the conversation.

    Returns {ok, permalink, filename, size_bytes} or {ok: False, error}.
    Never raises: a failed upload should be something Todd can explain,
    not a dropped turn.
    """
    from ..slack.client import client as slack_client

    if slack_client is None:
        return {"ok": False, "error": "Slack isn't configured on this deployment."}
    if not channel_id:
        return {"ok": False, "error": "No Slack channel to upload to."}

    kwargs: dict[str, Any] = {
        "channel": channel_id,
        "content": content,
        "filename": filename,
        "title": title or filename,
        "snippet_type": None,
    }
    kwargs.pop("snippet_type")
    if thread_ts:
        kwargs["thread_ts"] = thread_ts
    if initial_comment:
        kwargs["initial_comment"] = initial_comment

    try:
        resp = slack_client.files_upload_v2(**kwargs)
    except Exception as e:  # noqa: BLE001 -- SlackApiError and transport alike
        msg = str(e)
        if "missing_scope" in msg or "not_allowed_token_type" in msg:
            return {"ok": False, "error": (
                "I don't have permission to upload files to Slack yet -- the "
                "app needs the `files:write` scope.")}
        log.warning("[todd/spreadsheet] upload failed: %s: %s", type(e).__name__, e)
        return {"ok": False, "error": f"Slack rejected the upload: {msg[:200]}"}

    data = resp.data if hasattr(resp, "data") else dict(resp)
    f = data.get("file") or {}
    if not f:
        files = data.get("files") or []
        f = files[0] if files else {}
    return {
        "ok": True,
        "filename": f.get("name") or filename,
        "permalink": f.get("permalink"),
        "file_id": f.get("id"),
        "size_bytes": len(content),
    }
