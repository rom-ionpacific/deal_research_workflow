"""Files people drop into Slack, turned into text Todd can read.

Flow for one uploaded file:

    Slack event `files[]` entry
      -> download from url_private_download with the TODD BOT token
      -> POST the bytes to dce /internal/extract-text
      -> persist the text in research.slack_uploaded_file
      -> hand the turn a short excerpt; the rest comes back through the
         `read_uploaded_file` tool

Three things about that chain are deliberate.

**The bot token, not SLACK_USER_TOKEN.** `deals_tracker` reads the
weekly tracker with the user token because the bot is not in
#existing_pipeline and lacked `files:read`. That does not generalise: a
file shared in someone's DM with Todd is visible to Todd, and to the
user token only if that same person happens to be the token's owner.
Per-user uploads have to be read as the bot. (This does mean the app
now needs the `files:read` scope, which it did not before.)

**Extraction in dce, not here.** drw carries openpyxl and nothing else;
dce already has PyMuPDF + Gemini-vision OCR, python-docx, python-pptx,
the xlsx `keep_links=False` memory fix and a recursive zip reader, all
behind one `extract_text`. Porting them would mean a second copy to
keep tuned. See `../document_body.py`, which makes the same call for a
different reason.

**The body goes in a table, not in the conversation.** A DM is one
eternal `slack_conversation` row whose `message_history` is trimmed to
`HISTORY_CAP`; anything inlined there is also re-sent to Anthropic on
every later turn. So the turn carries `EXCERPT_CHARS` of text and the
model pulls the rest deliberately -- the same split as
`read_document_summary` vs `read_document`.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

import psycopg2.extras

from ...config import settings
from ...db import get_conn

log = logging.getLogger(__name__)


# Mirrors dce document_scanner.SUPPORTED_MIME. Kept as an explicit local
# set rather than fetched, so an unreadable file is refused before we
# spend a download on it -- the user gets "I can't read .mov files"
# immediately instead of after 40 MB of transfer.
PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
XLS_MIME = "application/vnd.ms-excel"
ZIP_MIME = "application/zip"

SUPPORTED_MIME = {
    PDF_MIME, DOCX_MIME, PPTX_MIME, XLSX_MIME, XLS_MIME, ZIP_MIME,
    "text/plain", "image/jpeg", "image/png",
}

# Slack's `filetype` label -> mime. Needed because Slack hands back
# `application/octet-stream` for a fair number of Office uploads, and
# `text/plain` covers a long tail of code-snippet filetypes.
FILETYPE_TO_MIME = {
    "pdf": PDF_MIME,
    "docx": DOCX_MIME, "doc": DOCX_MIME,
    "pptx": PPTX_MIME, "ppt": PPTX_MIME,
    "xlsx": XLSX_MIME, "xlsm": XLSX_MIME,
    "xls": XLS_MIME, "csv": "text/plain",
    "zip": ZIP_MIME,
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
}
_TEXTY_FILETYPES = {
    "text", "txt", "markdown", "md", "json", "yaml", "javascript", "python",
    "sql", "html", "css", "xml", "shell", "java", "go", "ruby", "php", "c",
    "cpp", "csharp", "rust", "swift", "kotlin", "scala", "r", "matlab",
}

# Size ceilings, mirroring dce's size_limit_for_mime. Checked against the
# event's declared `size` so an oversized file is never downloaded.
MAX_BYTES_DEFAULT = 30_000_000
MAX_BYTES_BY_MIME = {
    PDF_MIME: 60_000_000,     # deliberately BELOW dce's 250 MB PDF ceiling:
                              # that number is sized for a batch cron with
                              # minutes to spend, this path has a human
                              # waiting and a 180s gunicorn timeout.
    XLSX_MIME: 50_000_000,
    ZIP_MIME: 50_000_000,
}
MIN_BYTES = 100

MAX_FILES_PER_MESSAGE = 5     # beyond this we report rather than grind
EXCERPT_CHARS = 2_000         # inlined into the turn -- see module docstring
DCE_TIMEOUT_SECONDS = 170     # dce's gunicorn --timeout is 180


class UploadError(Exception):
    """Recoverable problem surfaced to the user as prose, not a 500."""


# ---------------------------------------------------------------------------
# Mime resolution
# ---------------------------------------------------------------------------

def resolve_mime(f: dict) -> Optional[str]:
    """Best-effort mime for a Slack file dict, or None if we can't read
    this type. Slack's own `mimetype` wins when it's something we
    support; otherwise fall back to the `filetype` label, which is the
    more reliable field for Office documents."""
    declared = (f.get("mimetype") or "").strip().lower()
    if declared in SUPPORTED_MIME:
        return declared
    filetype = (f.get("filetype") or "").strip().lower()
    if filetype in FILETYPE_TO_MIME:
        return FILETYPE_TO_MIME[filetype]
    if filetype in _TEXTY_FILETYPES:
        return "text/plain"
    if declared.startswith("text/"):
        return "text/plain"
    return None


def size_limit_for_mime(mime: str) -> int:
    return MAX_BYTES_BY_MIME.get(mime, MAX_BYTES_DEFAULT)


# ---------------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------------

def _download_to_temp(url: str, expected_max: int) -> tuple[str, int]:
    """Stream a Slack file to a temp path. Returns (path, bytes_written).

    Streams rather than reading into memory: this runs on a 512 MB
    instance and the bytes are about to be POSTed on to dce, so holding
    the whole file here would mean two copies resident at once.
    """
    token = settings.slack_bot_token
    if not token:
        raise UploadError(
            "SLACK_BOT_TOKEN isn't configured, so I can't download files "
            "from Slack."
        )
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}"}
    )
    fd, path = tempfile.mkstemp(prefix="slack_upload_")
    written = 0
    try:
        with os.fdopen(fd, "wb") as out:
            with urllib.request.urlopen(req, timeout=120) as resp:
                first = resp.read(64 * 1024)
                # Slack answers an unauthorised file fetch with 200 + an
                # HTML login page rather than a 401. deals_tracker learned
                # this the same way; checking the magic bytes is the only
                # reliable signal.
                if first[:15].lower().startswith(b"<!doctype html") or \
                        first[:6].lower() == b"<html>":
                    raise UploadError(
                        "Slack returned a login page instead of the file. The "
                        "bot token is probably missing the `files:read` scope."
                    )
                while first:
                    written += len(first)
                    if written > expected_max:
                        raise UploadError(
                            f"File is larger than the {expected_max:,}-byte "
                            "limit for its type."
                        )
                    out.write(first)
                    first = resp.read(1024 * 1024)
    except UploadError:
        _unlink(path)
        raise
    except urllib.error.HTTPError as e:
        _unlink(path)
        raise UploadError(
            f"Slack refused the download (HTTP {e.code}). The bot token may "
            "lack the `files:read` scope."
        ) from e
    except Exception as e:  # noqa: BLE001 -- any transport failure
        _unlink(path)
        raise UploadError(f"Couldn't download the file from Slack: {e}") from e
    return path, written


def _extract_via_dce(path: str, mime: str, filename: str) -> dict:
    """POST the file to dce /internal/extract-text and return its JSON.

    Streams the temp file as the request body with an explicit
    Content-Length -- without it urllib falls back to chunked transfer
    encoding, which the WSGI server on the other end reads as an empty
    body.
    """
    if not settings.dce_internal_url or not settings.dce_internal_secret:
        return {"ok": False, "error": "dce_internal_not_configured"}

    qs = urllib.parse.urlencode({"mime": mime, "filename": filename})
    url = f"{settings.dce_internal_url.rstrip('/')}/internal/extract-text?{qs}"
    size = os.path.getsize(path)
    with open(path, "rb") as body:
        req = urllib.request.Request(
            url,
            method="POST",
            data=body,
            headers={
                "X-Internal-Secret": settings.dce_internal_secret,
                "Content-Type": "application/octet-stream",
                "Content-Length": str(size),
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=DCE_TIMEOUT_SECONDS) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # 413/415/422 carry a JSON body explaining why; keep it.
            try:
                payload = json.loads(e.read().decode("utf-8"))
            except Exception:
                payload = {"error": f"http_{e.code}"}
            payload.setdefault("ok", False)
            return payload
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return {
                "ok": False,
                "error": f"dce_unreachable: {type(e).__name__}: {e}",
            }


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _store(
    *,
    slack_file_id: str,
    team_id: str,
    channel_id: str,
    thread_ts: Optional[str],
    slack_user_id: str,
    user_email: str,
    name: str,
    mimetype: Optional[str],
    filetype: Optional[str],
    size_bytes: Optional[int],
    permalink: Optional[str],
    ok: bool,
    body: Optional[str],
    error: Optional[str],
) -> None:
    """Upsert on Slack's file id. A re-upload of the same file id (or a
    retry after a transient dce failure) overwrites the previous
    outcome rather than accumulating rows."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO research.slack_uploaded_file
                (slack_file_id, team_id, channel_id, thread_ts, slack_user_id,
                 user_email, name, mimetype, filetype, size_bytes, permalink,
                 ok, body, total_chars, error)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (slack_file_id) DO UPDATE SET
                ok           = EXCLUDED.ok,
                body         = EXCLUDED.body,
                total_chars  = EXCLUDED.total_chars,
                error        = EXCLUDED.error,
                extracted_at = NOW()
            """,
            (slack_file_id, team_id, channel_id, thread_ts, slack_user_id,
             user_email, name, mimetype, filetype, size_bytes, permalink,
             ok, body, len(body or ""), error),
        )


def lookup_uploaded_file(
    *,
    slack_file_id: Optional[str] = None,
    name: Optional[str] = None,
    team_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    thread_ts: Optional[str] = None,
) -> Optional[dict]:
    """Resolve one stored upload by file id, or by (partial, case-
    insensitive) name within a conversation. Newest wins on a name tie --
    if someone uploads 'model.xlsx' twice, they mean the new one."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if slack_file_id:
            cur.execute(
                "SELECT * FROM research.slack_uploaded_file "
                " WHERE slack_file_id = %s",
                (slack_file_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
        if not name:
            return None
        cur.execute(
            """
            SELECT * FROM research.slack_uploaded_file
             WHERE name ILIKE %s
               AND (%s::text IS NULL OR team_id = %s)
               AND (%s::text IS NULL OR channel_id = %s)
             ORDER BY created_at DESC
             LIMIT 1
            """,
            (f"%{name}%", team_id, team_id, channel_id, channel_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def list_conversation_uploads(
    *, team_id: str, channel_id: str, limit: int = 20
) -> list[dict]:
    """Recent uploads in this channel/DM, newest first. Scoped to the
    channel rather than the thread: someone who dropped a file in the DM
    last week and asks about it today is in the same channel but a
    different turn, and that is the common case."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT slack_file_id, name, filetype, size_bytes, ok,
                   total_chars, error, permalink, created_at
              FROM research.slack_uploaded_file
             WHERE team_id = %s AND channel_id = %s
             ORDER BY created_at DESC
             LIMIT %s
            """,
            (team_id, channel_id, limit),
        )
        return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Ingestion (called once per inbound message that carries files)
# ---------------------------------------------------------------------------

def ingest_slack_files(
    files: list[dict],
    *,
    team_id: str,
    channel_id: str,
    thread_ts: Optional[str],
    slack_user_id: str,
    user_email: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Download + extract every attachment on one message.

    Returns one result dict per file:
        {slack_file_id, name, filetype, size_bytes, ok, total_chars,
         excerpt, truncated, error, permalink}

    Never raises: a file that can't be read comes back with ok=False and
    a human-readable `error`, because "I couldn't read that, here's why"
    is a better turn than a stack trace in Render.
    """
    results: list[dict] = []
    for f in files[:MAX_FILES_PER_MESSAGE]:
        try:
            results.append(_ingest_one(
                f,
                team_id=team_id, channel_id=channel_id, thread_ts=thread_ts,
                slack_user_id=slack_user_id, user_email=user_email,
                on_progress=on_progress,
            ))
        except Exception as e:  # noqa: BLE001 -- one bad file must not sink the turn
            log.exception("[todd/uploads] ingest failed for %s", f.get("name"))
            results.append({
                "slack_file_id": f.get("id") or "",
                "name": f.get("name") or "(unnamed)",
                "filetype": f.get("filetype"),
                "size_bytes": f.get("size"),
                "ok": False,
                "total_chars": 0,
                "excerpt": None,
                "truncated": False,
                "error": f"{type(e).__name__}: {e}",
                "permalink": f.get("permalink"),
            })

    for f in files[MAX_FILES_PER_MESSAGE:]:
        results.append({
            "slack_file_id": f.get("id") or "",
            "name": f.get("name") or "(unnamed)",
            "filetype": f.get("filetype"),
            "size_bytes": f.get("size"),
            "ok": False,
            "total_chars": 0,
            "excerpt": None,
            "truncated": False,
            "error": (f"skipped: only the first {MAX_FILES_PER_MESSAGE} files "
                      "on a message are read -- send the rest separately"),
            "permalink": f.get("permalink"),
        })
    return results


def _ingest_one(
    f: dict,
    *,
    team_id: str,
    channel_id: str,
    thread_ts: Optional[str],
    slack_user_id: str,
    user_email: str,
    on_progress: Optional[Callable[[str], None]],
) -> dict:
    file_id = f.get("id") or ""
    name = f.get("name") or "(unnamed)"
    filetype = (f.get("filetype") or "").lower() or None
    size = int(f.get("size") or 0)
    permalink = f.get("permalink")

    def result(ok: bool, *, body: str | None = None, error: str | None = None) -> dict:
        excerpt = None
        truncated = False
        if body:
            excerpt = body[:EXCERPT_CHARS]
            truncated = len(body) > EXCERPT_CHARS
        return {
            "slack_file_id": file_id,
            "name": name,
            "filetype": filetype,
            "size_bytes": size or None,
            "ok": ok,
            "total_chars": len(body or ""),
            "excerpt": excerpt,
            "truncated": truncated,
            "error": error,
            "permalink": permalink,
        }

    # Already read this exact file? Extraction is the expensive part and
    # Slack file ids are stable, so a re-share or a Slack retry is free.
    if file_id:
        cached = lookup_uploaded_file(slack_file_id=file_id)
        if cached is not None and (cached["ok"] or cached.get("error")):
            log.info("[todd/uploads] cache hit %s (%s)", name, file_id)
            return result(cached["ok"], body=cached.get("body"),
                          error=cached.get("error"))

    mime = resolve_mime(f)
    if mime is None:
        error = (f"I can't read `.{filetype or '?'}` files -- I handle PDF, "
                 "Word, PowerPoint, Excel, plain text, images and zips.")
        _store_safe(file_id=file_id, team_id=team_id, channel_id=channel_id,
                    thread_ts=thread_ts, slack_user_id=slack_user_id,
                    user_email=user_email, name=name, mimetype=f.get("mimetype"),
                    filetype=filetype, size_bytes=size or None,
                    permalink=permalink, ok=False, body=None, error=error)
        return result(False, error=error)

    limit = size_limit_for_mime(mime)
    if size and size > limit:
        error = (f"That file is {size / 1_000_000:.1f} MB, over the "
                 f"{limit / 1_000_000:.0f} MB limit for {filetype or mime}.")
        _store_safe(file_id=file_id, team_id=team_id, channel_id=channel_id,
                    thread_ts=thread_ts, slack_user_id=slack_user_id,
                    user_email=user_email, name=name, mimetype=f.get("mimetype"),
                    filetype=filetype, size_bytes=size or None,
                    permalink=permalink, ok=False, body=None, error=error)
        return result(False, error=error)
    if size and size < MIN_BYTES:
        error = f"That file is only {size} bytes -- there's nothing in it to read."
        _store_safe(file_id=file_id, team_id=team_id, channel_id=channel_id,
                    thread_ts=thread_ts, slack_user_id=slack_user_id,
                    user_email=user_email, name=name, mimetype=f.get("mimetype"),
                    filetype=filetype, size_bytes=size or None,
                    permalink=permalink, ok=False, body=None, error=error)
        return result(False, error=error)

    url = f.get("url_private_download") or f.get("url_private")
    if not url:
        error = "Slack didn't give me a download URL for that file."
        return result(False, error=error)

    if on_progress:
        on_progress(name)

    path = None
    try:
        path, _written = _download_to_temp(url, limit)
        payload = _extract_via_dce(path, mime, name)
    except UploadError as e:
        _store_safe(file_id=file_id, team_id=team_id, channel_id=channel_id,
                    thread_ts=thread_ts, slack_user_id=slack_user_id,
                    user_email=user_email, name=name, mimetype=f.get("mimetype"),
                    filetype=filetype, size_bytes=size or None,
                    permalink=permalink, ok=False, body=None, error=str(e))
        return result(False, error=str(e))
    finally:
        if path:
            _unlink(path)

    if payload.get("ok"):
        body = payload.get("text") or ""
        _store_safe(file_id=file_id, team_id=team_id, channel_id=channel_id,
                    thread_ts=thread_ts, slack_user_id=slack_user_id,
                    user_email=user_email, name=name, mimetype=mime,
                    filetype=filetype, size_bytes=size or None,
                    permalink=permalink, ok=True, body=body, error=None)
        log.info("[todd/uploads] read %s (%s) -> %d chars", name, mime, len(body))
        return result(True, body=body)

    error = _friendly_error(payload.get("error") or "extraction_failed", name)
    _store_safe(file_id=file_id, team_id=team_id, channel_id=channel_id,
                thread_ts=thread_ts, slack_user_id=slack_user_id,
                user_email=user_email, name=name, mimetype=mime,
                filetype=filetype, size_bytes=size or None,
                permalink=permalink, ok=False, body=None, error=error)
    log.warning("[todd/uploads] %s failed: %s", name, payload.get("error"))
    return result(False, error=error)


def _friendly_error(raw: str, name: str) -> str:
    """Turn a dce error code into something worth saying out loud. The
    raw code is kept on the end -- when this shows up in a Slack thread
    it is also the bug report."""
    if raw.startswith("no_text_extracted"):
        return (f"I downloaded {name} but got no text out of it -- it may be "
                "a scan with no readable text layer, or an empty file.")
    if raw.startswith("too_large"):
        return f"{name} is too large for me to read."
    if raw.startswith("unsupported_mime"):
        return f"I don't have a reader for {name}'s file type."
    if raw == "dce_internal_not_configured":
        return ("My document reader isn't configured on this deployment "
                "(`DCE_INTERNAL_URL` / `DCE_INTERNAL_SECRET`).")
    if raw.startswith("dce_unreachable"):
        return ("My document reader didn't answer -- it may be starting up, "
                f"or the file was slow to parse. Try again. ({raw})")
    return f"I couldn't read {name}: {raw}"


def _store_safe(*, file_id: str, **kw: Any) -> None:
    """Persisting is best-effort: a DB hiccup must not turn a readable
    file into a failed turn. The cost of losing the row is a re-download
    next time, which is exactly what happens today anyway."""
    if not file_id:
        return
    try:
        _store(slack_file_id=file_id, **kw)
    except Exception:
        log.exception("[todd/uploads] failed to persist %s", kw.get("name"))


# ---------------------------------------------------------------------------
# Turn assembly
# ---------------------------------------------------------------------------

def describe_uploads_for_model(results: list[dict]) -> str:
    """The block appended to the user's message describing what they
    attached. Short by construction -- this text is persisted in
    message_history and re-sent on every later turn in the
    conversation, so the full body deliberately stays out of it."""
    if not results:
        return ""
    lines = ["", "---", "ATTACHED FILES (uploaded by the user with this message):"]
    for r in results:
        head = f"- `{r['name']}`"
        if r.get("filetype"):
            head += f" ({r['filetype']}"
            if r.get("size_bytes"):
                head += f", {r['size_bytes'] / 1_000_000:.1f} MB"
            head += ")"
        if not r["ok"]:
            lines.append(f"{head} -- COULD NOT READ: {r['error']}")
            continue
        head += (f" -- read OK, {r['total_chars']:,} characters of text, "
                 f"file_id `{r['slack_file_id']}`")
        lines.append(head)
        if r.get("excerpt"):
            marker = " [EXCERPT -- truncated]" if r["truncated"] else " [full text]"
            lines.append(f"  Beginning of file{marker}:")
            lines.append("  ---8<---")
            for line in r["excerpt"].splitlines():
                lines.append(f"  {line}")
            lines.append("  ---8<---")
    readable = [r for r in results if r["ok"] and r["truncated"]]
    if readable:
        lines.append(
            "Only the beginning of the truncated file(s) is shown above. Call "
            "`read_uploaded_file(file_id=..., query=...)` to read the rest -- "
            "pass `query` to pull just the parts about a topic instead of the "
            "whole document."
        )
    return "\n".join(lines)
