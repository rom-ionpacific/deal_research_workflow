"""Document body reader.

Reads `dealcloud.document.body` (lazy-cached full text) for one
document, falling back to the deal_cloud_enhancer
`/internal/document-body/{id}` HTTP endpoint to trigger extraction
when the cache is empty.

Identifier resolution: accepts `document_id` (preferred),
`document_name` (partial case-insensitive match), or `web_url`.
First non-None is used in that order of priority.

Output shape: see `DocumentBodyResult` -- callers (the chat tool)
JSON-encode it for Claude.
"""
from __future__ import annotations

import urllib.parse
import urllib.request
import urllib.error
import json
from dataclasses import dataclass
from typing import Optional

import psycopg2.extras

from ..config import settings
from ..db import get_conn


@dataclass
class DocumentBodyResult:
    document_id: int | None
    ok: bool
    name: str | None
    path: str | None
    web_url: str | None
    modified_at: str | None
    total_chars: int
    returned_chars: int
    truncated: bool
    body: str | None
    cached: bool
    error: str | None


def _resolve_document_id(
    document_id: Optional[int],
    document_name: Optional[str],
    web_url: Optional[str],
) -> Optional[int]:
    """Resolve to a document_id given any of the three identifiers."""
    if document_id is not None:
        return document_id

    if web_url:
        with get_conn() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                "SELECT id FROM dealcloud.document WHERE web_url = %s LIMIT 1",
                (web_url,),
            )
            row = cur.fetchone()
            if row:
                return row["id"]
        return None

    if document_name:
        with get_conn() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            # Prefer the most-recently-modified match so stale duplicates
            # don't win when a user gives a name that hit multiple times.
            cur.execute(
                """
                SELECT id FROM dealcloud.document
                 WHERE name ILIKE %s
                 ORDER BY modified_at DESC NULLS LAST, id DESC
                 LIMIT 1
                """,
                (f"%{document_name}%",),
            )
            row = cur.fetchone()
            if row:
                return row["id"]
        return None

    return None


def _read_cached_row(document_id: int) -> Optional[dict]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT id, name, path, web_url, modified_at, mime_type, summary,
                   body, body_extracted_at, body_extraction_error
              FROM dealcloud.document
             WHERE id = %s
            """,
            (document_id,),
        )
        return cur.fetchone()


def _trigger_extraction_via_dce(document_id: int) -> dict:
    """Call the dce internal endpoint to extract on-demand. Returns the
    parsed JSON response (or a dict with `error` if the call fails)."""
    if not settings.dce_internal_url or not settings.dce_internal_secret:
        return {
            "ok": False,
            "error": "dce_internal_not_configured",
        }

    url = f"{settings.dce_internal_url.rstrip('/')}/internal/document-body/{document_id}"
    req = urllib.request.Request(
        url,
        method="POST",
        headers={
            "X-Internal-Secret": settings.dce_internal_secret,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
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


def _filter_body_to_snippets(
    body: str, query: str, max_chars: int, ctx_chars: int = 500,
) -> str:
    """Return the parts of `body` near case-insensitive occurrences of
    `query`, with `ctx_chars` of context on each side, joined by
    `\\n\\n---\\n\\n`, capped at `max_chars`. If the query is not
    found at all, returns an empty string (caller surfaces that)."""
    if not query:
        return body[:max_chars]
    needle = query.lower()
    hay = body.lower()
    if needle not in hay:
        return ""
    # Walk all match positions, build (start, end) spans with context.
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        i = hay.find(needle, pos)
        if i < 0:
            break
        start = max(0, i - ctx_chars)
        end = min(len(body), i + len(needle) + ctx_chars)
        spans.append((start, end))
        pos = i + len(needle)
    # Merge overlapping spans (cheap because they're already sorted).
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    # Concatenate up to max_chars, marking gaps.
    out_parts: list[str] = []
    used = 0
    sep = "\n\n---\n\n"
    for s, e in merged:
        snippet = body[s:e]
        if used and used + len(sep) + len(snippet) > max_chars:
            break
        if out_parts:
            out_parts.append(sep)
            used += len(sep)
        out_parts.append(snippet)
        used += len(snippet)
        if used >= max_chars:
            break
    return "".join(out_parts)[:max_chars]


def get_document_body(
    *,
    document_id: Optional[int] = None,
    document_name: Optional[str] = None,
    web_url: Optional[str] = None,
    max_chars: int = 20_000,
    query: Optional[str] = None,
) -> DocumentBodyResult:
    """Return the document body, extracting on-demand if not cached.

    When `query` is given, the returned body is filtered to ~500 char
    windows around each case-insensitive match. Use this for long docs
    (PPM, LPA) where the user is asking about a specific topic."""

    doc_id = _resolve_document_id(document_id, document_name, web_url)
    if doc_id is None:
        return DocumentBodyResult(
            document_id=None, ok=False, name=None, path=None, web_url=None,
            modified_at=None, total_chars=0, returned_chars=0,
            truncated=False, body=None, cached=False,
            error="document_not_found",
        )

    row = _read_cached_row(doc_id)
    if row is None:
        return DocumentBodyResult(
            document_id=doc_id, ok=False, name=None, path=None, web_url=None,
            modified_at=None, total_chars=0, returned_chars=0,
            truncated=False, body=None, cached=False,
            error="document_not_found",
        )

    body = row["body"]
    cached = body is not None
    error = row["body_extraction_error"]

    if body is None and error is None:
        # Lazy: ask dce to extract + cache.
        dce_resp = _trigger_extraction_via_dce(doc_id)
        if dce_resp.get("ok"):
            body = dce_resp.get("body")
            cached = False
        else:
            error = dce_resp.get("error") or "dce_extraction_failed"

    if body is None:
        return DocumentBodyResult(
            document_id=doc_id, ok=False, name=row["name"], path=row["path"],
            web_url=row["web_url"],
            modified_at=row["modified_at"].isoformat() if row["modified_at"] else None,
            total_chars=0, returned_chars=0, truncated=False, body=None,
            cached=cached, error=error or "no_body",
        )

    total = len(body)

    if query:
        snippets = _filter_body_to_snippets(body, query, max_chars=max_chars)
        if snippets:
            return DocumentBodyResult(
                document_id=doc_id, ok=True, name=row["name"], path=row["path"],
                web_url=row["web_url"],
                modified_at=row["modified_at"].isoformat() if row["modified_at"] else None,
                total_chars=total, returned_chars=len(snippets),
                truncated=False, body=snippets, cached=cached, error=None,
            )
        # Query not found: fall through to the unfiltered head with an
        # `error` hint so the model knows the query missed and can
        # decide whether to read more or try a different query.
        head = body[:max_chars]
        return DocumentBodyResult(
            document_id=doc_id, ok=True, name=row["name"], path=row["path"],
            web_url=row["web_url"],
            modified_at=row["modified_at"].isoformat() if row["modified_at"] else None,
            total_chars=total, returned_chars=len(head),
            truncated=total > max_chars, body=head, cached=cached,
            error=f"query_not_found:{query!r}",
        )

    truncated = total > max_chars
    returned = body[:max_chars] if truncated else body

    return DocumentBodyResult(
        document_id=doc_id,
        ok=True,
        name=row["name"],
        path=row["path"],
        web_url=row["web_url"],
        modified_at=row["modified_at"].isoformat() if row["modified_at"] else None,
        total_chars=total,
        returned_chars=len(returned),
        truncated=truncated,
        body=returned,
        cached=cached,
        error=None,
    )


def to_tool_output(result: DocumentBodyResult) -> dict:
    """Render a DocumentBodyResult as the dict Claude will see."""
    return {
        "document_id":  result.document_id,
        "ok":           result.ok,
        "name":         result.name,
        "path":         result.path,
        "web_url":      result.web_url,
        "modified_at":  result.modified_at,
        "total_chars":  result.total_chars,
        "returned_chars": result.returned_chars,
        "truncated":    result.truncated,
        "body":         result.body,
        "cached":       result.cached,
        "error":        result.error,
    }
