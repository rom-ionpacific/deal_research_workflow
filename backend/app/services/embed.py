"""Query-time embedding helper for the deal_research_workflow API.

Counterpart to deal_cloud_enhancer/embed_lib.py: same model + dimensions
so the query embedding is comparable against the stored
dealcloud.organization_embedding vectors. Single-query API (the route
needs one embedding per search call) so we skip the batch interface.

stdlib urllib to avoid pulling in a new dep -- matches the pattern in
services/toltiq_adhoc.py. LRU-cached by query text so repeated
searches (the FE debounces but still re-fires on focus changes / tab
switches) don't burn redundant OpenAI calls.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from urllib import error as urllib_error
from urllib import request as urllib_request

from ..config import settings

logger = logging.getLogger(__name__)


# Must match deal_cloud_enhancer/embed_lib.py. If you change the model
# there, change it here -- otherwise query embeddings live in a
# different geometry than the indexed ones and recall craters.
MODEL = "text-embedding-3-small"
DIMS = 1536
HTTP_TIMEOUT = 30


class EmbedNotConfigured(Exception):
    """OPENAI_API_KEY isn't set. Caller should fall back to trigram."""


class EmbedError(Exception):
    """Network / API failure. Caller should fall back to trigram."""


@lru_cache(maxsize=256)
def embed_query(text: str) -> tuple[float, ...]:
    """Embed a single query string. Cached so the debounced FE
    refire-on-focus pattern doesn't burn extra OpenAI calls. Returns
    a tuple so it's hashable + lru_cache-compatible; callers can
    pass it directly to pgvector via vector_literal()."""
    if not settings.openai_api_key:
        raise EmbedNotConfigured(
            "OPENAI_API_KEY not set on this API instance"
        )
    if not text or not text.strip():
        raise EmbedError("Empty query")

    body = json.dumps({"model": MODEL, "input": text}).encode("utf-8")
    req = urllib_request.Request(
        "https://api.openai.com/v1/embeddings",
        data=body,
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "deal_research_workflow/1.0",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib_error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        logger.warning("OpenAI %d on /embeddings: %s", e.code, detail)
        raise EmbedError(f"OpenAI {e.code}: {detail}") from e
    except urllib_error.URLError as e:
        raise EmbedError(f"OpenAI network error: {e.reason}") from e

    if not data.get("data"):
        raise EmbedError(f"OpenAI returned no data: {data!r:.200}")
    emb = data["data"][0]["embedding"]
    if len(emb) != DIMS:
        raise EmbedError(f"OpenAI returned {len(emb)} dims, expected {DIMS}")
    return tuple(emb)


def vector_literal(emb: tuple[float, ...]) -> str:
    """Format a tuple of floats as the textual form pgvector accepts
    when cast with ::public.vector. Matches embed_lib's format."""
    return "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
