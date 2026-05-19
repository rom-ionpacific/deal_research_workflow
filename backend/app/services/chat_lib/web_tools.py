"""Opt-in web search tool for chat. Backed by Gemini 2.5 Pro with Google
Search grounding -- same pattern slack_helper/run_ai_updates.py uses for
the AI news digest.

The tool is registered into a phase registry only when the user has
explicitly enabled external sources via the per-message toggle on the
frontend. The orchestrator clones the cached phase registry and calls
`register_web_tools` for that turn, so a turn with the toggle off sees
no `web_search` schema in the tools array at all.

Output shape: a brief text answer synthesised by Gemini plus a `sources`
list of `{url, title}` so the Claude orchestrator can cite them as
clickable markdown links. We do NOT scrape page bodies -- the synthesised
answer + snippets that Gemini's grounding already gives are enough to
answer most factual lookups without a separate fetch tool.
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from ...config import settings
from .tool_registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# Gemini's pricing is favourable for grounded calls relative to Claude's
# `web_search` server-tool, and we already depend on it for slack_helper.
# Pro because we want the same quality bar as the consolidation work that
# validated Pro >> Flash Lite > Sonnet for grounded judgement tasks.
GEMINI_MODEL = "gemini-2.5-pro"

# How many grounding chunks (URL + title pairs) to surface to Claude.
# Keep it tight -- Claude only needs enough to cite, not a wall of refs.
MAX_SOURCES = 8


class WebSearchInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Search query in natural language. Gemini with Google Search "
            "grounding will fetch real-time web results and synthesise a "
            "brief answer plus citations. Keep the query focused -- one "
            "factual question per call. Examples: 'What is Soma Capital "
            "AUM in 2025?', 'Who is the CEO of Bitmovin?'."
        ),
        min_length=1,
        max_length=400,
    )


def web_search(inp: WebSearchInput, ctx: dict) -> ToolResult:
    """Synchronous Gemini call -- chat_lib runs this in asyncio.to_thread
    so the SSE stream isn't blocked. Returns a structured object so the
    orchestrator can cite distinctly from internal sources."""
    if not settings.gemini_api_key:
        return ToolResult(output={
            "error": "Web search is not configured on this server "
                     "(GEMINI_API_KEY missing). Tell the user external "
                     "sources are unavailable right now and answer from "
                     "internal data only.",
        })

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return ToolResult(output={
            "error": "google-genai package not installed on the server. "
                     "Web search disabled.",
        })

    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=inp.query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
    except Exception as e:
        logger.exception("web_search Gemini call failed")
        return ToolResult(output={
            "error": f"Web search failed: {type(e).__name__}: {e}",
        })

    answer = (response.text or "").strip()
    sources = _extract_sources(response)

    return ToolResult(output={
        "answer": answer,
        "sources": sources,
        "note": (
            "These are EXTERNAL web sources. Cite them in your reply as "
            "markdown links: `[title](url)`. Distinguish them from "
            "internal data-room sources by mentioning they came from a "
            "web search."
        ),
    })


def _extract_sources(response: Any) -> list[dict]:
    """Pull (title, url) pairs out of Gemini's grounding metadata. The
    SDK shape is a little nested -- candidate -> grounding_metadata ->
    grounding_chunks[] -> web.{uri, title}. Be defensive: any field can
    be None on edge-case responses."""
    out: list[dict] = []
    try:
        candidates = response.candidates or []
        if not candidates:
            return out
        gm = getattr(candidates[0], "grounding_metadata", None)
        if gm is None:
            return out
        chunks = getattr(gm, "grounding_chunks", None) or []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if web is None:
                continue
            url = getattr(web, "uri", None) or ""
            title = getattr(web, "title", None) or url
            if not url:
                continue
            out.append({"url": url, "title": title})
            if len(out) >= MAX_SOURCES:
                break
    except Exception:
        logger.exception("failed to extract grounding sources")
    return out


def register_web_tools(registry: ToolRegistry) -> None:
    """Add the web_search tool to an existing registry. Caller should
    clone the cached phase registry first so we don't mutate the
    module-level base across turns."""
    registry.tool(
        "web_search",
        (
            "OPTIONAL external source. Search the public web via Gemini "
            "with Google Search grounding. Returns a synthesised answer "
            "plus a list of `{url, title}` sources. Use this only when "
            "the internal data room / dossier / database genuinely "
            "doesn't have the answer (e.g. recent news, public market "
            "data, regulatory filings, third-party coverage). Always "
            "cite the returned URLs as markdown links `[title](url)` "
            "and mention that the information came from a web search."
        ),
        WebSearchInput,
    )(web_search)
