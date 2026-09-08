"""MCP-only tool: rewrite Claude's own output in plain English via OpenAI.

Analysts told us they were copying Claude answers into ChatGPT because the
prose is verbose and acronym-dense, then pasting the result back. That
round trip happens outside every system we control -- so the deal text
leaves Ion with no record that it did. `rewrite_plain_english` pulls that
loop back inside a Claude chat (which the Compliance-API archive already
captures), so the rewrite is a logged tool call instead of an untracked
paste into someone's personal ChatGPT account.

Reuses `settings.openai_api_key`, which this repo already sends Ion deal
text to for pgvector query embeddings (services/embed.py) -- same vendor,
same key, no new secret to provision. Same stdlib-urllib call pattern as
embed.py, so no new dependency either.

Three things this tool does that a bare API call would not:

  1. **Scrubs on the way OUT.** `make_response_filter()` (wired in
     mcp/server.py) scrubs tool *output* -- text coming back to Claude.
     This is the first tool that sends Ion text *to a third party*, so the
     response filter is the wrong direction entirely: it would mask an SSN
     only after OpenAI had already seen it. The inbound `text` is run
     through the scrubber here, before the request is built, and the
     redaction tally is reported so the analyst knows a placeholder is in
     their rewrite.

  2. **Checks numeric fidelity.** A paraphraser that turns "8.4x" into
     "roughly 8x", or quietly drops a basis-point figure, is worse than
     verbose prose in an IC context. Every digit-bearing token in the sent
     text is compared against the returned text and any dropped/added
     figure is reported. Advisory, not fatal -- a legitimate `max_words`
     summary will drop numbers on purpose.

  3. **Reports real cost.** `usage` + `estimated_cost_usd` come back on
     every call, so per-call spend is auditable rather than modelled.
     Note that `completion_tokens` INCLUDES reasoning tokens on the
     gpt-5.6 family (~50-65 even on a one-sentence rewrite), and reasoning
     bills at the output rate -- which is why the cost is measured here
     instead of estimated from word counts.

Registered on `chat_mcp_tools.mcp_registry` only, NOT on
`chat_slack.tools.slack_registry`. Same reasoning as the DealCloud-write
tools: slack_registry is shared verbatim with Todd's Slack bot, so
registering there would hand an outbound-to-OpenAI egress path to anyone
who can DM Todd. Moving it is a one-line change if that's ever wanted.

API facts, established by probing the live endpoint (all three cost a 400
if you assume otherwise):
  * gpt-5.6-* rejects `max_tokens` -- it requires `max_completion_tokens`.
  * gpt-5.6-* rejects any `temperature` but the default 1.
  * `max_completion_tokens` is spent on reasoning FIRST, so too small a
    budget returns finish_reason="length" with an EMPTY message -- a
    silent blank rewrite unless you check for it (we do). It is also
    non-deterministic: the same input can come back empty once and answer
    fine on the next call, hence the automatic single retry on a doubled
    budget. Asking for a SHORT output makes this more likely, not less --
    compression is what the model reasons hardest about.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from claude_enterprise_utils.scrubber import Scrubber

from ..config import settings
from .chat_lib import ToolResult
from .chat_mcp_tools import mcp_registry

logger = logging.getLogger(__name__)

API_URL = "https://api.openai.com/v1/chat/completions"
HTTP_TIMEOUT = 180

# Cap the input so one pasted data-room dump can't turn into a $2 call.
# 60k chars is ~15k tokens -- comfortably a full IC memo.
MAX_INPUT_CHARS = 60_000

# Absolute ceiling on billed output tokens per call, whatever the input.
MAX_OUTPUT_TOKENS = 32_000

# Reasoning comes out of the same budget as the prose, and is spent FIRST.
# Every request therefore gets at least this much headroom on top of the
# prose estimate. It is generous on purpose: max_completion_tokens is a
# ceiling, not a charge -- you are billed for tokens actually generated,
# so an unused allowance costs nothing, while too small an allowance costs
# you the whole call (billed, no output).
#
# 8k is not arbitrary. A 558-token prompt with max_words=50 burned 2,080
# reasoning tokens and returned an EMPTY message: asking for compression
# makes the model reason harder, so scaling the budget down with max_words
# -- which an earlier version of this function did -- is exactly backwards.
MIN_REASONING_BUDGET_TOKENS = 8_000

# effort -> (model, $/1M input, $/1M output). Cached input bills at 10% of
# the input rate across this family. Verified against
# https://developers.openai.com/api/docs/pricing on 2026-09-08.
#
# Deliberately an allowlist keyed by effort rather than a free-text model
# field: Claude picking its own model string is how you end up billing
# plain-English rewrites at gpt-6-astra's $10/$50.
_EFFORT_MODELS: dict[str, tuple[str, float, float]] = {
    "fast": ("gpt-5.6-luna", 0.20, 1.20),
    "standard": ("gpt-5.6-terra", 2.00, 12.00),
    "high": ("gpt-5.6-sol", 4.00, 20.00),
}
_CACHED_INPUT_DISCOUNT = 0.10


# ---------------------------------------------------------------------------
# Style prompt
# ---------------------------------------------------------------------------

# Acronyms common enough that expanding them adds noise rather than
# clarity. Everything else gets spelled out on first use.
_ACRONYM_ALLOWLIST = (
    "USD, CEO, CFO, COO, CTO, IPO, US, UK, EU, UAE, VAT, AI, "
    "Q1, Q2, Q3, Q4, FY"
)

_BASE_RULES = f"""You rewrite text so a smart reader with no exposure to \
private-equity or venture jargon can follow it on one read.

Hard constraints -- violating any of these makes the rewrite useless:
- NEVER change a number. Figures, percentages, multiples, dates, \
currencies, basis points and ranges must appear exactly as written in the \
source. Do not round, re-derive, convert, or reformat them.
- NEVER change a name. Companies, funds, people, products and places keep \
their exact spelling.
- Add NOTHING. No new facts, framing, caveats, conclusions or \
recommendations that are not already in the source.
- Drop NOTHING of substance. This is a rewrite, not a summary.
- If a term's meaning is not clear from the source, keep it verbatim \
rather than guessing at what it stands for.

Style:
- Short sentences. Active voice. Concrete subjects that do the verbs.
- Expand every acronym on first use as "full name (ACRONYM)", then use the \
acronym. Exceptions (leave as-is): {_ACRONYM_ALLOWLIST}.
- Replace jargon with the plain word: "headwinds" -> the actual problem, \
"utilise" -> "use", "at this juncture" -> "now".
- Cut hedging stacks ("it may potentially be possible that") down to the \
single claim being made.
- No throat-clearing. Start with the substance.

Output ONLY the rewritten text. No preamble, no "Here is the rewrite", no \
commentary on what you changed, no closing summary."""

_AUDIENCE_RULES = {
    "plain": "Audience: a general business reader. Neutral register.",
    "analyst_note": (
        "Audience: an internal analyst note. Keep it tight and factual; "
        "bullets are fine where the source has parallel items."
    ),
    "ic_memo": (
        "Audience: an Investment Committee memo. Formal but direct. Keep "
        "every figure and its stated source. Preserve any existing "
        "headings and their order."
    ),
    "lp_email": (
        "Audience: an email to a Limited Partner (LP) investor. Courteous "
        "and precise; assume they know finance but not Ion's internal "
        "shorthand. Expand internal shorthand fully."
    ),
    "exec_summary": (
        "Audience: a busy executive. Lead with the conclusion, then the "
        "supporting facts. Still a rewrite, not a summary -- keep the "
        "substance."
    ),
}


def _build_system_prompt(
    audience: str, preserve_structure: bool, max_words: int | None
) -> str:
    parts = [_BASE_RULES, _AUDIENCE_RULES.get(audience, _AUDIENCE_RULES["plain"])]
    if preserve_structure:
        parts.append(
            "Preserve the source's structure: same headings, same bullet "
            "and paragraph breaks, same order. Rewrite within it."
        )
    else:
        parts.append(
            "You may restructure freely (reorder, merge, or re-bullet) if "
            "it makes the text easier to follow."
        )
    if max_words:
        parts.append(
            f"Target at most {max_words} words. If that forces you to cut, "
            f"cut repetition and hedging first, then the least "
            f"load-bearing detail -- never a figure that supports a claim."
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Numeric fidelity check
# ---------------------------------------------------------------------------

# Any run of digits, with thousands separators and one decimal part. Picks
# up 42, 1,234, 8.4, 118.5 -- the surrounding %/x/$ doesn't matter, we only
# care that the magnitude survived.
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _number_multiset(text: str) -> Counter:
    """Digit-bearing tokens in `text`, normalised so 1,234 == 1234."""
    return Counter(
        m.group(0).replace(",", "").rstrip(".") for m in _NUM_RE.finditer(text)
    )


def _number_diff(sent: str, returned: str) -> dict:
    src, out = _number_multiset(sent), _number_multiset(returned)
    dropped = sorted((src - out).elements())
    added = sorted((out - src).elements())
    return {
        "numbers_in_source": sum(src.values()),
        "numbers_dropped": dropped,
        "numbers_added": added,
        "numbers_intact": not dropped and not added,
    }


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------

class RewriteNotConfigured(Exception):
    """OPENAI_API_KEY isn't set on this instance."""


class RewriteError(Exception):
    """Network / API failure, with a message safe to show the user."""


def _estimate_cost_usd(usage: dict, price_in: float, price_out: float) -> float:
    prompt = usage.get("prompt_tokens", 0) or 0
    completion = usage.get("completion_tokens", 0) or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    fresh = max(prompt - cached, 0)
    cost = (
        fresh * price_in / 1e6
        + cached * price_in * _CACHED_INPUT_DISCOUNT / 1e6
        + completion * price_out / 1e6
    )
    return round(cost, 6)


def _output_budget(text: str, max_words: int | None) -> int:
    """Token budget for the response: enough prose for the rewrite, plus
    a reasoning allowance that scales with the INPUT (not the requested
    output). A short `max_words` must never shrink the total -- see
    MIN_REASONING_BUDGET_TOKENS."""
    input_tokens = int(len(text) / 4)  # ~4 chars/token
    prose = int(max_words * 1.6) if max_words else int(input_tokens * 1.3)
    reasoning = max(MIN_REASONING_BUDGET_TOKENS, input_tokens)
    return min(prose + reasoning, MAX_OUTPUT_TOKENS)


def _call_openai(model: str, system: str, text: str, budget: int) -> dict:
    if not settings.openai_api_key:
        raise RewriteNotConfigured(
            "OPENAI_API_KEY is not set on this service, so the rewrite "
            "tool has nothing to call. (The same key powers semantic "
            "search here -- if it's missing, search is silently on the "
            "trigram fallback too.)"
        )

    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            # gpt-5.6-* rejects max_tokens outright, and rejects any
            # temperature but the default -- see the module docstring.
            "max_completion_tokens": budget,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "deal_research_workflow/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        logger.warning("OpenAI %d on /chat/completions: %s", e.code, detail)
        raise RewriteError(f"OpenAI returned {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RewriteError(f"OpenAI unreachable: {type(e).__name__}: {e}") from e


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class RewriteInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description=(
            "The text to rewrite -- normally an answer you just produced "
            "in this conversation. Paste it in full; do not summarise it "
            "first, that's this tool's job."
        ),
    )
    audience: Literal[
        "plain", "analyst_note", "ic_memo", "lp_email", "exec_summary"
    ] = Field(
        "plain",
        description=(
            "Who will read it. Sets register and how aggressively internal "
            "shorthand is expanded. 'plain' is the safe default."
        ),
    )
    effort: Literal["fast", "standard", "high"] = Field(
        "standard",
        description=(
            "Model tier. 'fast' (~$0.003 for a 1,500-word rewrite) is "
            "usually enough for wording cleanup; 'standard' (~$0.03) for "
            "anything client-facing; 'high' (~$0.05) only for dense "
            "technical prose that needs real restructuring. Cost is "
            "reported back on every call."
        ),
    )
    max_words: int | None = Field(
        None,
        ge=50,
        le=20_000,
        description=(
            "Optional length target. Omit for a straight rewrite -- "
            "with a target this becomes a summary and may drop detail "
            "(dropped figures are reported). Soft, not enforced: the "
            "model often overshoots, and a miss is reported back as a "
            "warning rather than silently honoured."
        ),
    )
    preserve_structure: bool = Field(
        True,
        description=(
            "Keep the source's headings, bullets and ordering. Set false "
            "only if the user explicitly wants it reorganised."
        ),
    )


@mcp_registry.tool(
    "rewrite_plain_english",
    (
        "Rewrite text into plain, readable English using OpenAI's model "
        "instead of your own wording -- for when the user says an answer "
        "is too verbose, too dense, or too full of acronyms to follow, or "
        "asks to 'run this through ChatGPT'. Typically you pass your own "
        "previous answer as `text`. "
        "This SENDS THE TEXT TO OPENAI, a third party: only call it on "
        "text the user has asked to have rewritten, never speculatively, "
        "and never on raw document bodies the user hasn't seen. Structured "
        "identifiers (SSNs, bank/account numbers, tax IDs, cards) are "
        "masked before the text leaves, and the tool reports what it "
        "masked. "
        "The result is a rewrite, NOT a fact-check: it cannot add "
        "information and is instructed never to alter a figure or a name. "
        "Every call returns `numeric_fidelity` -- if `numbers_intact` is "
        "false, show the user the dropped/added figures alongside the "
        "rewrite rather than presenting it as clean. Also returns the "
        "model used and `estimated_cost_usd`. Present the rewritten text "
        "verbatim; do not re-edit it, or you've undone the point of "
        "calling this."
    ),
    RewriteInput,
)
def rewrite_plain_english(inp: RewriteInput, ctx: dict) -> ToolResult:
    if len(inp.text) > MAX_INPUT_CHARS:
        return ToolResult(
            output={
                "ok": False,
                "error": "input_too_long",
                "message": (
                    f"text is {len(inp.text):,} characters; the cap is "
                    f"{MAX_INPUT_CHARS:,} (~15k tokens, about one full IC "
                    f"memo). Rewrite it in sections."
                ),
            }
        )

    model, price_in, price_out = _EFFORT_MODELS[inp.effort]

    # Scrub BEFORE the request is built -- the server-level response
    # filter runs the wrong direction for an outbound call.
    scrubbed = Scrubber().scrub(inp.text)
    system = _build_system_prompt(
        inp.audience, inp.preserve_structure, inp.max_words
    )
    budget = _output_budget(scrubbed.text, inp.max_words)

    # Budget exhaustion is non-deterministic -- the same input can reason
    # itself into an empty message on one call and answer fine on the
    # next. One automatic retry on a doubled budget, rather than handing
    # the analyst a blank result and making them re-ask.
    cost = 0.0
    usage: dict = {}
    finish = None
    rewritten = ""
    attempts = 0

    while attempts < 2:
        attempts += 1
        try:
            raw = _call_openai(model, system, scrubbed.text, budget)
        except (RewriteNotConfigured, RewriteError) as e:
            return ToolResult(
                output={
                    "ok": False,
                    "error": type(e).__name__,
                    "message": str(e),
                    "estimated_cost_usd": round(cost, 6),
                }
            )

        usage = raw.get("usage") or {}
        cost += _estimate_cost_usd(usage, price_in, price_out)
        choices = raw.get("choices") or []
        if not choices:
            return ToolResult(
                output={
                    "ok": False,
                    "error": "empty_response",
                    "message": f"OpenAI returned no choices: {str(raw)[:200]}",
                    "model": model,
                    "estimated_cost_usd": round(cost, 6),
                }
            )

        finish = choices[0].get("finish_reason")
        rewritten = (choices[0].get("message") or {}).get("content") or ""
        if rewritten.strip():
            break

        # Empty message: the whole budget went to reasoning before any
        # prose was emitted. Retry once with more room.
        retry_budget = min(budget * 2, MAX_OUTPUT_TOKENS)
        logger.warning(
            "rewrite_plain_english: empty message (finish=%s, %s reasoning "
            "tokens) on budget %s; %s",
            finish,
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            budget,
            f"retrying at {retry_budget}" if retry_budget > budget else "no room to retry",
        )
        if retry_budget <= budget:
            break
        budget = retry_budget

    if not rewritten.strip():
        return ToolResult(
            output={
                "ok": False,
                "error": "no_text_returned",
                "message": (
                    f"The model returned no text after {attempts} "
                    f"attempt(s) (finish_reason={finish!r}); the entire "
                    f"output budget went to reasoning before any prose was "
                    f"emitted. Retry with effort='fast', drop max_words, or "
                    f"split the text into sections."
                ),
                "model": model,
                "usage": usage,
                "estimated_cost_usd": round(cost, 6),
            }
        )

    fidelity = _number_diff(scrubbed.text, rewritten)
    word_count = len(rewritten.split())

    warnings: list[str] = []
    # max_words is a soft target the model can and does ignore -- an
    # observed run returned 190 words against a 60-word cap. Surface the
    # miss rather than letting the caller assume the cap was honoured.
    if inp.max_words and word_count > inp.max_words * 1.2:
        warnings.append(
            f"length target missed: asked for <={inp.max_words} words, got "
            f"{word_count}. The cap is a soft instruction, not enforced -- "
            f"re-call with a lower target or trim it yourself"
        )
    if scrubbed.changed:
        warnings.append(
            f"masked before sending to OpenAI: {scrubbed.redactions} -- the "
            f"rewrite contains [REDACTED:...] placeholders where these were"
        )
    if not fidelity["numbers_intact"]:
        warnings.append(
            "figures changed between source and rewrite (see "
            "numeric_fidelity) -- show the user the diff before they use it"
        )
    if finish == "length":
        warnings.append(
            "output hit the token budget and may be cut off mid-sentence"
        )

    return ToolResult(
        output={
            "ok": True,
            "rewritten_text": rewritten,
            "audience": inp.audience,
            "model": model,
            "usage": usage,
            "estimated_cost_usd": round(cost, 6),
            "numeric_fidelity": fidelity,
            "warnings": warnings,
            "attempts": attempts,
            "source_chars": len(inp.text),
            "rewritten_chars": len(rewritten),
            "rewritten_words": word_count,
        }
    )
