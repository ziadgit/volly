"""Prompt rewriter: judge critique → new artist system prompt.

Runs the rewriter step of the Volly loop. Takes the current artist system
prompt plus the judge's critique and suggestions, calls Gemini at High
thinking, and enforces the invariants from ``specs/07-rewriter.md`` on
the returned prompt.
"""

from __future__ import annotations

import logging
import re

from google.genai import errors as genai_errors

from volly.gemini_client import GeminiClient, Thinking
from volly.judge import JudgeResult

_log = logging.getLogger(__name__)

_MAX_LEN = 4000
_ANCHOR = "You are an ASCII artist."
_MAX_SUBJECT_MENTIONS = 2

_SYSTEM_TEMPLATE = """\
You are improving a system prompt used by an ASCII artist.
The artist's task is to draw "{subject}" — but the new prompt should
generalize to other subjects too.

You will receive:
- the current system prompt
- a critique from a vision judge
- specific suggestions from the judge

Produce the new system prompt. Rules:
- Start with "You are an ASCII artist."
- ≤ 4000 chars.
- Mention "{subject}" at most twice.
- Prefer concrete, transferable techniques over subject-specific recipes.
- Keep useful instructions from the current prompt; integrate the new ones.
- Output ONLY the new system prompt. No commentary, no fences."""


def _build_user_message(
    current_prompt: str, judge_result: JudgeResult, subject: str
) -> str:
    suggestions = judge_result.prompt_suggestions or []
    if suggestions:
        bullets = "\n".join(f"- {s}" for s in suggestions)
    else:
        bullets = "- (no specific suggestions returned)"
    return (
        f"Subject the artist is currently drawing: {subject}\n\n"
        "Current system prompt:\n"
        "```\n"
        f"{current_prompt}\n"
        "```\n\n"
        "Judge critique:\n"
        f"{judge_result.critique}\n\n"
        "Judge suggestions:\n"
        f"{bullets}\n\n"
        "Now produce the new system prompt."
    )


def _count_subject_mentions(text: str, subject: str) -> int:
    if not subject:
        return 0
    pattern = rf"\b{re.escape(subject)}\b"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _enforce_invariants(raw: str, subject: str) -> str:
    out = raw.strip()
    if not out.startswith(_ANCHOR):
        _log.info("rewriter: anchor missing, injecting prefix")
        out = f"{_ANCHOR} {out}" if out else _ANCHOR
    if len(out) > _MAX_LEN:
        _log.warning("rewriter: output %d chars, truncating to %d", len(out), _MAX_LEN)
        out = out[:_MAX_LEN]
    mentions = _count_subject_mentions(out, subject)
    if mentions > _MAX_SUBJECT_MENTIONS:
        _log.warning(
            "rewriter: subject %r mentioned %d times (max %d) — possible overfitting",
            subject,
            mentions,
            _MAX_SUBJECT_MENTIONS,
        )
    return out


async def rewrite(
    client: GeminiClient,
    current_prompt: str,
    judge_result: JudgeResult,
    subject: str,
    *,
    thinking: Thinking = Thinking.HIGH,
) -> str:
    """Rewrite the artist's system prompt given a judge critique.

    Returns a new system prompt with the invariants from
    ``specs/07-rewriter.md`` enforced post-hoc: anchor prefix, hard
    length cap, and a logged warning when the subject is over-mentioned.
    """
    system = _SYSTEM_TEMPLATE.format(subject=subject)
    user = _build_user_message(current_prompt, judge_result, subject)
    try:
        raw = await client.text(system, user, thinking=thinking)
    except genai_errors.APIError as exc:
        _log.info("rewriter degraded: %s; keeping prior prompt", exc)
        return current_prompt
    return _enforce_invariants(raw, subject)
