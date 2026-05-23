"""Parallel ASCII candidate generation.

Given a system prompt and a subject, dispatch ``k`` independent Gemini
calls via ``asyncio.gather`` and return the survivors as
:class:`Candidate` objects with code fences stripped. Diversity comes
from sampling temperature, not prompt variation. See
``specs/04-actor.md``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from volly.gemini_client import GeminiClient, Thinking

_USER_TEMPLATE = (
    "Subject: {subject}\n\n"
    "Draw it as ASCII art. Respond with only the ASCII drawing, no commentary,\n"
    "no code fences."
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """One actor output, ready for rendering."""

    text: str
    index: int
    raw: str


def _strip_fences(raw: str) -> str:
    """Remove a single leading/trailing triple-backtick fence if present.

    Handles ```````, `````ascii``, and any
    language tag. Leaves text untouched if no fence is found. Preserves
    interior whitespace — only the fence lines themselves are removed.
    """
    if "```" not in raw:
        return raw
    stripped = raw.strip("\n")
    lines = stripped.split("\n")
    if not lines or not lines[0].lstrip().startswith("```"):
        return raw
    body_start = 1
    body_end = len(lines)
    if body_end > body_start and lines[-1].strip() == "```":
        body_end -= 1
    return "\n".join(lines[body_start:body_end])


async def generate(
    client: GeminiClient,
    system_prompt: str,
    subject: str,
    *,
    k: int = 8,
    thinking: Thinking = Thinking.LOW,
    temperature: float = 1.0,
) -> list[Candidate]:
    """Generate up to ``k`` ASCII candidates in parallel.

    Failed SDK calls are dropped silently; the returned list may be
    shorter than ``k``. Candidate ``index`` reflects the original
    dispatch slot (0..k-1) so callers can correlate with logs even when
    some slots dropped out.
    """
    if k <= 0:
        return []

    user = _USER_TEMPLATE.format(subject=subject)
    coros = [
        client.text(system_prompt, user, thinking=thinking, temperature=temperature)
        for _ in range(k)
    ]
    results = await asyncio.gather(*coros, return_exceptions=True)

    candidates: list[Candidate] = []
    failures = 0
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            failures += 1
            _log.warning("actor candidate %d failed: %s", i, result)
            continue
        candidates.append(Candidate(text=_strip_fences(result), index=i, raw=result))

    if failures and len(candidates) < k:
        _log.info("actor produced %d/%d candidates (%d failed)", len(candidates), k, failures)
    return candidates
