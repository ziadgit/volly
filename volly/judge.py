"""Vision-judge for ASCII candidates.

Sends every rendered candidate to Gemini in a single multimodal JSON call,
along with optional prior-iteration context images, and returns a ranking
as :class:`JudgeResult`. See ``specs/06-judge.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import PIL.Image
from pydantic import BaseModel, Field, ValidationError

from volly.gemini_client import GeminiClient, Thinking

_log = logging.getLogger(__name__)

_HISTORY_LIMIT = 4


@dataclass(frozen=True)
class HistoryEntry:
    """One prior iteration's snapshot, as the judge consumes it.

    Defined here so the judge can be reasoned about independently of the
    not-yet-built ``volly.state`` module. :class:`RunHistory` will
    project its records onto this shape when the loop is wired.
    """

    iter_index: int
    best_image: PIL.Image.Image
    critique: str
    top_score: float


class CandidateScore(BaseModel):
    index: int
    score: float = Field(ge=0.0, le=1.0)
    why: str


class JudgeResult(BaseModel):
    scores: list[CandidateScore]
    best_index: int
    worst_index: int
    critique: str
    prompt_suggestions: list[str]


_SYSTEM_TEMPLATE = """\
You are evaluating ASCII art renderings of "{subject}".
You will see {n} candidate images.
Score each 0.0–1.0 on: recognizability of the subject, composition,
proportions, and use of negative space.
Identify the best and worst.
Then suggest 1–3 concrete improvements to the artist's system prompt —
the prompt is included below. Be specific, not generic.

Current artist system prompt:
---
{system_prompt}
---"""


def _trim_history(history: list[HistoryEntry] | None) -> list[HistoryEntry]:
    if not history:
        return []
    return list(history)[-_HISTORY_LIMIT:]


def _build_user_message(subject: str, n: int, trimmed: list[HistoryEntry]) -> str:
    lines = [
        f"Subject to draw: {subject}",
        "",
        f"Images 0..{n - 1} are the {n} candidates to rank (in that order).",
    ]
    if trimmed:
        history_offsets = ", ".join(
            f"image {n + offset} = prior iteration {h.iter_index} best"
            for offset, h in enumerate(trimmed)
        )
        lines.extend(
            [
                f"Then, after the candidates, prior best images follow: {history_offsets}.",
                "Use these to track progress and avoid repeating earlier critiques.",
                "",
                "Prior iterations (oldest first):",
            ]
        )
        for h in trimmed:
            crit = " ".join(h.critique.split())
            lines.append(
                f"  - Iteration {h.iter_index} — top score {h.top_score:.2f} — critique: {crit}"
            )
    lines.extend(
        [
            "",
            "Return JSON: per-candidate scores (0.0-1.0) each with a one-sentence "
            "rationale, the best and worst indices, a 2-4 sentence critique, and "
            "1-3 concrete prompt-improvement suggestions.",
        ]
    )
    return "\n".join(lines)


def _fallback_result(n: int) -> JudgeResult:
    return JudgeResult(
        scores=[
            CandidateScore(index=i, score=0.5, why="fallback: judge returned invalid JSON")
            for i in range(n)
        ],
        best_index=0,
        worst_index=max(n - 1, 0),
        critique="Judge fell back to uniform scoring after repeated schema-validation failures.",
        prompt_suggestions=[],
    )


async def rank(
    client: GeminiClient,
    subject: str,
    system_prompt: str,
    images: list[PIL.Image.Image],
    *,
    history: list[HistoryEntry] | None = None,
    thinking: Thinking = Thinking.HIGH,
) -> JudgeResult:
    """Rank ``images`` with one multimodal Gemini call.

    History is trimmed to the most recent :data:`_HISTORY_LIMIT` entries
    and appended to the image payload after the candidates. On persistent
    ``ValidationError`` from the SDK, returns a uniform fallback result
    and logs the degradation — never raises, so the loop stays alive.
    """
    n = len(images)
    trimmed = _trim_history(history)
    system = _SYSTEM_TEMPLATE.format(subject=subject, n=n, system_prompt=system_prompt)
    user = _build_user_message(subject, n, trimmed)
    payload = list(images) + [h.best_image for h in trimmed]

    try:
        return await client.json(
            system,
            user,
            JudgeResult,
            images=payload,
            thinking=thinking,
        )
    except ValidationError as exc:
        _log.warning("judge fell back to uniform scoring: %s", exc)
        return _fallback_result(n)
