"""Vision-judge for ASCII candidates.

Sends every rendered candidate to Gemini in a single multimodal JSON call,
along with optional prior-iteration context images, and returns a ranking
as :class:`JudgeResult`. See ``specs/06-judge.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import PIL.Image
from google.genai import errors as genai_errors
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

_SYSTEM_TEMPLATE_TEXT = """\
You are evaluating ASCII art renderings of "{subject}".
You will see {n} candidate drawings as raw ASCII text (no images attached).
Score each 0.0–1.0 on: recognizability of the subject, composition,
proportions, and use of negative space — judging from the raw text alone.
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


def _build_user_message(
    subject: str,
    n: int,
    trimmed: list[HistoryEntry],
    *,
    include_images: bool = True,
    texts: list[str] | None = None,
) -> str:
    lines = [f"Subject to draw: {subject}", ""]
    if include_images:
        lines.append(f"Images 0..{n - 1} are the {n} candidates to rank (in that order).")
    else:
        lines.append(
            f"Rank the following {n} ASCII drawings by raw text alone — no images attached."
        )
        for i, text in enumerate(texts or []):
            lines.extend(["", f"Candidate {i}:", "~~~", text, "~~~"])
    if trimmed:
        if include_images:
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
        else:
            lines.extend(
                [
                    "",
                    "Prior iterations (oldest first — critique only, no images attached):",
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


def _api_fallback_result(n: int, reason: str) -> JudgeResult:
    return JudgeResult(
        scores=[
            CandidateScore(index=i, score=0.5, why="fallback: judge degraded")
            for i in range(n)
        ],
        best_index=0,
        worst_index=max(n - 1, 0),
        critique=f"judge degraded: {reason}",
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
    include_images: bool = True,
    texts: list[str] | None = None,
) -> JudgeResult:
    """Rank ``images`` with one (multimodal or text-only) Gemini call.

    History is trimmed to the most recent :data:`_HISTORY_LIMIT` entries
    and appended to the image payload after the candidates. On persistent
    ``ValidationError`` from the SDK, returns a uniform fallback result
    and logs the degradation — never raises, so the loop stays alive.

    When ``include_images=False`` (text-judge ablation mode per
    ``specs/06-judge.md``), no images are attached: ``texts`` is required
    and each candidate's raw ASCII is inlined in the user message. Prior
    iterations' image attachments are also omitted, but their textual
    critique summaries are still included. Used by ``--ablate-judge`` to
    log the vision-vs-text top-3 delta on identical candidate sets.
    """
    if include_images:
        n = len(images)
    else:
        if texts is None:
            raise ValueError("rank(include_images=False) requires `texts` to be provided.")
        n = len(texts)
    trimmed = _trim_history(history)
    template = _SYSTEM_TEMPLATE if include_images else _SYSTEM_TEMPLATE_TEXT
    system = template.format(subject=subject, n=n, system_prompt=system_prompt)
    user = _build_user_message(
        subject, n, trimmed, include_images=include_images, texts=texts
    )
    payload: list[PIL.Image.Image] | None
    if include_images:
        payload = list(images) + [h.best_image for h in trimmed]
    else:
        payload = None

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
    except genai_errors.APIError as exc:
        _log.warning("judge degraded on APIError: %s", exc)
        return _api_fallback_result(n, str(exc))
