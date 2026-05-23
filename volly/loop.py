"""Loop orchestration — generate → render → judge → rewrite.

Owns the iteration cycle and the two-arm (evolving vs. static control)
comparison from ``specs/02-loop.md``. The only module that touches actor,
renderer, judge, rewriter, state, and persistence together — every other
module stays single-purpose.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import PIL.Image

from volly import actor, judge, renderer, rewriter
from volly.actor import Candidate
from volly.gemini_client import GeminiClient, Thinking
from volly.judge import HistoryEntry, JudgeResult
from volly.state import IterationRecord, RunHistory, win_rate

SEED_PROMPT = "You are an ASCII artist. Draw the requested subject."

CURATED_SUBJECTS: frozenset[str] = frozenset(
    {"cat", "house", "fish", "coffee cup", "smiley", "sailboat", "tree", "heart", "star"}
)

_log = logging.getLogger(__name__)

_JUDGE_HISTORY_LIMIT = 4


@dataclass
class LoopConfig:
    subject: str
    iterations: int = 8
    candidates: int = 8
    no_control: bool = False
    out_dir: Path | None = None
    actor_thinking: Thinking = Thinking.LOW
    judge_thinking: Thinking = Thinking.HIGH
    rewriter_thinking: Thinking = Thinking.HIGH


def validate_subject(subject: str) -> str:
    """Lowercase and confirm ``subject`` is on the curated list."""
    normalized = subject.strip().lower()
    if normalized not in CURATED_SUBJECTS:
        choices = ", ".join(sorted(CURATED_SUBJECTS))
        raise ValueError(f"subject {subject!r} not in curated list: {choices}")
    return normalized


def _slug(value: str) -> str:
    return value.replace(" ", "-")


def _default_run_dir(subject: str, started_at: datetime) -> Path:
    base = Path(os.environ.get("VOLLY_RUN_DIR", "runs"))
    stamp = started_at.strftime("%Y%m%dT%H%M%S")
    return base / f"{stamp}-{_slug(subject)}"


def _last_best_candidate(history: RunHistory, arm: str) -> Candidate | None:
    record = history.latest(arm)  # type: ignore[arg-type]
    if record is None or not record.candidates:
        return None
    best_index = record.judge.best_index
    for cand in record.candidates:
        if cand.index == best_index:
            return cand
    return record.candidates[0]


def _judge_history_for(history: RunHistory, arm: str) -> list[HistoryEntry]:
    entries: list[HistoryEntry] = []
    for record in history.iterations:
        if record.arm != arm:
            continue
        try:
            with PIL.Image.open(record.best_image_path) as src:
                img = src.copy()
        except (FileNotFoundError, PIL.UnidentifiedImageError):
            _log.warning(
                "judge history: cannot read %s, skipping iteration %d",
                record.best_image_path,
                record.iter_index,
            )
            continue
        scores = [s.score for s in record.judge.scores]
        top = max(scores) if scores else 0.0
        entries.append(
            HistoryEntry(
                iter_index=record.iter_index,
                best_image=img,
                critique=record.judge.critique,
                top_score=top,
            )
        )
    return entries[-_JUDGE_HISTORY_LIMIT:]


def _pad_candidates(
    cands: list[Candidate], k: int, prior_best: Candidate | None
) -> list[Candidate]:
    """Return exactly ``min(k, len(cands)+pad)`` candidates in slot order.

    Each surviving candidate keeps its dispatch ``index``. Empty slots are
    backfilled with ``prior_best`` when available; iteration 1 (no prior)
    simply yields a shorter list.
    """
    by_slot = {c.index: c for c in cands}
    padded: list[Candidate] = []
    for slot in range(k):
        if slot in by_slot:
            padded.append(by_slot[slot])
        elif prior_best is not None:
            padded.append(
                Candidate(text=prior_best.text, index=slot, raw=prior_best.raw)
            )
    return padded


async def _run_arm(
    client: GeminiClient,
    *,
    arm: str,
    iter_index: int,
    system_prompt: str,
    subject: str,
    k: int,
    actor_thinking: Thinking,
    judge_thinking: Thinking,
    judge_history: list[HistoryEntry],
    iter_dir: Path,
    prior_best: Candidate | None,
) -> IterationRecord:
    cands_raw = await actor.generate(
        client, system_prompt, subject, k=k, thinking=actor_thinking
    )
    cands = _pad_candidates(cands_raw, k, prior_best)

    arm_dir = iter_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / "prompt.txt").write_text(system_prompt)

    images: list[PIL.Image.Image] = []
    for cand in cands:
        img = renderer.render(cand.text)
        img.save(arm_dir / f"cand-{cand.index:02d}.png")
        images.append(img)

    if images:
        judge_result = await judge.rank(
            client,
            subject,
            system_prompt,
            images,
            history=judge_history,
            thinking=judge_thinking,
        )
    else:
        _log.warning("iter %d arm %s: zero candidates, fabricating empty judge", iter_index, arm)
        judge_result = JudgeResult(
            scores=[], best_index=0, worst_index=0, critique="no candidates", prompt_suggestions=[]
        )

    best_path = arm_dir / "best.png"
    best_index = judge_result.best_index
    if images:
        chosen = (
            images[best_index] if 0 <= best_index < len(images) else images[0]
        )
        chosen.save(best_path)

    scores = [s.score for s in judge_result.scores]
    return IterationRecord(
        iter_index=iter_index,
        arm=arm,  # type: ignore[arg-type]
        system_prompt=system_prompt,
        candidates=cands,
        judge=judge_result,
        best_image_path=best_path,
        win_rate=win_rate(scores),
    )


async def run(config: LoopConfig, *, client: GeminiClient | None = None) -> RunHistory:
    """Run the full evolving + control loop. Returns a populated ``RunHistory``."""
    subject = validate_subject(config.subject)
    started_at = datetime.now(UTC)
    run_dir = config.out_dir or _default_run_dir(subject, started_at)
    run_dir.mkdir(parents=True, exist_ok=True)

    owns_client = client is None
    if client is None:
        client = GeminiClient()

    history = RunHistory(
        subject=subject,
        started_at=started_at,
        run_dir=run_dir,
        seed_prompt=SEED_PROMPT,
    )

    evolving_prompt = SEED_PROMPT
    control_prompt = SEED_PROMPT

    try:
        for iter_index in range(1, config.iterations + 1):
            iter_dir = run_dir / f"iter-{iter_index:02d}"
            _log.info("iter %d/%d starting", iter_index, config.iterations)

            arm_tasks = [
                _run_arm(
                    client,
                    arm="evolving",
                    iter_index=iter_index,
                    system_prompt=evolving_prompt,
                    subject=subject,
                    k=config.candidates,
                    actor_thinking=config.actor_thinking,
                    judge_thinking=config.judge_thinking,
                    judge_history=_judge_history_for(history, "evolving"),
                    iter_dir=iter_dir,
                    prior_best=_last_best_candidate(history, "evolving"),
                )
            ]
            if not config.no_control:
                arm_tasks.append(
                    _run_arm(
                        client,
                        arm="control",
                        iter_index=iter_index,
                        system_prompt=control_prompt,
                        subject=subject,
                        k=config.candidates,
                        actor_thinking=config.actor_thinking,
                        judge_thinking=config.judge_thinking,
                        judge_history=_judge_history_for(history, "control"),
                        iter_dir=iter_dir,
                        prior_best=_last_best_candidate(history, "control"),
                    )
                )

            arm_results = await asyncio.gather(*arm_tasks)
            for record in arm_results:
                history.add(record)

            evolving_record = arm_results[0]
            evolving_prompt = await rewriter.rewrite(
                client,
                evolving_prompt,
                evolving_record.judge,
                subject,
                thinking=config.rewriter_thinking,
            )

            history.save()
    except Exception:
        _log.exception("loop crashed; persisting partial state")
        history.save()
        raise
    finally:
        if owns_client:
            close = getattr(client, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    _log.debug("client aclose raised; ignoring", exc_info=True)

    return history


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="volly",
        description="Self-improving ASCII art via system prompt learning.",
    )
    parser.add_argument("--subject", required=True, help="curated subject (e.g. cat, tree, star)")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="skip the static-control arm to halve API spend",
    )
    parser.add_argument("--out", type=Path, default=None, help="override VOLLY_RUN_DIR")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        config = LoopConfig(
            subject=args.subject,
            iterations=args.iterations,
            candidates=args.candidates,
            no_control=args.no_control,
            out_dir=args.out,
        )
        history = asyncio.run(run(config))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130

    latest = history.latest("evolving")
    if latest is not None:
        print(f"best-of-{config.candidates}: {latest.best_image_path}")
    print(f"run dir: {history.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
