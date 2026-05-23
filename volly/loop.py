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
    {
        "cat",
        "house",
        "fish",
        "coffee cup",
        "smiley",
        "sailboat",
        "tree",
        "heart",
        "star",
        "capybara",
        "owl",
        "mushroom",
    }
)

# Per-subject "rehearsed" prompts used by ``--demo`` to pre-warm the evolving arm.
# Each is roughly what the rewriter would converge toward after a few iterations:
# subject-specific structural guidance + a small character set + a centering note.
# Control arm stays on SEED_PROMPT so the audience still sees flat-seed baseline.
DEMO_PROMPTS: dict[str, str] = {
    "cat": (
        "You are an ASCII artist. Draw a cat in approximately 8-12 lines and "
        "16-24 columns. Show two pointed ears at the top, two round eyes spaced "
        "apart on the same row, a small nose and mouth below, and whiskers "
        "extending horizontally from the cheeks. A simple curved body and tail "
        "underneath help. Stick to a small, consistent character set of "
        "()<>^v.o\\/_- and avoid mixing in dense block characters. Leave a "
        "blank line above and below the figure for centering."
    ),
    "house": (
        "You are an ASCII artist. Draw a house in approximately 8-12 lines and "
        "18-26 columns. Start with a triangular roof made from slashes "
        "converging at a peak, then a rectangular wall body below it built "
        "from straight lines. Include at least one window (a small square) and "
        "a door (a taller rectangle near the bottom). Keep all vertical edges "
        "aligned column-wise so the walls do not skew. Use a consistent "
        "character set of /\\|_- and basic punctuation; avoid dense fill "
        "characters. Center the figure with blank padding rows."
    ),
    "fish": (
        "You are an ASCII artist. Draw a fish in approximately 5-9 lines and "
        "18-28 columns. Build an oval or teardrop body oriented horizontally, "
        "with a triangular tail fin on one side made from converging "
        "diagonals. Place a small eye (a dot or circle) near the front of the "
        "body and a curved mouth in front of it. Optional: one or two small "
        "fins along the body. Use light characters like ()<>{}.,o*- and keep "
        "the silhouette closed so the body reads as a single shape. Center "
        "with leading and trailing blank rows."
    ),
    "coffee cup": (
        "You are an ASCII artist. Draw a coffee cup in approximately 7-11 "
        "lines and 14-22 columns. Start with one or two short wavy lines at "
        "the top to suggest rising steam, then a horizontal rim, then "
        "vertical cup walls that curve slightly inward at the base. Attach a "
        "small loop handle to one side of the cup, drawn with ) or }. "
        "Optionally add a saucer line beneath the cup. Use a consistent "
        "character set of ()|/\\~_-' and keep wall columns vertically "
        "aligned. Pad with blank lines for centering."
    ),
    "smiley": (
        "You are an ASCII artist. Draw a smiley face in approximately 7-11 "
        "lines and 14-22 columns. Build a round outline using parentheses and "
        "slashes for the curved sides and underscores or hyphens for the top "
        "and bottom arcs. Inside, place two evenly-spaced eyes on the same "
        "row (use o or *) and a curved mouth below them spanning roughly the "
        "eye width. Keep the face symmetric left-to-right and the inner "
        "features centered within the outline. Use a consistent light "
        "character set and pad with blank lines for centering."
    ),
    "sailboat": (
        "You are an ASCII artist. Draw a sailboat in approximately 8-12 lines "
        "and 18-28 columns. At the top, build a triangular sail using "
        "converging diagonals around a vertical mast in the middle. Below the "
        "sail, draw a horizontal deck line, then a hull shaped like a shallow "
        "trapezoid or smile underneath. Add a few short wavy lines below the "
        "hull to suggest water. Keep the mast a single column so the sail "
        "balances around it. Use /\\|_~-' and similar light characters; avoid "
        "dense fills. Pad with blank rows for centering."
    ),
    "tree": (
        "You are an ASCII artist. Draw a tree in approximately 8-12 lines and "
        "14-22 columns. Build a roughly triangular or rounded canopy at the "
        "top using leaf-like characters such as * & %, widening from the apex "
        "downward over several rows. Below the canopy, draw a short trunk "
        "one or two columns wide using | or ||, centered horizontally "
        "beneath the canopy. Keep the trunk aligned with the canopy's "
        "vertical axis so the figure does not lean. Use a consistent "
        "character set and pad with blank rows for centering."
    ),
    "heart": (
        "You are an ASCII artist. Draw a heart in approximately 6-10 lines "
        "and 12-20 columns. Build two rounded lobes at the top using "
        "parentheses or slashes, joined across the middle, then taper the "
        "sides downward with diagonals that converge to a single point at "
        "the bottom. Keep the figure left-right symmetric column by column. "
        "Use a small character set of ()/\\_-* and avoid mixing in dense "
        "block characters. Pad with leading and trailing blank rows so the "
        "figure sits centered in the canvas."
    ),
    "star": (
        "You are an ASCII artist. Draw a five-pointed star in approximately "
        "7-11 lines and 14-22 columns. Start with a single point at the top, "
        "two arms reaching outward and slightly downward, then two more arms "
        "reaching further down and outward at the base — five tips total. "
        "Build the silhouette from converging diagonals using / and \\, with "
        "the outline closed so the shape reads as one figure. Keep it "
        "left-right symmetric column by column. Pad with blank rows above "
        "and below for centering."
    ),
    "capybara": (
        "You are an ASCII artist. Draw a capybara in approximately 10-16 "
        "lines and 28-44 columns — the larger canvas lets you suggest fur "
        "texture and body mass that smaller subjects don't need. Build a "
        "rounded barrel-shaped body sitting low on short stubby legs, with "
        "a blocky muzzle, small rounded ears on top of the head, and a "
        "single visible eye. Use a tonal palette `. , : ; - = + * # @` "
        "(roughly light to dark) to suggest fur shading across the body — "
        "lighter on the belly and around the muzzle, denser along the back "
        "and shadow side. Keep the outline closed so the silhouette reads "
        "as one solid animal. Pad with blank rows for centering."
    ),
    "owl": (
        "You are an ASCII artist. Draw an owl in approximately 10-14 lines "
        "and 18-28 columns — the larger canvas lets the round disc-faced "
        "silhouette and plumage texture read clearly. Build a pear-shaped "
        "body sitting upright on short stubby legs or a perched bar, with "
        "two wide round eyes high on the head, a small triangular beak "
        "centered between them, and small angled ear tufts at the top "
        "corners. Use a tonal palette `. , : ; - = + * # @` (roughly light "
        "to dark) to suggest feather shading — lighter on the breast and "
        "facial disc, denser along the wings and crown. Keep the silhouette "
        "closed so the figure reads as one solid shape. Pad with blank rows "
        "for centering."
    ),
    "mushroom": (
        "You are an ASCII artist. Draw a mushroom in approximately 10-14 "
        "lines and 16-26 columns — the larger canvas lets you shade the "
        "cap's dome and suggest gill texture under it. Build a rounded "
        "dome-shaped cap on top, slightly overhanging a short straight stem "
        "below it, and ground the figure with a short ground line or two "
        "small grass tufts at the base. Use a tonal palette `. , : ; - = + "
        "* # @` (roughly light to dark) for cap shading — lightest at the "
        "crown, denser along the lower curve and the underside; a few small "
        "dots scattered on the cap are optional. Keep the outline closed "
        "and pad with blank rows for centering."
    ),
}

_log = logging.getLogger(__name__)

_JUDGE_HISTORY_LIMIT = 4

# Tier preset bundles (specs/02-loop.md §"Tier presets"). Selected via
# ``--tier {free,paid}``; explicit flags override individual preset values.
# When ``--tier`` is omitted, no preset is applied and raw argparse defaults
# stand (rpm=None→env/30, candidates=8, no_control=False, max_retry_wait=90).
_TIER_PRESETS: dict[str, dict[str, object]] = {
    "free": {
        "rpm": 4,
        "candidates": 3,
        "no_control": True,
        "max_retry_wait_s": 3700.0,
    },
    "paid": {
        "rpm": 900,
        "candidates": 8,
        "no_control": False,
        "max_retry_wait_s": 90.0,
    },
}

# Iteration-1 wedge handling (specs/02-loop.md §"Iteration-1 wedge handling").
# Iter 1 has no prior best to pad from, so a zero-candidate arm cannot be
# repaired locally — only a fresh attempt helps. Sleep between retries is a
# module-level seam so tests can monkeypatch it to skip the 60s wait.
_ITER_ONE_MAX_RETRIES = 2
_ITER_ONE_RETRY_SLEEP_S = 60.0

_ITER_ONE_WEDGED_BANNER = (
    "iter 1 wedged — likely rate-limited; try --rpm=<lower> or upgrade tier"
)


class IterationOneWedgedError(RuntimeError):
    """Iteration 1 failed to produce ≥1 candidate per arm after retries.

    Raised after ``_ITER_ONE_MAX_RETRIES + 1`` attempts when at least one arm
    still returns zero candidates. ``main`` catches this, prints
    ``_ITER_ONE_WEDGED_BANNER`` to stderr, and exits with code 3.
    """


async def _iter_one_retry_sleep(seconds: float) -> None:
    """Sleep between iter-1 retries. Module seam so tests can fast-forward."""
    await asyncio.sleep(seconds)


@dataclass
class LoopConfig:
    subject: str | None = None
    iterations: int = 8
    candidates: int = 8
    no_control: bool = False
    out_dir: Path | None = None
    actor_thinking: Thinking = Thinking.LOW
    judge_thinking: Thinking = Thinking.HIGH
    rewriter_thinking: Thinking = Thinking.HIGH
    ablate_judge: bool = False
    demo_mode: bool = False
    rpm: int | None = None
    max_retry_wait_s: float = 90.0
    resume: Path | None = None


def validate_subject(subject: str) -> str:
    """Lowercase and confirm ``subject`` is on the curated list."""
    normalized = subject.strip().lower()
    if normalized not in CURATED_SUBJECTS:
        choices = ", ".join(sorted(CURATED_SUBJECTS))
        raise ValueError(f"subject {subject!r} not in curated list: {choices}")
    return normalized


def _last_complete_iter(history: RunHistory, *, expects_control: bool) -> int:
    """Highest ``iter_index`` whose recorded arms cover the configured set.

    With ``expects_control=False`` a single ``evolving`` record satisfies an
    iteration; with ``expects_control=True`` both ``evolving`` and ``control``
    must be present. Returns ``0`` when no iteration qualifies — caller
    interprets that as "start fresh in the existing run_dir".
    """
    expected: set[str] = {"evolving"}
    if expects_control:
        expected = {"evolving", "control"}
    by_iter: dict[int, set[str]] = {}
    for record in history.iterations:
        by_iter.setdefault(record.iter_index, set()).add(record.arm)
    complete = [idx for idx, arms in by_iter.items() if expected.issubset(arms)]
    return max(complete) if complete else 0


def _load_resume_state(resume_dir: Path) -> RunHistory:
    """Load ``state.json`` for resume; raise ``ValueError`` on miss/corrupt.

    ``main`` translates ``ValueError`` to rc=2 with a stderr banner, so the
    operator gets the same exit code path as ``--subject dragon``.
    """
    state_path = resume_dir / "state.json"
    if not state_path.exists():
        raise ValueError(f"--resume: {state_path} does not exist")
    try:
        return RunHistory.load(state_path)
    except Exception as exc:
        raise ValueError(f"--resume: failed to load {state_path}: {exc}") from exc


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
    ablate_judge: bool = False,
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
        if ablate_judge:
            await _log_text_judge_delta(
                client=client,
                arm=arm,
                iter_index=iter_index,
                subject=subject,
                system_prompt=system_prompt,
                images=images,
                cands=cands,
                judge_history=judge_history,
                judge_thinking=judge_thinking,
                vision_result=judge_result,
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


async def _log_text_judge_delta(
    *,
    client: GeminiClient,
    arm: str,
    iter_index: int,
    subject: str,
    system_prompt: str,
    images: list[PIL.Image.Image],
    cands: list[Candidate],
    judge_history: list[HistoryEntry],
    judge_thinking: Thinking,
    vision_result: JudgeResult,
) -> None:
    """Run a text-only judge on the same candidates, log the top-3 score delta.

    Per ``specs/06-judge.md``'s ablation hook + the P2 fix_plan item:
    measures how much the vision channel adds over raw-ASCII judgment on
    an identical candidate set. Failures are logged and swallowed —
    ablation must never crash the live loop.
    """
    try:
        text_result = await judge.rank(
            client,
            subject,
            system_prompt,
            images,
            history=judge_history,
            thinking=judge_thinking,
            include_images=False,
            texts=[c.text for c in cands],
        )
    except Exception:
        _log.exception("ablation iter %d arm %s: text-judge failed", iter_index, arm)
        return
    vision_top3 = win_rate([s.score for s in vision_result.scores])
    text_top3 = win_rate([s.score for s in text_result.scores])
    _log.info(
        "ablation iter %d arm %s: vision_top3=%.3f text_top3=%.3f delta=%+.3f",
        iter_index,
        arm,
        vision_top3,
        text_top3,
        vision_top3 - text_top3,
    )


def _build_arm_tasks(
    *,
    client: GeminiClient,
    config: LoopConfig,
    history: RunHistory,
    iter_index: int,
    iter_dir: Path,
    subject: str,
    evolving_prompt: str,
    control_prompt: str,
) -> list:
    """Build the per-iteration list of ``_run_arm`` coroutines.

    Module-level (not nested in ``run``) so the iter-1 retry while-loop can
    re-invoke it without tripping ruff's B023 (closure-over-loop-variable)
    check — each call gets a fresh batch of coroutines bound to the current
    iteration's prompt + dir.
    """
    tasks = [
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
            ablate_judge=config.ablate_judge,
        )
    ]
    if not config.no_control:
        tasks.append(
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
                ablate_judge=config.ablate_judge,
            )
        )
    return tasks


async def run(config: LoopConfig, *, client: GeminiClient | None = None) -> RunHistory:
    """Run the full evolving + control loop. Returns a populated ``RunHistory``."""
    if config.resume is not None:
        history = _load_resume_state(config.resume)
        subject = history.subject
        if config.subject is not None and validate_subject(config.subject) != subject:
            raise ValueError(
                f"--resume subject mismatch: state.json has {subject!r}, "
                f"--subject was {config.subject!r}"
            )
        started_at = history.started_at
        run_dir = config.resume
        n = _last_complete_iter(history, expects_control=not config.no_control)
        history.iterations = [r for r in history.iterations if r.iter_index <= n]
        start_iter = n + 1
        if n >= 1:
            evolving_record = next(
                r for r in history.iterations
                if r.iter_index == n and r.arm == "evolving"
            )
            evolving_prompt = evolving_record.system_prompt
            _log.info(
                "resume: continuing %s at iter %d (loaded %d complete iters from %s)",
                subject, start_iter, n, run_dir,
            )
        else:
            evolving_prompt = (
                DEMO_PROMPTS[subject] if config.demo_mode else SEED_PROMPT
            )
            _log.info(
                "resume: no complete iterations in %s; starting fresh at iter 1 in same run_dir",
                run_dir,
            )
    else:
        if config.subject is None:
            raise ValueError("subject is required when --resume is not set")
        subject = validate_subject(config.subject)
        started_at = datetime.now(UTC)
        run_dir = config.out_dir or _default_run_dir(subject, started_at)
        history = RunHistory(
            subject=subject,
            started_at=started_at,
            run_dir=run_dir,
            seed_prompt=SEED_PROMPT,
        )
        evolving_prompt = DEMO_PROMPTS[subject] if config.demo_mode else SEED_PROMPT
        start_iter = 1
        if config.demo_mode:
            _log.info(
                "demo mode: evolving arm pre-warmed with rehearsed prompt for %r",
                subject,
            )

    run_dir.mkdir(parents=True, exist_ok=True)

    owns_client = client is None
    if client is None:
        client = GeminiClient(
            rpm=config.rpm, max_retry_wait_s=config.max_retry_wait_s
        )

    control_prompt = SEED_PROMPT

    try:
        for iter_index in range(start_iter, config.iterations + 1):
            iter_dir = run_dir / f"iter-{iter_index:02d}"
            _log.info("iter %d/%d starting", iter_index, config.iterations)

            attempts_remaining = (
                _ITER_ONE_MAX_RETRIES + 1 if iter_index == 1 else 1
            )
            while True:
                arm_results = await asyncio.gather(
                    *_build_arm_tasks(
                        client=client,
                        config=config,
                        history=history,
                        iter_index=iter_index,
                        iter_dir=iter_dir,
                        subject=subject,
                        evolving_prompt=evolving_prompt,
                        control_prompt=control_prompt,
                    )
                )
                attempts_remaining -= 1

                if iter_index != 1:
                    break

                empty = [r for r in arm_results if not r.candidates]
                if not empty:
                    break

                if attempts_remaining == 0:
                    raise IterationOneWedgedError(
                        f"iter 1 wedged: {len(empty)} arm(s) produced 0 candidates "
                        f"after {_ITER_ONE_MAX_RETRIES + 1} attempts"
                    )

                for record in empty:
                    _log.warning(
                        "iter 1 produced %d/%d candidates (arm=%s); "
                        "retrying iter 1 in %.0fs (RPM=%d)",
                        len(record.candidates),
                        config.candidates,
                        record.arm,
                        _ITER_ONE_RETRY_SLEEP_S,
                        client.rpm,
                    )
                await _iter_one_retry_sleep(_ITER_ONE_RETRY_SLEEP_S)

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
    parser.add_argument(
        "--subject",
        default=None,
        help=(
            "curated subject (e.g. cat, tree, star); required unless --resume "
            "is set, in which case the subject is loaded from state.json"
        ),
    )
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="candidates per iteration (default 8 unless overridden by --tier preset)",
    )
    parser.add_argument(
        "--no-control",
        action="store_true",
        default=None,
        help="skip the static-control arm to halve API spend (default off unless --tier free)",
    )
    parser.add_argument(
        "--ablate-judge",
        action="store_true",
        help="also run a text-only judge per iteration and log vision-vs-text top-3 delta",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="pre-warm evolving arm with a rehearsed subject-specific prompt (control stays on seed)",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=None,
        help="cap Gemini requests per minute (default: GEMINI_RPM env, else 30; for free tier use --tier free which sets rpm=4, one below the 5 RPM ceiling for safety margin)",
    )
    parser.add_argument(
        "--max-retry-wait",
        type=float,
        default=None,
        help=(
            "per-call cap (seconds) on honoring server retryDelay; above the "
            "cap the client enters patient mode (warn + heartbeat sleep + "
            "retry, no crash). Default 90; set ~3700 to wait through an hourly "
            "quota reset"
        ),
    )
    parser.add_argument(
        "--tier",
        choices=sorted(_TIER_PRESETS.keys()),
        default=None,
        help=(
            "operational preset bundle: 'free' = rpm=4 candidates=3 --no-control "
            "max-retry-wait=3700 (sponsorship / no-billing); 'paid' = rpm=900 "
            "candidates=8 with control max-retry-wait=90 (Tier 1+). Omit to keep "
            "argparse defaults. Explicit flags override individual preset values"
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help="override VOLLY_RUN_DIR")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "resume an existing run from <run-dir>; loads state.json, continues "
            "at the iteration after the last fully-completed one, reuses the "
            "run_dir (ignores --out and the subject from state.json wins). "
            "See specs/02-loop.md §Resumable runs"
        ),
    )
    return parser.parse_args(argv)


def _resolve_loop_config(args: argparse.Namespace) -> LoopConfig:
    """Build a ``LoopConfig`` from argparse ``args``, applying ``--tier`` preset.

    Resolution rule (spec 02 §"Tier presets"): explicit flags (non-None at the
    parser level) win. If ``--tier`` selects a preset, unset flags fall back to
    the preset value. If ``--tier`` is omitted, unset flags fall back to the
    raw argparse default — preserving the pre-tier-preset behavior bit-for-bit.
    """
    preset = _TIER_PRESETS.get(args.tier) if args.tier else None

    def pick(flag_value: object, preset_key: str, raw_default: object) -> object:
        if flag_value is not None:
            return flag_value
        if preset is not None:
            return preset[preset_key]
        return raw_default

    return LoopConfig(
        subject=args.subject,
        iterations=args.iterations,
        candidates=pick(args.candidates, "candidates", 8),  # type: ignore[arg-type]
        no_control=pick(args.no_control, "no_control", False),  # type: ignore[arg-type]
        out_dir=args.out,
        ablate_judge=args.ablate_judge,
        demo_mode=args.demo,
        rpm=pick(args.rpm, "rpm", None),  # type: ignore[arg-type]
        max_retry_wait_s=pick(args.max_retry_wait, "max_retry_wait_s", 90.0),  # type: ignore[arg-type]
        resume=args.resume,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        config = _resolve_loop_config(args)
        history = asyncio.run(run(config))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except IterationOneWedgedError:
        print(_ITER_ONE_WEDGED_BANNER, file=sys.stderr)
        return 3
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
