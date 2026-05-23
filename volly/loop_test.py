"""Tests for ``volly.loop``. ``GeminiClient`` is fully stubbed — no network."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from volly import loop
from volly.actor import Candidate
from volly.gemini_client import GeminiClient, Thinking
from volly.judge import CandidateScore, JudgeResult
from volly.state import RunHistory


def _judge_result(n: int, *, best: int = 0) -> JudgeResult:
    return JudgeResult(
        scores=[
            CandidateScore(index=i, score=min(0.95, 0.4 + 0.05 * i), why=f"r{i}")
            for i in range(n)
        ],
        best_index=best,
        worst_index=n - 1,
        critique="ok",
        prompt_suggestions=["use thinner lines"],
    )


def _stub_client(
    *,
    actor_text: str = "/\\_/\\\n( o.o )\n > ^ <",
    rewrite_text: str = "You are an ASCII artist. Now improved.",
    judge_results: list[JudgeResult] | None = None,
) -> GeminiClient:
    client = MagicMock(spec=GeminiClient)

    async def text_router(system, user, *, thinking=Thinking.LOW, temperature=1.0):
        if thinking is Thinking.HIGH:
            return rewrite_text
        return actor_text

    client.text = AsyncMock(side_effect=text_router)
    client.json = AsyncMock(side_effect=list(judge_results or []))
    client.multimodal = AsyncMock()
    return client


def test_seed_prompt_matches_spec() -> None:
    assert loop.SEED_PROMPT == "You are an ASCII artist. Draw the requested subject."


def test_curated_subjects_match_overview_list() -> None:
    assert loop.CURATED_SUBJECTS == frozenset(
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
        }
    )


def test_validate_subject_normalizes_case_and_whitespace() -> None:
    assert loop.validate_subject("Cat") == "cat"
    assert loop.validate_subject("  coffee cup  ") == "coffee cup"


def test_validate_subject_rejects_off_list() -> None:
    with pytest.raises(ValueError):
        loop.validate_subject("dragon")


async def test_run_no_control_persists_state_and_artifacts(tmp_path: Path) -> None:
    client = _stub_client(
        judge_results=[_judge_result(8, best=2), _judge_result(8, best=5)],
    )
    config = loop.LoopConfig(
        subject="cat",
        iterations=2,
        candidates=8,
        no_control=True,
        out_dir=tmp_path,
    )

    history = await loop.run(config, client=client)

    assert len(history.iterations) == 2
    assert all(r.arm == "evolving" for r in history.iterations)
    # 8 actor.text calls per iter + 1 rewriter call per iter, 2 iters → 18 text calls
    assert client.text.await_count == 2 * (8 + 1)
    # one judge.json call per iter
    assert client.json.await_count == 2
    # First evolving prompt = seed; second = rewriter output
    assert history.prompt_versions() == [
        loop.SEED_PROMPT,
        "You are an ASCII artist. Now improved.",
    ]

    assert (tmp_path / "state.json").exists()
    iter01 = tmp_path / "iter-01" / "evolving"
    assert (iter01 / "prompt.txt").read_text() == loop.SEED_PROMPT
    assert (iter01 / "best.png").exists()
    assert (iter01 / "cand-00.png").exists()
    assert (iter01 / "cand-07.png").exists()

    # No control artifacts
    assert not (tmp_path / "iter-01" / "control").exists()


async def test_run_with_control_records_both_arms_and_freezes_control_prompt(
    tmp_path: Path,
) -> None:
    client = _stub_client(
        judge_results=[
            _judge_result(4, best=1),  # iter1 evolving
            _judge_result(4, best=0),  # iter1 control
            _judge_result(4, best=3),  # iter2 evolving
            _judge_result(4, best=2),  # iter2 control
        ],
    )
    config = loop.LoopConfig(
        subject="tree",
        iterations=2,
        candidates=4,
        no_control=False,
        out_dir=tmp_path,
    )

    history = await loop.run(config, client=client)

    arms = [r.arm for r in history.iterations]
    assert arms.count("evolving") == 2
    assert arms.count("control") == 2

    control_prompts = {r.system_prompt for r in history.iterations if r.arm == "control"}
    assert control_prompts == {loop.SEED_PROMPT}

    # Evolving prompts diverge from seed after iteration 1
    evolving_prompts = [r.system_prompt for r in history.iterations if r.arm == "evolving"]
    assert evolving_prompts[0] == loop.SEED_PROMPT
    assert evolving_prompts[1] == "You are an ASCII artist. Now improved."

    assert (tmp_path / "iter-01" / "control" / "best.png").exists()
    assert (tmp_path / "iter-01" / "evolving" / "best.png").exists()


async def test_run_persists_state_after_each_iteration(tmp_path: Path) -> None:
    client = _stub_client(
        judge_results=[_judge_result(2, best=0), _judge_result(2, best=1)],
    )
    config = loop.LoopConfig(
        subject="star", iterations=2, candidates=2, no_control=True, out_dir=tmp_path
    )

    await loop.run(config, client=client)

    state = RunHistory.load(tmp_path / "state.json")
    assert state.subject == "star"
    assert len(state.iterations) == 2
    assert state.seed_prompt == loop.SEED_PROMPT


async def test_run_pads_candidates_from_prior_best_when_actor_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Iteration 2 actor returns 1/3 candidates; loop pads slots 0,2 from prior best."""
    calls = {"n": 0}

    async def stub_generate(client, system_prompt, subject, *, k, thinking, temperature=1.0):
        calls["n"] += 1
        if calls["n"] == 1:
            # iter 1: all 3 slots filled
            return [
                Candidate(text=f"art-1-{i}", index=i, raw=f"raw-1-{i}") for i in range(3)
            ]
        # iter 2: only slot 1 survives
        return [Candidate(text="art-2-1", index=1, raw="raw-2-1")]

    monkeypatch.setattr(loop.actor, "generate", stub_generate)

    client = _stub_client(
        judge_results=[_judge_result(3, best=2), _judge_result(3, best=1)],
    )
    config = loop.LoopConfig(
        subject="fish", iterations=2, candidates=3, no_control=True, out_dir=tmp_path
    )

    history = await loop.run(config, client=client)

    iter2 = history.iterations[1]
    indices = [c.index for c in iter2.candidates]
    assert indices == [0, 1, 2]
    # Slot 0 and 2 padded from iter-1 best (best_index=2 → text "art-1-2")
    assert iter2.candidates[0].text == "art-1-2"
    assert iter2.candidates[1].text == "art-2-1"
    assert iter2.candidates[2].text == "art-1-2"


async def test_run_iteration_one_with_zero_candidates_raises_after_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Iter 1 with 0 cands retries twice then raises IterationOneWedgedError.

    Per ``specs/02-loop.md`` §"Iteration-1 wedge handling": no prior best to
    pad from, so calling the judge with an empty image list is forbidden —
    the only safe recovery is a fresh attempt, up to 2 retries.
    """
    attempts = {"n": 0}

    async def empty(*a, **k):
        attempts["n"] += 1
        return []

    sleeps: list[float] = []

    async def fast_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(loop.actor, "generate", empty)
    monkeypatch.setattr(loop, "_iter_one_retry_sleep", fast_sleep)
    client = _stub_client(judge_results=[])
    config = loop.LoopConfig(
        subject="cat", iterations=1, candidates=4, no_control=True, out_dir=tmp_path
    )

    with caplog.at_level("WARNING", logger="volly.loop"):
        with pytest.raises(loop.IterationOneWedgedError):
            await loop.run(config, client=client)

    # 1 initial attempt + 2 retries = 3 actor.generate calls
    assert attempts["n"] == 3
    # Slept twice (between attempts, not after the final failure)
    assert sleeps == [loop._ITER_ONE_RETRY_SLEEP_S, loop._ITER_ONE_RETRY_SLEEP_S]
    # Warning surfaced for each empty arm on each non-final attempt (1 arm × 2 = 2)
    msgs = [r.getMessage() for r in caplog.records if "iter 1 produced" in r.getMessage()]
    assert len(msgs) == 2
    assert all("retrying iter 1" in m for m in msgs)


async def test_run_iteration_one_retry_succeeds_on_second_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flaky iter-1 actor recovers on the first retry and the run continues."""
    attempts = {"n": 0}

    async def flaky(client, system_prompt, subject, *, k, thinking, temperature=1.0):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return []
        return [Candidate(text=f"art-{i}", index=i, raw=f"raw-{i}") for i in range(k)]

    async def fast_sleep(seconds: float) -> None:
        return

    monkeypatch.setattr(loop.actor, "generate", flaky)
    monkeypatch.setattr(loop, "_iter_one_retry_sleep", fast_sleep)

    client = _stub_client(judge_results=[_judge_result(3, best=0)])
    config = loop.LoopConfig(
        subject="cat", iterations=1, candidates=3, no_control=True, out_dir=tmp_path
    )

    history = await loop.run(config, client=client)
    assert attempts["n"] == 2
    assert len(history.iterations) == 1
    assert len(history.iterations[0].candidates) == 3


async def test_run_iteration_one_partial_shortfall_does_not_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Iter 1 with 1/k cands (some > 0) records the partial set without retrying.

    Per spec 02 §"Iteration-1 wedge handling", retry kicks in ONLY for zero
    candidates; partial shortfall is recorded as-is so the judge still ranks
    what we have.
    """
    attempts = {"n": 0}

    async def partial(client, system_prompt, subject, *, k, thinking, temperature=1.0):
        attempts["n"] += 1
        # Always return 1/k candidates so a retry would be visible as a 2nd call
        return [Candidate(text="art-1", index=1, raw="raw-1")]

    monkeypatch.setattr(loop.actor, "generate", partial)
    client = _stub_client(judge_results=[_judge_result(1, best=0)])
    config = loop.LoopConfig(
        subject="cat", iterations=1, candidates=4, no_control=True, out_dir=tmp_path
    )

    history = await loop.run(config, client=client)
    assert attempts["n"] == 1
    assert len(history.iterations) == 1
    assert len(history.iterations[0].candidates) == 1


async def test_run_iteration_one_wedged_only_one_arm_empty_still_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ANY arm (not both) hits 0 cands in iter 1, the whole iter retries.

    Iter 1 has no prior best for either arm — a single empty arm cannot be
    repaired in-place, so the orchestrator retries both arms together.
    """
    calls = {"n": 0}

    async def first_call_empty_then_full(
        client, system_prompt, subject, *, k, thinking, temperature=1.0
    ):
        calls["n"] += 1
        if calls["n"] == 1:
            # First dispatched arm (evolving) of attempt 1 returns empty.
            return []
        return [Candidate(text=f"art-{i}", index=i, raw=f"raw-{i}") for i in range(k)]

    async def fast_sleep(seconds: float) -> None:
        return

    monkeypatch.setattr(loop.actor, "generate", first_call_empty_then_full)
    monkeypatch.setattr(loop, "_iter_one_retry_sleep", fast_sleep)

    # Attempt 1: control arm calls judge (1). Attempt 2: both arms (2). Total 3.
    client = _stub_client(
        judge_results=[
            _judge_result(2, best=0),
            _judge_result(2, best=0),
            _judge_result(2, best=0),
        ],
    )
    config = loop.LoopConfig(
        subject="cat",
        iterations=1,
        candidates=2,
        no_control=False,
        out_dir=tmp_path,
    )

    history = await loop.run(config, client=client)
    # Final attempt succeeded — both arms recorded exactly once
    arms = sorted(r.arm for r in history.iterations)
    assert arms == ["control", "evolving"]
    # 2 attempts × 2 arms = 4 actor.generate calls
    assert calls["n"] == 4


def test_main_returns_three_with_banner_on_iter_one_wedge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``main`` translates ``IterationOneWedgedError`` into rc=3 + banner on stderr."""

    async def empty(*a, **k):
        return []

    async def fast_sleep(seconds: float) -> None:
        return

    monkeypatch.setattr(loop.actor, "generate", empty)
    monkeypatch.setattr(loop, "_iter_one_retry_sleep", fast_sleep)

    def fake_ctor(*args, **kwargs):
        return _stub_client(judge_results=[])

    monkeypatch.setattr("volly.loop.GeminiClient", fake_ctor)

    rc = loop.main(
        [
            "--subject", "cat",
            "--iterations", "1",
            "--candidates", "1",
            "--no-control",
            "--out", str(tmp_path),
        ]
    )

    assert rc == 3
    err = capsys.readouterr().err
    assert loop._ITER_ONE_WEDGED_BANNER in err


async def test_run_judge_history_caps_at_four_prior_iterations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[int] = []

    async def stub_rank(client, subject, system_prompt, images, *, history=None, thinking):
        captured.append(len(history or []))
        return _judge_result(len(images), best=0)

    monkeypatch.setattr(loop.judge, "rank", stub_rank)

    client = _stub_client(judge_results=[])
    config = loop.LoopConfig(
        subject="heart", iterations=6, candidates=2, no_control=True, out_dir=tmp_path
    )
    await loop.run(config, client=client)

    # Iteration N sees min(N-1, 4) prior entries
    assert captured == [0, 1, 2, 3, 4, 4]


async def test_run_partial_state_persists_on_judge_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boom_after_n_judge_calls = {"n": 0}

    async def flaky_rank(client, subject, system_prompt, images, *, history=None, thinking):
        boom_after_n_judge_calls["n"] += 1
        if boom_after_n_judge_calls["n"] == 2:
            raise RuntimeError("judge died")
        return _judge_result(len(images), best=0)

    monkeypatch.setattr(loop.judge, "rank", flaky_rank)
    client = _stub_client(judge_results=[])
    config = loop.LoopConfig(
        subject="cat", iterations=3, candidates=2, no_control=True, out_dir=tmp_path
    )

    with pytest.raises(RuntimeError, match="judge died"):
        await loop.run(config, client=client)

    state = RunHistory.load(tmp_path / "state.json")
    assert len(state.iterations) == 1
    assert state.iterations[0].iter_index == 1


def test_main_returns_nonzero_on_invalid_subject(capsys: pytest.CaptureFixture[str]) -> None:
    rc = loop.main(["--subject", "dragon", "--iterations", "1", "--candidates", "1"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "dragon" in err


def test_main_help_lists_required_subject_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        loop._parse_args(["--help"])
    out = capsys.readouterr().out
    assert "--subject" in out
    assert "--no-control" in out
    assert "--ablate-judge" in out
    assert "--demo" in out


def test_parse_args_ablate_judge_defaults_off() -> None:
    args = loop._parse_args(["--subject", "cat"])
    assert args.ablate_judge is False
    args = loop._parse_args(["--subject", "cat", "--ablate-judge"])
    assert args.ablate_judge is True


async def test_run_ablate_judge_doubles_judge_calls_and_logs_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """--ablate-judge runs vision + text judge on the same candidates per arm."""
    seen_calls: list[bool] = []

    async def stub_rank(
        client, subject, system_prompt, images, *, history=None, thinking,
        include_images=True, texts=None,
    ):
        seen_calls.append(include_images)
        if not include_images:
            # Text judge: lower scores so delta is positive.
            return JudgeResult(
                scores=[CandidateScore(index=i, score=0.2, why=f"t{i}") for i in range(len(images))],
                best_index=0,
                worst_index=len(images) - 1,
                critique="text-only",
                prompt_suggestions=[],
            )
        return _judge_result(len(images), best=0)

    monkeypatch.setattr(loop.judge, "rank", stub_rank)

    client = _stub_client(judge_results=[])
    config = loop.LoopConfig(
        subject="cat",
        iterations=2,
        candidates=2,
        no_control=False,
        out_dir=tmp_path,
        ablate_judge=True,
    )

    with caplog.at_level("INFO", logger="volly.loop"):
        await loop.run(config, client=client)

    # 2 arms × 2 iters × 2 modes (vision + text) = 8 rank calls
    assert len(seen_calls) == 8
    assert seen_calls.count(True) == 4
    assert seen_calls.count(False) == 4

    delta_lines = [r.getMessage() for r in caplog.records if "ablation iter" in r.getMessage()]
    assert len(delta_lines) == 4
    assert any("arm evolving" in line and "delta=" in line for line in delta_lines)
    assert any("arm control" in line for line in delta_lines)


async def test_run_ablation_text_judge_failure_does_not_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A crash in the text-only judge must be logged and swallowed."""

    async def stub_rank(
        client, subject, system_prompt, images, *, history=None, thinking,
        include_images=True, texts=None,
    ):
        if not include_images:
            raise RuntimeError("text judge boom")
        return _judge_result(len(images), best=0)

    monkeypatch.setattr(loop.judge, "rank", stub_rank)

    client = _stub_client(judge_results=[])
    config = loop.LoopConfig(
        subject="cat",
        iterations=1,
        candidates=2,
        no_control=True,
        out_dir=tmp_path,
        ablate_judge=True,
    )

    with caplog.at_level("ERROR", logger="volly.loop"):
        history = await loop.run(config, client=client)

    assert len(history.iterations) == 1
    assert any("text-judge failed" in r.getMessage() for r in caplog.records)


def test_demo_prompts_cover_every_curated_subject() -> None:
    assert set(loop.DEMO_PROMPTS.keys()) == set(loop.CURATED_SUBJECTS)


def test_demo_prompts_anchor_and_diverge_from_seed() -> None:
    """Each rehearsed prompt has the rewriter anchor, mentions its subject,
    differs from SEED_PROMPT, and fits under the rewriter's 4000-char cap."""
    import re

    for subject, prompt in loop.DEMO_PROMPTS.items():
        assert prompt.startswith("You are an ASCII artist."), subject
        assert prompt != loop.SEED_PROMPT, subject
        assert len(prompt) <= 4000, subject
        pattern = re.compile(rf"\b{re.escape(subject)}\b", re.IGNORECASE)
        assert pattern.search(prompt) is not None, subject


def test_parse_args_demo_defaults_off() -> None:
    args = loop._parse_args(["--subject", "cat"])
    assert args.demo is False
    args = loop._parse_args(["--subject", "cat", "--demo"])
    assert args.demo is True


async def test_run_demo_mode_pre_warms_evolving_keeps_control_on_seed(
    tmp_path: Path,
) -> None:
    """--demo seeds the evolving arm with DEMO_PROMPTS[subject]; control stays on seed."""
    client = _stub_client(
        judge_results=[
            _judge_result(2, best=0),  # iter1 evolving
            _judge_result(2, best=0),  # iter1 control
        ],
    )
    config = loop.LoopConfig(
        subject="coffee cup",
        iterations=1,
        candidates=2,
        no_control=False,
        out_dir=tmp_path,
        demo_mode=True,
    )

    history = await loop.run(config, client=client)

    evolving = [r for r in history.iterations if r.arm == "evolving"][0]
    control = [r for r in history.iterations if r.arm == "control"][0]
    assert evolving.system_prompt == loop.DEMO_PROMPTS["coffee cup"]
    assert evolving.system_prompt != loop.SEED_PROMPT
    assert control.system_prompt == loop.SEED_PROMPT

    # Artifacts on disk reflect the rehearsed prompt for evolving, seed for control.
    evolving_dir = tmp_path / "iter-01" / "evolving"
    control_dir = tmp_path / "iter-01" / "control"
    assert (evolving_dir / "prompt.txt").read_text() == loop.DEMO_PROMPTS["coffee cup"]
    assert (control_dir / "prompt.txt").read_text() == loop.SEED_PROMPT


def test_parse_args_rpm_defaults_none() -> None:
    """``--rpm`` unset → ``None`` so the client falls back to env/30."""
    args = loop._parse_args(["--subject", "cat"])
    assert args.rpm is None
    args = loop._parse_args(["--subject", "cat", "--rpm", "5"])
    assert args.rpm == 5


def test_main_help_lists_rpm_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        loop._parse_args(["--help"])
    assert "--rpm" in capsys.readouterr().out


def test_main_passes_rpm_to_gemini_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--rpm 5`` reaches the auto-constructed ``GeminiClient`` ctor."""
    captured: dict[str, object] = {}

    def fake_ctor(*args, **kwargs):
        captured.update(kwargs)
        client = _stub_client(judge_results=[_judge_result(1, best=0)])
        return client

    monkeypatch.setattr("volly.loop.GeminiClient", fake_ctor)

    rc = loop.main(
        [
            "--subject", "cat",
            "--iterations", "1",
            "--candidates", "1",
            "--no-control",
            "--rpm", "5",
            "--out", str(tmp_path),
        ]
    )

    assert rc == 0
    assert captured.get("rpm") == 5


def test_parse_args_max_retry_wait_defaults_none_at_parser_level() -> None:
    """``--max-retry-wait`` parses to None when unset so tier presets can fill it;
    the effective 90.0 default is enforced inside ``_resolve_loop_config``."""
    args = loop._parse_args(["--subject", "cat"])
    assert args.max_retry_wait is None
    args = loop._parse_args(["--subject", "cat", "--max-retry-wait", "3700"])
    assert args.max_retry_wait == 3700.0
    # Effective default — sentinel None resolves to 90.0 when --tier is omitted.
    assert loop._resolve_loop_config(args).max_retry_wait_s == 3700.0
    bare = loop._parse_args(["--subject", "cat"])
    assert loop._resolve_loop_config(bare).max_retry_wait_s == 90.0


def test_main_help_lists_max_retry_wait_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        loop._parse_args(["--help"])
    assert "--max-retry-wait" in capsys.readouterr().out


def test_main_passes_max_retry_wait_to_gemini_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--max-retry-wait 3700`` reaches the auto-constructed ``GeminiClient`` ctor."""
    captured: dict[str, object] = {}

    def fake_ctor(*args, **kwargs):
        captured.update(kwargs)
        return _stub_client(judge_results=[_judge_result(1, best=0)])

    monkeypatch.setattr("volly.loop.GeminiClient", fake_ctor)

    rc = loop.main(
        [
            "--subject", "cat",
            "--iterations", "1",
            "--candidates", "1",
            "--no-control",
            "--max-retry-wait", "3700",
            "--out", str(tmp_path),
        ]
    )

    assert rc == 0
    assert captured.get("max_retry_wait_s") == 3700.0


def test_main_max_retry_wait_default_passes_90_to_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the flag, the loop still threads the 90s default through."""
    captured: dict[str, object] = {}

    def fake_ctor(*args, **kwargs):
        captured.update(kwargs)
        return _stub_client(judge_results=[_judge_result(1, best=0)])

    monkeypatch.setattr("volly.loop.GeminiClient", fake_ctor)

    rc = loop.main(
        [
            "--subject", "cat",
            "--iterations", "1",
            "--candidates", "1",
            "--no-control",
            "--out", str(tmp_path),
        ]
    )

    assert rc == 0
    assert captured.get("max_retry_wait_s") == 90.0


async def test_run_demo_mode_off_uses_seed_for_both_arms(tmp_path: Path) -> None:
    """Sanity: with demo_mode=False (the default), both arms start on SEED_PROMPT."""
    client = _stub_client(
        judge_results=[_judge_result(2, best=0), _judge_result(2, best=0)],
    )
    config = loop.LoopConfig(
        subject="cat",
        iterations=1,
        candidates=2,
        no_control=False,
        out_dir=tmp_path,
    )
    assert config.demo_mode is False

    history = await loop.run(config, client=client)
    iter1_prompts = {r.system_prompt for r in history.iterations}
    assert iter1_prompts == {loop.SEED_PROMPT}


# ---------------------------------------------------------------------------
# --resume <run-dir>  (specs/02-loop.md §"Resumable runs")
# ---------------------------------------------------------------------------


async def _seed_resumable_run(
    tmp_path: Path,
    *,
    iterations: int = 2,
    candidates: int = 2,
    no_control: bool = True,
    subject: str = "cat",
) -> Path:
    """Run loop.run with a stub client to leave a real state.json on disk."""
    n_judge = iterations if no_control else iterations * 2
    client = _stub_client(
        judge_results=[_judge_result(candidates, best=0) for _ in range(n_judge)],
    )
    config = loop.LoopConfig(
        subject=subject,
        iterations=iterations,
        candidates=candidates,
        no_control=no_control,
        out_dir=tmp_path,
    )
    await loop.run(config, client=client)
    return tmp_path


async def test_resume_continues_from_last_completed_iter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume run starts at iter N+1 and uses iter N's evolving prompt."""
    run_dir = await _seed_resumable_run(tmp_path, iterations=3, candidates=2)
    pre = RunHistory.load(run_dir / "state.json")
    assert len(pre.iterations) == 3
    last_prompt = pre.iterations[-1].system_prompt
    seen_iters: list[int] = []
    seen_prompts: list[str] = []

    async def stub_generate(client, system_prompt, subject, *, k, thinking, temperature=1.0):
        seen_prompts.append(system_prompt)
        return [Candidate(text=f"art-{i}", index=i, raw=f"raw-{i}") for i in range(k)]

    monkeypatch.setattr(loop.actor, "generate", stub_generate)

    async def stub_rank(client, subject, system_prompt, images, *, history=None, thinking):
        seen_iters.append(len(images))
        return _judge_result(len(images), best=0)

    monkeypatch.setattr(loop.judge, "rank", stub_rank)

    client = _stub_client(judge_results=[])
    config = loop.LoopConfig(
        iterations=5, candidates=2, no_control=True, resume=run_dir,
    )
    history = await loop.run(config, client=client)

    iter_indices = sorted({r.iter_index for r in history.iterations})
    assert iter_indices == [1, 2, 3, 4, 5]
    # New runs only fired for iters 4 and 5 (one arm each = 2 generate calls)
    assert len(seen_prompts) == 2
    # The very first newly-dispatched call used iter 3's recorded prompt as input
    assert seen_prompts[0] == last_prompt


async def test_resume_reuses_existing_run_dir_no_timestamp_subdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume must write into the same dir — no new ``YYYYMMDD-subject`` child."""
    run_dir = await _seed_resumable_run(tmp_path, iterations=1, candidates=2)
    children_before = sorted(p.name for p in run_dir.iterdir())

    async def stub_generate(client, system_prompt, subject, *, k, thinking, temperature=1.0):
        return [Candidate(text=f"x-{i}", index=i, raw="r") for i in range(k)]

    async def stub_rank(client, subject, system_prompt, images, *, history=None, thinking):
        return _judge_result(len(images), best=0)

    monkeypatch.setattr(loop.actor, "generate", stub_generate)
    monkeypatch.setattr(loop.judge, "rank", stub_rank)

    client = _stub_client(judge_results=[])
    config = loop.LoopConfig(
        iterations=2, candidates=2, no_control=True, resume=run_dir,
    )
    history = await loop.run(config, client=client)

    assert history.run_dir == run_dir
    # iter-02 added next to iter-01 and state.json — no extra timestamped dirs
    children_after = sorted(p.name for p in run_dir.iterdir())
    new = set(children_after) - set(children_before)
    assert new == {"iter-02"}


async def test_resume_drops_partial_half_and_reruns_that_iter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If iter N has evolving but not control, resume re-runs iter N cleanly."""
    # Build a state by hand: iter 1 complete (both arms), iter 2 evolving only.
    from volly.state import IterationRecord

    def _rec(iter_index: int, arm: str, prompt: str) -> IterationRecord:
        return IterationRecord(
            iter_index=iter_index,
            arm=arm,  # type: ignore[arg-type]
            system_prompt=prompt,
            candidates=[Candidate(text="c", index=0, raw="r")],
            judge=_judge_result(1, best=0),
            best_image_path=tmp_path / f"iter-{iter_index:02d}" / arm / "best.png",
            win_rate=0.5,
        )

    from datetime import UTC, datetime
    pre = RunHistory(
        subject="cat",
        started_at=datetime.now(UTC),
        run_dir=tmp_path,
        seed_prompt=loop.SEED_PROMPT,
        iterations=[
            _rec(1, "evolving", loop.SEED_PROMPT),
            _rec(1, "control", loop.SEED_PROMPT),
            _rec(2, "evolving", "iter-2-prompt"),
        ],
    )
    pre.save()

    seen_iters: list[int] = []

    async def stub_generate(client, system_prompt, subject, *, k, thinking, temperature=1.0):
        return [Candidate(text=f"x-{i}", index=i, raw="r") for i in range(k)]

    async def stub_rank(client, subject, system_prompt, images, *, history=None, thinking):
        seen_iters.append(len(images))
        return _judge_result(len(images), best=0)

    monkeypatch.setattr(loop.actor, "generate", stub_generate)
    monkeypatch.setattr(loop.judge, "rank", stub_rank)

    client = _stub_client(judge_results=[])
    config = loop.LoopConfig(
        iterations=2, candidates=1, no_control=False, resume=tmp_path,
    )
    history = await loop.run(config, client=client)

    # The partial iter-2 evolving record is dropped; the resumed loop re-runs iter 2
    # as a fresh both-arms iteration. Then nothing more (iterations=2).
    iter_indices = sorted(r.iter_index for r in history.iterations)
    assert iter_indices == [1, 1, 2, 2]
    # iter 2 evolving prompt should come from iter 1's evolving prompt (SEED), not "iter-2-prompt"
    iter2_evolving = [r for r in history.iterations if r.iter_index == 2 and r.arm == "evolving"]
    assert iter2_evolving[0].system_prompt == loop.SEED_PROMPT


async def test_resume_empty_iterations_starts_at_iter_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A state.json with no iterations resumes as a fresh iter-1 in the same run_dir."""
    from datetime import UTC, datetime
    RunHistory(
        subject="cat",
        started_at=datetime.now(UTC),
        run_dir=tmp_path,
        seed_prompt=loop.SEED_PROMPT,
        iterations=[],
    ).save()

    async def stub_generate(client, system_prompt, subject, *, k, thinking, temperature=1.0):
        return [Candidate(text=f"x-{i}", index=i, raw="r") for i in range(k)]

    async def stub_rank(client, subject, system_prompt, images, *, history=None, thinking):
        return _judge_result(len(images), best=0)

    monkeypatch.setattr(loop.actor, "generate", stub_generate)
    monkeypatch.setattr(loop.judge, "rank", stub_rank)

    client = _stub_client(judge_results=[])
    config = loop.LoopConfig(
        iterations=1, candidates=2, no_control=True, resume=tmp_path,
    )
    history = await loop.run(config, client=client)

    assert len(history.iterations) == 1
    assert history.iterations[0].iter_index == 1
    assert history.iterations[0].system_prompt == loop.SEED_PROMPT


def test_resume_missing_state_json_raises(tmp_path: Path) -> None:
    config = loop.LoopConfig(
        iterations=1, candidates=1, no_control=True, resume=tmp_path,
    )
    with pytest.raises(ValueError, match="does not exist"):
        # main is sync; run is async — drive run synchronously via asyncio.run-equivalent.
        import asyncio
        asyncio.run(loop.run(config))


def test_resume_malformed_state_json_raises(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text("{not json")
    config = loop.LoopConfig(
        iterations=1, candidates=1, no_control=True, resume=tmp_path,
    )
    with pytest.raises(ValueError, match="failed to load"):
        import asyncio
        asyncio.run(loop.run(config))


def test_parse_args_resume_defaults_none() -> None:
    args = loop._parse_args(["--subject", "cat"])
    assert args.resume is None
    args = loop._parse_args(["--resume", "/tmp/foo"])
    assert args.resume == Path("/tmp/foo")


def test_parse_args_subject_not_required_when_resume() -> None:
    """``--subject`` becomes optional once ``--resume`` is in play."""
    args = loop._parse_args(["--resume", "/tmp/foo"])
    assert args.subject is None
    assert args.resume == Path("/tmp/foo")


def test_main_help_lists_resume_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        loop._parse_args(["--help"])
    assert "--resume" in capsys.readouterr().out


async def test_run_subject_required_when_no_resume() -> None:
    """Without ``--resume`` and without a subject, run() raises ValueError."""
    config = loop.LoopConfig(iterations=1, candidates=1, no_control=True)
    with pytest.raises(ValueError, match="subject is required"):
        await loop.run(config)


async def test_resume_subject_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit ``--subject`` that contradicts state.json's subject errors out."""
    await _seed_resumable_run(tmp_path, iterations=1, candidates=1, subject="cat")
    config = loop.LoopConfig(
        subject="tree",
        iterations=1, candidates=1, no_control=True, resume=tmp_path,
    )
    with pytest.raises(ValueError, match="subject mismatch"):
        await loop.run(config)


def test_main_resume_without_subject_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`volly --resume <dir>` (no --subject) loads subject from state.json."""
    # Seed a small run via the same stub path the other tests use.
    seed_client = _stub_client(judge_results=[_judge_result(1, best=0)])
    asyncio_run = __import__("asyncio").run
    asyncio_run(
        loop.run(
            loop.LoopConfig(
                subject="star", iterations=1, candidates=1,
                no_control=True, out_dir=tmp_path,
            ),
            client=seed_client,
        )
    )

    async def stub_generate(client, system_prompt, subject, *, k, thinking, temperature=1.0):
        return [Candidate(text="x", index=0, raw="r")]

    async def stub_rank(client, subject, system_prompt, images, *, history=None, thinking):
        return _judge_result(len(images), best=0)

    monkeypatch.setattr(loop.actor, "generate", stub_generate)
    monkeypatch.setattr(loop.judge, "rank", stub_rank)

    def fake_ctor(*args, **kwargs):
        return _stub_client(judge_results=[])

    monkeypatch.setattr("volly.loop.GeminiClient", fake_ctor)

    rc = loop.main(
        [
            "--resume", str(tmp_path),
            "--iterations", "2",
            "--candidates", "1",
            "--no-control",
        ]
    )

    assert rc == 0
    # The loaded subject from state.json carried through end-to-end.
    state = RunHistory.load(tmp_path / "state.json")
    assert state.subject == "star"
    assert len(state.iterations) == 2


def test_main_subject_mismatch_with_resume_returns_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--subject foo --resume <cat-run>`` returns rc=2 with a helpful stderr."""
    seed_client = _stub_client(judge_results=[_judge_result(1, best=0)])
    import asyncio as _aio
    _aio.run(
        loop.run(
            loop.LoopConfig(
                subject="cat", iterations=1, candidates=1,
                no_control=True, out_dir=tmp_path,
            ),
            client=seed_client,
        )
    )

    def fake_ctor(*args, **kwargs):
        return _stub_client(judge_results=[])

    monkeypatch.setattr("volly.loop.GeminiClient", fake_ctor)

    rc = loop.main(
        [
            "--subject", "tree",
            "--resume", str(tmp_path),
            "--iterations", "2",
            "--candidates", "1",
            "--no-control",
        ]
    )
    assert rc == 2


# ---------------------------------------------------------------------------
# --tier {free,paid} preset bundles  (specs/02-loop.md §"Tier presets")
# ---------------------------------------------------------------------------


def test_tier_presets_match_spec_table() -> None:
    """Spec 02 §"Tier presets" pins the exact preset values per tier."""
    assert loop._TIER_PRESETS["free"] == {
        "rpm": 4,
        "candidates": 3,
        "no_control": True,
        "max_retry_wait_s": 3700.0,
    }
    assert loop._TIER_PRESETS["paid"] == {
        "rpm": 900,
        "candidates": 8,
        "no_control": False,
        "max_retry_wait_s": 90.0,
    }


def test_parse_args_tier_defaults_none() -> None:
    """``--tier`` omitted → ``args.tier is None`` (no preset applied)."""
    args = loop._parse_args(["--subject", "cat"])
    assert args.tier is None
    args = loop._parse_args(["--subject", "cat", "--tier", "free"])
    assert args.tier == "free"
    args = loop._parse_args(["--subject", "cat", "--tier", "paid"])
    assert args.tier == "paid"


def test_parse_args_tier_rejects_unknown_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """argparse rejects ``--tier dragon`` via ``choices=``."""
    with pytest.raises(SystemExit):
        loop._parse_args(["--subject", "cat", "--tier", "dragon"])


def test_main_help_lists_tier_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        loop._parse_args(["--help"])
    out = capsys.readouterr().out
    assert "--tier" in out
    assert "free" in out
    assert "paid" in out


def test_resolve_loop_config_no_tier_preserves_existing_defaults() -> None:
    """Omitting ``--tier`` is the legacy path — argparse defaults stand."""
    args = loop._parse_args(["--subject", "cat"])
    cfg = loop._resolve_loop_config(args)
    assert cfg.candidates == 8
    assert cfg.no_control is False
    assert cfg.rpm is None
    assert cfg.max_retry_wait_s == 90.0


def test_resolve_loop_config_tier_free_applies_all_preset_values() -> None:
    args = loop._parse_args(["--subject", "cat", "--tier", "free"])
    cfg = loop._resolve_loop_config(args)
    assert cfg.candidates == 3
    assert cfg.no_control is True
    assert cfg.rpm == 4
    assert cfg.max_retry_wait_s == 3700.0


def test_resolve_loop_config_tier_paid_applies_all_preset_values() -> None:
    args = loop._parse_args(["--subject", "cat", "--tier", "paid"])
    cfg = loop._resolve_loop_config(args)
    assert cfg.candidates == 8
    assert cfg.no_control is False
    assert cfg.rpm == 900
    assert cfg.max_retry_wait_s == 90.0


def test_resolve_loop_config_explicit_flag_overrides_tier_preset() -> None:
    """``--tier free --candidates 5`` keeps candidates=5; spec demands this."""
    args = loop._parse_args(
        [
            "--subject", "cat",
            "--tier", "free",
            "--candidates", "5",
            "--rpm", "20",
            "--max-retry-wait", "200",
        ]
    )
    cfg = loop._resolve_loop_config(args)
    assert cfg.candidates == 5
    assert cfg.rpm == 20
    assert cfg.max_retry_wait_s == 200.0
    # Untouched preset values still apply.
    assert cfg.no_control is True


def test_resolve_loop_config_no_control_overrides_paid_preset() -> None:
    """``--tier paid --no-control`` flips no_control off the preset's False."""
    args = loop._parse_args(
        ["--subject", "cat", "--tier", "paid", "--no-control"]
    )
    cfg = loop._resolve_loop_config(args)
    assert cfg.no_control is True
    # Other preset values still apply.
    assert cfg.rpm == 900


def test_main_tier_free_threads_preset_to_gemini_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: ``--tier free`` reaches the auto-constructed GeminiClient."""
    captured: dict[str, object] = {}

    def fake_ctor(*args, **kwargs):
        captured.update(kwargs)
        return _stub_client(judge_results=[_judge_result(3, best=0)])

    monkeypatch.setattr("volly.loop.GeminiClient", fake_ctor)

    rc = loop.main(
        [
            "--subject", "cat",
            "--iterations", "1",
            "--tier", "free",
            "--out", str(tmp_path),
        ]
    )

    assert rc == 0
    assert captured.get("rpm") == 4
    assert captured.get("max_retry_wait_s") == 3700.0


def test_last_complete_iter_no_control_only_evolving() -> None:
    """With ``expects_control=False`` a lone evolving record satisfies the iter."""
    from datetime import UTC, datetime

    from volly.state import IterationRecord

    def _rec(idx: int, arm: str) -> IterationRecord:
        return IterationRecord(
            iter_index=idx, arm=arm,  # type: ignore[arg-type]
            system_prompt=loop.SEED_PROMPT,
            candidates=[Candidate(text="c", index=0, raw="r")],
            judge=_judge_result(1, best=0),
            best_image_path=Path("ignored"),
            win_rate=0.5,
        )

    hist = RunHistory(
        subject="cat",
        started_at=datetime.now(UTC),
        run_dir=Path("/tmp"),
        seed_prompt=loop.SEED_PROMPT,
        iterations=[_rec(1, "evolving"), _rec(2, "evolving"), _rec(3, "evolving")],
    )
    assert loop._last_complete_iter(hist, expects_control=False) == 3
    # With control expected, none of these qualify.
    assert loop._last_complete_iter(hist, expects_control=True) == 0


def test_last_complete_iter_both_arms_required() -> None:
    """With control expected, an iter missing the control record drops out."""
    from datetime import UTC, datetime

    from volly.state import IterationRecord

    def _rec(idx: int, arm: str) -> IterationRecord:
        return IterationRecord(
            iter_index=idx, arm=arm,  # type: ignore[arg-type]
            system_prompt=loop.SEED_PROMPT,
            candidates=[Candidate(text="c", index=0, raw="r")],
            judge=_judge_result(1, best=0),
            best_image_path=Path("ignored"),
            win_rate=0.5,
        )

    hist = RunHistory(
        subject="cat",
        started_at=datetime.now(UTC),
        run_dir=Path("/tmp"),
        seed_prompt=loop.SEED_PROMPT,
        iterations=[
            _rec(1, "evolving"), _rec(1, "control"),
            _rec(2, "evolving"), _rec(2, "control"),
            _rec(3, "evolving"),  # control missing — partial
        ],
    )
    assert loop._last_complete_iter(hist, expects_control=True) == 2
