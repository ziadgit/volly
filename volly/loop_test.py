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
