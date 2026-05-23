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
        {"cat", "house", "fish", "coffee cup", "smiley", "sailboat", "tree", "heart", "star"}
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


async def test_run_iteration_one_with_zero_candidates_does_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def empty(*a, **k):
        return []

    monkeypatch.setattr(loop.actor, "generate", empty)
    client = _stub_client(judge_results=[])
    config = loop.LoopConfig(
        subject="cat", iterations=1, candidates=4, no_control=True, out_dir=tmp_path
    )

    history = await loop.run(config, client=client)
    assert len(history.iterations) == 1
    record = history.iterations[0]
    assert record.candidates == []
    assert record.judge.scores == []


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
