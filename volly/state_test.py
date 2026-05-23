"""Tests for ``volly.state``. No model calls — pure data."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from volly.actor import Candidate
from volly.judge import CandidateScore, JudgeResult
from volly.state import IterationRecord, RunHistory, win_rate


def _judge(scores: list[float], *, best: int = 0, worst: int = 0) -> JudgeResult:
    return JudgeResult(
        scores=[CandidateScore(index=i, score=s, why=f"r{i}") for i, s in enumerate(scores)],
        best_index=best,
        worst_index=worst,
        critique="ok",
        prompt_suggestions=["tighten proportions"],
    )


def _record(
    *,
    iter_index: int,
    arm: str,
    prompt: str,
    scores: list[float],
    run_dir: Path,
) -> IterationRecord:
    judge = _judge(scores, best=scores.index(max(scores)), worst=scores.index(min(scores)))
    return IterationRecord(
        iter_index=iter_index,
        arm=arm,  # type: ignore[arg-type]
        system_prompt=prompt,
        candidates=[
            Candidate(text=f"art-{i}", index=i, raw=f"raw-{i}") for i in range(len(scores))
        ],
        judge=judge,
        best_image_path=run_dir / f"iter-{iter_index:02d}" / arm / "best.png",
        win_rate=win_rate(scores),
    )


def test_win_rate_top_three_mean() -> None:
    assert win_rate([0.1, 0.9, 0.5, 0.7, 0.2]) == pytest.approx((0.9 + 0.7 + 0.5) / 3)


def test_win_rate_fewer_than_three_averages_all() -> None:
    assert win_rate([0.4, 0.8]) == pytest.approx(0.6)


def test_win_rate_empty_is_zero() -> None:
    assert win_rate([]) == 0.0


def test_win_rate_single_score() -> None:
    assert win_rate([0.42]) == pytest.approx(0.42)


def test_add_appends_in_order(tmp_path: Path) -> None:
    history = RunHistory(
        subject="cat",
        started_at=datetime(2026, 5, 23, 12, 0, tzinfo=UTC),
        run_dir=tmp_path,
        seed_prompt="You are an ASCII artist.",
    )
    r1 = _record(iter_index=1, arm="evolving", prompt="p1", scores=[0.5], run_dir=tmp_path)
    r2 = _record(iter_index=1, arm="control", prompt="p0", scores=[0.4], run_dir=tmp_path)
    history.add(r1)
    history.add(r2)
    assert history.iterations == [r1, r2]


def test_latest_returns_most_recent_record_for_arm(tmp_path: Path) -> None:
    history = RunHistory(
        subject="cat",
        started_at=datetime(2026, 5, 23, tzinfo=UTC),
        run_dir=tmp_path,
        seed_prompt="seed",
    )
    history.add(_record(iter_index=1, arm="evolving", prompt="p1", scores=[0.5], run_dir=tmp_path))
    history.add(_record(iter_index=1, arm="control", prompt="p0", scores=[0.3], run_dir=tmp_path))
    history.add(_record(iter_index=2, arm="evolving", prompt="p2", scores=[0.7], run_dir=tmp_path))

    latest_evolving = history.latest("evolving")
    latest_control = history.latest("control")
    assert latest_evolving is not None and latest_evolving.iter_index == 2
    assert latest_control is not None and latest_control.iter_index == 1


def test_latest_returns_none_when_arm_absent(tmp_path: Path) -> None:
    history = RunHistory(
        subject="cat",
        started_at=datetime(2026, 5, 23, tzinfo=UTC),
        run_dir=tmp_path,
        seed_prompt="seed",
    )
    history.add(_record(iter_index=1, arm="evolving", prompt="p1", scores=[0.5], run_dir=tmp_path))
    assert history.latest("control") is None


def test_win_rates_filters_by_arm(tmp_path: Path) -> None:
    history = RunHistory(
        subject="cat",
        started_at=datetime(2026, 5, 23, tzinfo=UTC),
        run_dir=tmp_path,
        seed_prompt="seed",
    )
    history.add(_record(iter_index=1, arm="evolving", prompt="p1", scores=[0.5, 0.6, 0.7], run_dir=tmp_path))
    history.add(_record(iter_index=1, arm="control", prompt="p0", scores=[0.3, 0.3, 0.3], run_dir=tmp_path))
    history.add(_record(iter_index=2, arm="evolving", prompt="p2", scores=[0.8, 0.8, 0.8], run_dir=tmp_path))

    assert history.win_rates("evolving") == [pytest.approx(0.6), pytest.approx(0.8)]
    assert history.win_rates("control") == [pytest.approx(0.3)]


def test_prompt_versions_collects_evolving_arm_only(tmp_path: Path) -> None:
    history = RunHistory(
        subject="cat",
        started_at=datetime(2026, 5, 23, tzinfo=UTC),
        run_dir=tmp_path,
        seed_prompt="seed",
    )
    history.add(_record(iter_index=1, arm="evolving", prompt="v1", scores=[0.5], run_dir=tmp_path))
    history.add(_record(iter_index=1, arm="control", prompt="seed", scores=[0.4], run_dir=tmp_path))
    history.add(_record(iter_index=2, arm="evolving", prompt="v2", scores=[0.6], run_dir=tmp_path))
    history.add(_record(iter_index=2, arm="control", prompt="seed", scores=[0.4], run_dir=tmp_path))
    assert history.prompt_versions() == ["v1", "v2"]


def test_diff_unified_format(tmp_path: Path) -> None:
    history = RunHistory(
        subject="cat",
        started_at=datetime(2026, 5, 23, tzinfo=UTC),
        run_dir=tmp_path,
        seed_prompt="seed",
    )
    history.add(_record(iter_index=1, arm="evolving", prompt="line a\nline b\n", scores=[0.5], run_dir=tmp_path))
    history.add(_record(iter_index=2, arm="evolving", prompt="line a\nline c\n", scores=[0.6], run_dir=tmp_path))

    diff = history.diff(1)
    assert "--- prompt v0" in diff
    assert "+++ prompt v1" in diff
    assert "-line b" in diff
    assert "+line c" in diff


def test_diff_identical_prompts_returns_empty(tmp_path: Path) -> None:
    history = RunHistory(
        subject="cat",
        started_at=datetime(2026, 5, 23, tzinfo=UTC),
        run_dir=tmp_path,
        seed_prompt="seed",
    )
    history.add(_record(iter_index=1, arm="evolving", prompt="same\n", scores=[0.5], run_dir=tmp_path))
    history.add(_record(iter_index=2, arm="evolving", prompt="same\n", scores=[0.6], run_dir=tmp_path))
    assert history.diff(1) == ""


@pytest.mark.parametrize("i", [-1, 0, 1, 5])
def test_diff_out_of_range_returns_empty(tmp_path: Path, i: int) -> None:
    history = RunHistory(
        subject="cat",
        started_at=datetime(2026, 5, 23, tzinfo=UTC),
        run_dir=tmp_path,
        seed_prompt="seed",
    )
    history.add(_record(iter_index=1, arm="evolving", prompt="only one", scores=[0.5], run_dir=tmp_path))
    assert history.diff(i) == ""


def test_save_writes_state_json_and_creates_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "2026-05-23T15-04-cat"
    history = RunHistory(
        subject="cat",
        started_at=datetime(2026, 5, 23, 15, 4, tzinfo=UTC),
        run_dir=run_dir,
        seed_prompt="You are an ASCII artist.",
    )
    history.add(_record(iter_index=1, arm="evolving", prompt="p1", scores=[0.5], run_dir=run_dir))

    target = history.save()

    assert target == run_dir / "state.json"
    assert target.is_file()
    assert not (run_dir / "state.json.tmp").exists()


def test_save_load_round_trip(tmp_path: Path) -> None:
    history = RunHistory(
        subject="cat",
        started_at=datetime(2026, 5, 23, 15, 4, tzinfo=UTC),
        run_dir=tmp_path,
        seed_prompt="You are an ASCII artist.",
    )
    for i in range(1, 4):
        history.add(
            _record(
                iter_index=i,
                arm="evolving",
                prompt=f"version {i}",
                scores=[0.1 * i, 0.15 * i, 0.2 * i, 0.25 * i],
                run_dir=tmp_path,
            )
        )
        history.add(
            _record(
                iter_index=i,
                arm="control",
                prompt="seed",
                scores=[0.4, 0.4, 0.4],
                run_dir=tmp_path,
            )
        )

    path = history.save()
    loaded = RunHistory.load(path)

    assert loaded.subject == history.subject
    assert loaded.started_at == history.started_at
    assert loaded.run_dir == history.run_dir
    assert loaded.seed_prompt == history.seed_prompt
    assert len(loaded.iterations) == len(history.iterations)
    for original, restored in zip(history.iterations, loaded.iterations, strict=True):
        assert restored.iter_index == original.iter_index
        assert restored.arm == original.arm
        assert restored.system_prompt == original.system_prompt
        assert restored.candidates == original.candidates
        assert restored.judge.model_dump() == original.judge.model_dump()
        assert restored.best_image_path == original.best_image_path
        assert restored.win_rate == pytest.approx(original.win_rate)


def test_save_overwrites_existing_file_atomically(tmp_path: Path) -> None:
    history = RunHistory(
        subject="cat",
        started_at=datetime(2026, 5, 23, tzinfo=UTC),
        run_dir=tmp_path,
        seed_prompt="seed",
    )
    history.add(_record(iter_index=1, arm="evolving", prompt="p1", scores=[0.5], run_dir=tmp_path))
    first = history.save()
    first_size = first.stat().st_size

    history.add(_record(iter_index=2, arm="evolving", prompt="p2", scores=[0.6], run_dir=tmp_path))
    second = history.save()

    assert second == first
    assert second.stat().st_size > first_size
    assert not (tmp_path / "state.json.tmp").exists()


def test_iteration_record_round_trip_preserves_pydantic_judge(tmp_path: Path) -> None:
    record = _record(
        iter_index=7,
        arm="control",
        prompt="prompt",
        scores=[0.1, 0.2, 0.3],
        run_dir=tmp_path,
    )
    restored = IterationRecord.from_dict(record.to_dict())
    assert restored == record
    assert isinstance(restored.judge, JudgeResult)
    assert isinstance(restored.candidates[0], Candidate)
    assert isinstance(restored.best_image_path, Path)
