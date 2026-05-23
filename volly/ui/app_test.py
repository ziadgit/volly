"""Tests for the pure helpers in :mod:`volly.ui.app`.

Panel renderers call ``st.*`` and are covered by manual smoke testing per
``specs/10-ui.md``. Everything testable without a Streamlit ScriptRunner
lives in the helpers exercised here.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from volly.actor import Candidate
from volly.judge import CandidateScore, JudgeResult
from volly.state import IterationRecord, RunHistory, win_rate
from volly.ui.app import (
    latest_run_dir,
    load_history,
    run_root,
    winrate_chart_data,
)


def _judge(scores: list[float]) -> JudgeResult:
    return JudgeResult(
        scores=[CandidateScore(index=i, score=s, why=f"r{i}") for i, s in enumerate(scores)],
        best_index=scores.index(max(scores)) if scores else 0,
        worst_index=scores.index(min(scores)) if scores else 0,
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
    return IterationRecord(
        iter_index=iter_index,
        arm=arm,  # type: ignore[arg-type]
        system_prompt=prompt,
        candidates=[
            Candidate(text=f"art-{i}", index=i, raw=f"raw-{i}") for i in range(len(scores))
        ],
        judge=_judge(scores),
        best_image_path=run_dir / f"iter-{iter_index:02d}" / arm / "best.png",
        win_rate=win_rate(scores),
    )


def _history(tmp_path: Path) -> RunHistory:
    return RunHistory(
        subject="cat",
        started_at=datetime(2026, 5, 23, 12, 0, tzinfo=UTC),
        run_dir=tmp_path,
        seed_prompt="You are an ASCII artist.",
    )


def test_run_root_defaults_to_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOLLY_RUN_DIR", raising=False)
    assert run_root() == Path("runs")


def test_run_root_honors_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOLLY_RUN_DIR", str(tmp_path / "elsewhere"))
    assert run_root() == tmp_path / "elsewhere"


def test_latest_run_dir_missing_root_returns_none(tmp_path: Path) -> None:
    assert latest_run_dir(tmp_path / "does-not-exist") is None


def test_latest_run_dir_empty_root_returns_none(tmp_path: Path) -> None:
    assert latest_run_dir(tmp_path) is None


def test_latest_run_dir_ignores_dirs_without_state_json(tmp_path: Path) -> None:
    (tmp_path / "incomplete").mkdir()
    assert latest_run_dir(tmp_path) is None


def test_latest_run_dir_picks_newest_state_json(tmp_path: Path) -> None:
    older = tmp_path / "20260101-cat"
    newer = tmp_path / "20260201-tree"
    older.mkdir()
    newer.mkdir()
    (older / "state.json").write_text("{}")
    (newer / "state.json").write_text("{}")
    # Force a measurable mtime gap so the test isn't filesystem-resolution-dependent.
    os.utime(older / "state.json", (1_700_000_000, 1_700_000_000))
    os.utime(newer / "state.json", (1_800_000_000, 1_800_000_000))

    assert latest_run_dir(tmp_path) == newer


def test_latest_run_dir_skips_files_in_root(tmp_path: Path) -> None:
    (tmp_path / "stray.txt").write_text("ignored")
    run = tmp_path / "20260101-cat"
    run.mkdir()
    (run / "state.json").write_text("{}")
    assert latest_run_dir(tmp_path) == run


def test_load_history_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_history(tmp_path) is None


def test_load_history_returns_none_on_corrupt_json(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text("not json {{{")
    assert load_history(tmp_path) is None


def test_load_history_returns_none_on_schema_mismatch(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text('{"unexpected": "shape"}')
    assert load_history(tmp_path) is None


def test_load_history_roundtrips_saved_run(tmp_path: Path) -> None:
    history = _history(tmp_path)
    history.add(
        _record(iter_index=1, arm="evolving", prompt="seed", scores=[0.5], run_dir=tmp_path)
    )
    history.save()

    loaded = load_history(tmp_path)
    assert loaded is not None
    assert loaded.subject == "cat"
    assert len(loaded.iterations) == 1
    assert loaded.iterations[0].system_prompt == "seed"


def test_winrate_chart_data_empty_history(tmp_path: Path) -> None:
    assert winrate_chart_data(_history(tmp_path)) == {}


def test_winrate_chart_data_evolving_only(tmp_path: Path) -> None:
    history = _history(tmp_path)
    history.add(_record(iter_index=1, arm="evolving", prompt="p1", scores=[0.4], run_dir=tmp_path))
    history.add(_record(iter_index=2, arm="evolving", prompt="p2", scores=[0.7], run_dir=tmp_path))

    data = winrate_chart_data(history)
    assert set(data) == {"evolving"}
    assert data["evolving"] == [pytest.approx(0.4), pytest.approx(0.7)]


def test_winrate_chart_data_both_arms_equal_length(tmp_path: Path) -> None:
    history = _history(tmp_path)
    history.add(_record(iter_index=1, arm="evolving", prompt="p1", scores=[0.4], run_dir=tmp_path))
    history.add(_record(iter_index=1, arm="control", prompt="seed", scores=[0.2], run_dir=tmp_path))
    history.add(_record(iter_index=2, arm="evolving", prompt="p2", scores=[0.7], run_dir=tmp_path))
    history.add(_record(iter_index=2, arm="control", prompt="seed", scores=[0.25], run_dir=tmp_path))

    data = winrate_chart_data(history)
    assert set(data) == {"evolving", "control"}
    assert len(data["evolving"]) == len(data["control"]) == 2


def test_winrate_chart_data_pads_shorter_arm_with_none(tmp_path: Path) -> None:
    history = _history(tmp_path)
    history.add(_record(iter_index=1, arm="evolving", prompt="p1", scores=[0.4], run_dir=tmp_path))
    history.add(_record(iter_index=1, arm="control", prompt="seed", scores=[0.2], run_dir=tmp_path))
    history.add(_record(iter_index=2, arm="evolving", prompt="p2", scores=[0.7], run_dir=tmp_path))
    # Control arm stops at iter 1, evolving continues to iter 2.

    data = winrate_chart_data(history)
    assert data["evolving"] == [pytest.approx(0.4), pytest.approx(0.7)]
    assert data["control"] == [pytest.approx(0.2), None]
