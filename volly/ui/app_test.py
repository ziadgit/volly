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
from volly.loop import CURATED_SUBJECTS
from volly.state import IterationRecord, RunHistory, win_rate
from volly.ui.app import (
    ARM_COLORS,
    _resolve_subject,
    _winrate_chart,
    latest_run_dir,
    load_history,
    run_root,
    sanitize_subject,
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
    assert winrate_chart_data(_history(tmp_path)) == []


def test_winrate_chart_data_evolving_only(tmp_path: Path) -> None:
    history = _history(tmp_path)
    history.add(_record(iter_index=1, arm="evolving", prompt="p1", scores=[0.4], run_dir=tmp_path))
    history.add(_record(iter_index=2, arm="evolving", prompt="p2", scores=[0.7], run_dir=tmp_path))

    data = winrate_chart_data(history)
    assert {r["arm"] for r in data} == {"evolving"}
    assert [(r["iteration"], r["win_rate"]) for r in data] == [
        (1, pytest.approx(0.4)),
        (2, pytest.approx(0.7)),
    ]


def test_winrate_chart_data_both_arms_equal_length(tmp_path: Path) -> None:
    history = _history(tmp_path)
    history.add(_record(iter_index=1, arm="evolving", prompt="p1", scores=[0.4], run_dir=tmp_path))
    history.add(_record(iter_index=1, arm="control", prompt="seed", scores=[0.2], run_dir=tmp_path))
    history.add(_record(iter_index=2, arm="evolving", prompt="p2", scores=[0.7], run_dir=tmp_path))
    history.add(_record(iter_index=2, arm="control", prompt="seed", scores=[0.25], run_dir=tmp_path))

    data = winrate_chart_data(history)
    evolving = [r for r in data if r["arm"] == "evolving"]
    control = [r for r in data if r["arm"] == "control"]
    assert [r["win_rate"] for r in evolving] == [pytest.approx(0.4), pytest.approx(0.7)]
    assert [r["win_rate"] for r in control] == [pytest.approx(0.2), pytest.approx(0.25)]
    assert [r["iteration"] for r in evolving] == [1, 2]
    assert [r["iteration"] for r in control] == [1, 2]


def test_winrate_chart_data_ragged_arms_no_padding(tmp_path: Path) -> None:
    """Altair handles ragged series natively, so the long-form output skips
    iterations where an arm has no data rather than padding with None."""
    history = _history(tmp_path)
    history.add(_record(iter_index=1, arm="evolving", prompt="p1", scores=[0.4], run_dir=tmp_path))
    history.add(_record(iter_index=1, arm="control", prompt="seed", scores=[0.2], run_dir=tmp_path))
    history.add(_record(iter_index=2, arm="evolving", prompt="p2", scores=[0.7], run_dir=tmp_path))
    # Control arm has no iter 2 — should appear in evolving series only.

    data = winrate_chart_data(history)
    evolving = [r for r in data if r["arm"] == "evolving"]
    control = [r for r in data if r["arm"] == "control"]
    assert [r["iteration"] for r in evolving] == [1, 2]
    assert [r["iteration"] for r in control] == [1]
    assert all(r["win_rate"] is not None for r in data)


def test_winrate_chart_data_omits_iterations_without_evolving_arm(tmp_path: Path) -> None:
    """If only the control arm has run (degenerate but possible in tests), the
    evolving series is simply absent — no zero-padding, no missing-key crash."""
    history = _history(tmp_path)
    history.add(_record(iter_index=1, arm="control", prompt="seed", scores=[0.2], run_dir=tmp_path))
    data = winrate_chart_data(history)
    assert {r["arm"] for r in data} == {"control"}


def test_winrate_chart_y_axis_bounded_to_unit_interval(tmp_path: Path) -> None:
    history = _history(tmp_path)
    history.add(_record(iter_index=1, arm="evolving", prompt="p1", scores=[0.4], run_dir=tmp_path))
    history.add(_record(iter_index=1, arm="control", prompt="seed", scores=[0.2], run_dir=tmp_path))

    chart = _winrate_chart(winrate_chart_data(history))
    spec = chart.to_dict()
    y_scale = spec["encoding"]["y"]["scale"]
    assert y_scale["domain"] == [0.0, 1.0]
    # ``clamp`` keeps a runaway 1.05 from blowing out the axis on the demo.
    assert y_scale["clamp"] is True


def test_winrate_chart_uses_explicit_arm_colors(tmp_path: Path) -> None:
    history = _history(tmp_path)
    history.add(_record(iter_index=1, arm="evolving", prompt="p1", scores=[0.4], run_dir=tmp_path))
    history.add(_record(iter_index=1, arm="control", prompt="seed", scores=[0.2], run_dir=tmp_path))

    chart = _winrate_chart(winrate_chart_data(history))
    color_scale = chart.to_dict()["encoding"]["color"]["scale"]
    assert color_scale["domain"] == ["evolving", "control"]
    assert color_scale["range"] == [ARM_COLORS["evolving"], ARM_COLORS["control"]]


def test_winrate_chart_color_scale_omits_missing_arm(tmp_path: Path) -> None:
    """On --no-control runs the legend should not show a phantom control entry."""
    history = _history(tmp_path)
    history.add(_record(iter_index=1, arm="evolving", prompt="p1", scores=[0.4], run_dir=tmp_path))
    history.add(_record(iter_index=2, arm="evolving", prompt="p2", scores=[0.7], run_dir=tmp_path))

    chart = _winrate_chart(winrate_chart_data(history))
    color_scale = chart.to_dict()["encoding"]["color"]["scale"]
    assert color_scale["domain"] == ["evolving"]
    assert color_scale["range"] == [ARM_COLORS["evolving"]]


# -- sanitize_subject / _resolve_subject ----------------------------------


@pytest.mark.parametrize("typed", sorted(CURATED_SUBJECTS))
def test_sanitize_subject_exact_match_passthrough(typed: str) -> None:
    assert sanitize_subject(typed) == typed


def test_sanitize_subject_case_insensitive() -> None:
    assert sanitize_subject("Cat") == "cat"
    assert sanitize_subject("TREE") == "tree"
    assert sanitize_subject("Coffee Cup") == "coffee cup"


def test_sanitize_subject_strips_whitespace() -> None:
    assert sanitize_subject("  cat  ") == "cat"
    assert sanitize_subject("\tsailboat\n") == "sailboat"


def test_sanitize_subject_empty_or_whitespace_returns_none() -> None:
    assert sanitize_subject("") is None
    assert sanitize_subject("   ") is None
    assert sanitize_subject("\n\t") is None


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("sailbot", "sailboat"),
        ("coffe cup", "coffee cup"),
        ("coffeecup", "coffee cup"),
        ("heeart", "heart"),
        ("smily", "smiley"),
        ("tre", "tree"),
        ("ct", "cat"),
    ],
)
def test_sanitize_subject_resolves_typo_to_curated(typed: str, expected: str) -> None:
    assert sanitize_subject(typed) == expected


@pytest.mark.parametrize("typed", ["airplane", "dragon", "qwerty", "xy", "robot"])
def test_sanitize_subject_unrelated_returns_none(typed: str) -> None:
    assert sanitize_subject(typed) is None


def test_sanitize_subject_accepts_custom_curated_set() -> None:
    """Helper is pure — the curated set is an arg, not a global lookup."""
    assert sanitize_subject("apricot", curated=["apricot", "fig"]) == "apricot"
    assert sanitize_subject("aprcot", curated=["apricot", "fig"]) == "apricot"
    assert sanitize_subject("banana", curated=["apricot", "fig"]) is None


def test_resolve_subject_empty_text_uses_dropdown_silently() -> None:
    subject, notice = _resolve_subject("", "house")
    assert subject == "house"
    assert notice is None


def test_resolve_subject_canonical_text_takes_precedence_no_notice() -> None:
    """Operator typed exactly 'cat' — dropdown shows 'house' but we honor the type."""
    subject, notice = _resolve_subject("cat", "house")
    assert subject == "cat"
    assert notice is None


def test_resolve_subject_fuzzy_match_surfaces_info_notice() -> None:
    subject, notice = _resolve_subject("sailbot", "cat")
    assert subject == "sailboat"
    assert notice is not None
    assert notice.startswith("info:")
    assert "sailbot" in notice and "sailboat" in notice


def test_resolve_subject_unmatched_falls_back_to_dropdown_with_warn() -> None:
    subject, notice = _resolve_subject("airplane", "house")
    assert subject == "house"
    assert notice is not None
    assert notice.startswith("warn:")
    assert "airplane" in notice and "house" in notice


def test_resolve_subject_whitespace_only_text_uses_dropdown() -> None:
    """Whitespace-only free-text should behave identically to an empty box —
    no warning, no caption, the operator just didn't type anything."""
    subject, notice = _resolve_subject("   ", "tree")
    assert subject == "tree"
    assert notice is None
