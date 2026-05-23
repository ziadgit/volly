"""End-to-end smoke test for ``python -m volly.loop``.

Drives :func:`volly.loop.main` in-process with a stubbed
:class:`GeminiClient` so the full CLI → orchestration → state.json →
printed best-image-path pipeline is exercised without any network I/O.

Mirrors the fix_plan's P0 smoke acceptance criteria literally: invoke with
``--subject cat --iterations 2`` and confirm a valid run-history JSON is
written and that ``main`` prints a best-of-N image path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from volly import loop
from volly.gemini_client import GeminiClient, Thinking
from volly.judge import CandidateScore, JudgeResult
from volly.state import RunHistory

_ACTOR_TEXT = "/\\_/\\\n( o.o )\n > ^ <"
_REWRITER_TEXT = "You are an ASCII artist. Use thinner whiskers and centered eyes."


def _judge_result(n: int, *, best: int) -> JudgeResult:
    return JudgeResult(
        scores=[
            CandidateScore(index=i, score=min(0.95, 0.4 + 0.05 * i), why=f"r{i}")
            for i in range(n)
        ],
        best_index=best,
        worst_index=max(0, n - 1),
        critique="ok",
        prompt_suggestions=["use thinner lines"],
    )


def _stub_client_factory(judge_results: list[JudgeResult]):
    """Return a zero-arg callable that yields a freshly-stubbed client.

    ``loop.main`` constructs ``GeminiClient()`` itself; the factory replaces
    that constructor.
    """

    def _build(*_args, **_kwargs) -> GeminiClient:
        client = MagicMock(spec=GeminiClient)

        async def text_router(system, user, *, thinking=Thinking.LOW, temperature=1.0):
            if thinking is Thinking.HIGH:
                return _REWRITER_TEXT
            return _ACTOR_TEXT

        client.text = AsyncMock(side_effect=text_router)
        client.json = AsyncMock(side_effect=list(judge_results))
        client.multimodal = AsyncMock()
        return client

    return _build


def test_cli_smoke_two_iterations_writes_state_and_prints_best_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 2 iters × 2 arms (evolving + control) = 4 judge calls.
    judge_results = [
        _judge_result(8, best=2),  # iter1 evolving
        _judge_result(8, best=4),  # iter1 control
        _judge_result(8, best=5),  # iter2 evolving
        _judge_result(8, best=1),  # iter2 control
    ]
    monkeypatch.setattr(loop, "GeminiClient", _stub_client_factory(judge_results))

    rc = loop.main(
        [
            "--subject",
            "cat",
            "--iterations",
            "2",
            "--candidates",
            "8",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0

    out = capsys.readouterr().out
    match = re.search(r"^best-of-8: (.+)$", out, re.MULTILINE)
    assert match is not None, f"missing best-of-8 line in stdout:\n{out}"
    best_path = Path(match.group(1))
    assert best_path.exists()
    assert best_path.suffix == ".png"
    assert best_path.name == "best.png"
    assert best_path.parent == tmp_path / "iter-02" / "evolving"

    assert f"run dir: {tmp_path}" in out

    state_path = tmp_path / "state.json"
    assert state_path.exists()
    json.loads(state_path.read_text())  # valid JSON

    history = RunHistory.load(state_path)
    assert history.subject == "cat"
    assert history.seed_prompt == loop.SEED_PROMPT
    assert len(history.iterations) == 4
    arms = [r.arm for r in history.iterations]
    assert arms.count("evolving") == 2
    assert arms.count("control") == 2

    # Evolving prompt updates after iter 1; control prompt stays at seed.
    evolving_prompts = [r.system_prompt for r in history.iterations if r.arm == "evolving"]
    assert evolving_prompts[0] == loop.SEED_PROMPT
    assert evolving_prompts[1] == _REWRITER_TEXT
    control_prompts = {r.system_prompt for r in history.iterations if r.arm == "control"}
    assert control_prompts == {loop.SEED_PROMPT}

    for iter_n in (1, 2):
        for arm in ("evolving", "control"):
            arm_dir = tmp_path / f"iter-{iter_n:02d}" / arm
            assert (arm_dir / "best.png").exists()
            assert (arm_dir / "prompt.txt").exists()
            for cand_n in range(8):
                assert (arm_dir / f"cand-{cand_n:02d}.png").exists()


def test_cli_smoke_no_control_halves_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    judge_results = [_judge_result(8, best=0), _judge_result(8, best=7)]
    monkeypatch.setattr(loop, "GeminiClient", _stub_client_factory(judge_results))

    rc = loop.main(
        [
            "--subject",
            "tree",
            "--iterations",
            "2",
            "--candidates",
            "8",
            "--no-control",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "best-of-8:" in out

    history = RunHistory.load(tmp_path / "state.json")
    assert all(r.arm == "evolving" for r in history.iterations)
    assert len(history.iterations) == 2

    for iter_n in (1, 2):
        assert (tmp_path / f"iter-{iter_n:02d}" / "evolving" / "best.png").exists()
        assert not (tmp_path / f"iter-{iter_n:02d}" / "control").exists()
