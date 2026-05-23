"""Tests for ``volly.judge``. ``GeminiClient.json`` is mocked — no network."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import PIL.Image
import pytest
from google.genai import errors as genai_errors
from pydantic import ValidationError

from volly.gemini_client import GeminiClient, Thinking
from volly.judge import (
    _HISTORY_LIMIT,
    CandidateScore,
    HistoryEntry,
    JudgeResult,
    rank,
)


def _client(json_mock: AsyncMock) -> GeminiClient:
    client = MagicMock(spec=GeminiClient)
    client.json = json_mock
    return client


def _img(color: str = "white") -> PIL.Image.Image:
    return PIL.Image.new("RGB", (4, 4), color)


def _ok_result(n: int) -> JudgeResult:
    return JudgeResult(
        scores=[CandidateScore(index=i, score=0.5, why="meh") for i in range(n)],
        best_index=0,
        worst_index=max(n - 1, 0),
        critique="ok.",
        prompt_suggestions=["be more specific"],
    )


def _make_validation_error() -> ValidationError:
    try:
        JudgeResult.model_validate({})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected JudgeResult to reject empty input")


async def test_rank_returns_judge_result_and_calls_json() -> None:
    images = [_img() for _ in range(8)]
    mock = AsyncMock(return_value=_ok_result(8))
    client = _client(mock)

    out = await rank(client, "cat", "be an artist", images)

    assert isinstance(out, JudgeResult)
    assert out.scores[0].index == 0
    call = mock.await_args
    assert call.args[2] is JudgeResult
    assert call.kwargs["images"] == images
    assert call.kwargs["thinking"] is Thinking.HIGH


async def test_rank_system_prompt_carries_subject_count_and_artist_prompt() -> None:
    images = [_img() for _ in range(3)]
    mock = AsyncMock(return_value=_ok_result(3))
    client = _client(mock)

    await rank(client, "fish", "you draw stuff", images)

    system = mock.await_args.args[0]
    assert 'evaluating ASCII art renderings of "fish"' in system
    assert "You will see 3 candidate images" in system
    assert "you draw stuff" in system


async def test_rank_truncates_history_to_last_four() -> None:
    images = [_img() for _ in range(2)]
    history = [
        HistoryEntry(
            iter_index=i, best_image=_img("black"), critique=f"c{i}", top_score=0.1 * i
        )
        for i in range(6)
    ]
    mock = AsyncMock(return_value=_ok_result(2))
    client = _client(mock)

    await rank(client, "cat", "p", images, history=history)

    payload = mock.await_args.kwargs["images"]
    assert len(payload) == 2 + _HISTORY_LIMIT
    assert payload[:2] == images
    assert payload[2:] == [history[i].best_image for i in (2, 3, 4, 5)]

    user = mock.await_args.args[1]
    for i in (2, 3, 4, 5):
        assert f"Iteration {i}" in user
    assert "Iteration 0" not in user
    assert "Iteration 1" not in user


async def test_rank_with_no_history_sends_only_candidates() -> None:
    images = [_img() for _ in range(5)]
    mock = AsyncMock(return_value=_ok_result(5))
    client = _client(mock)

    await rank(client, "cat", "p", images, history=None)

    payload = mock.await_args.kwargs["images"]
    assert payload == images
    user = mock.await_args.args[1]
    assert "Prior iterations" not in user


async def test_rank_with_empty_history_list_sends_only_candidates() -> None:
    images = [_img() for _ in range(2)]
    mock = AsyncMock(return_value=_ok_result(2))
    client = _client(mock)

    await rank(client, "cat", "p", images, history=[])

    payload = mock.await_args.kwargs["images"]
    assert payload == images


async def test_rank_passes_thinking_override() -> None:
    images = [_img()]
    mock = AsyncMock(return_value=_ok_result(1))
    client = _client(mock)

    await rank(client, "cat", "p", images, thinking=Thinking.MEDIUM)

    assert mock.await_args.kwargs["thinking"] is Thinking.MEDIUM


async def test_rank_falls_back_on_validation_error(caplog: pytest.LogCaptureFixture) -> None:
    images = [_img() for _ in range(3)]
    mock = AsyncMock(side_effect=_make_validation_error())
    client = _client(mock)

    with caplog.at_level("WARNING", logger="volly.judge"):
        out = await rank(client, "cat", "p", images)

    assert [s.index for s in out.scores] == [0, 1, 2]
    assert all(s.score == 0.5 for s in out.scores)
    assert out.prompt_suggestions == []
    assert out.best_index == 0
    assert out.worst_index == 2
    assert any("fell back" in rec.getMessage() for rec in caplog.records)


def test_candidate_score_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        CandidateScore(index=0, score=1.5, why="too hot")
    with pytest.raises(ValidationError):
        CandidateScore(index=0, score=-0.1, why="too cold")


async def test_rank_text_only_mode_attaches_no_images_and_inlines_ascii() -> None:
    images = [_img() for _ in range(3)]
    texts = ["AAA\nBBB", "CCC\nDDD", "EEE\nFFF"]
    mock = AsyncMock(return_value=_ok_result(3))
    client = _client(mock)

    await rank(client, "cat", "p", images, include_images=False, texts=texts)

    assert mock.await_args.kwargs["images"] is None
    user = mock.await_args.args[1]
    for i, t in enumerate(texts):
        assert f"Candidate {i}:" in user
        assert t in user
    system = mock.await_args.args[0]
    assert "no images attached" in system
    assert "3 candidate drawings" in system


async def test_rank_text_only_mode_requires_texts() -> None:
    images = [_img()]
    mock = AsyncMock(return_value=_ok_result(1))
    client = _client(mock)

    with pytest.raises(ValueError, match="texts"):
        await rank(client, "cat", "p", images, include_images=False)
    mock.assert_not_awaited()


async def test_rank_text_only_mode_keeps_history_critiques_but_drops_image_payload() -> None:
    images = [_img() for _ in range(2)]
    texts = ["a", "b"]
    history = [
        HistoryEntry(
            iter_index=i, best_image=_img("black"), critique=f"crit-{i}", top_score=0.3 + 0.1 * i
        )
        for i in (3, 4, 5, 6)
    ]
    mock = AsyncMock(return_value=_ok_result(2))
    client = _client(mock)

    await rank(client, "cat", "p", images, history=history, include_images=False, texts=texts)

    assert mock.await_args.kwargs["images"] is None
    user = mock.await_args.args[1]
    for h in history:
        assert f"Iteration {h.iter_index}" in user
        assert h.critique in user
    assert "image" not in user.split("Candidate 0", 1)[0].lower() or "no images" in user.lower()


async def test_rank_text_only_mode_falls_back_on_validation_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    images = [_img() for _ in range(2)]
    texts = ["a", "b"]
    mock = AsyncMock(side_effect=_make_validation_error())
    client = _client(mock)

    with caplog.at_level("WARNING", logger="volly.judge"):
        out = await rank(client, "cat", "p", images, include_images=False, texts=texts)

    assert [s.index for s in out.scores] == [0, 1]
    assert all(s.score == 0.5 for s in out.scores)


async def test_rank_falls_back_on_api_error_with_degraded_critique(
    caplog: pytest.LogCaptureFixture,
) -> None:
    images = [_img() for _ in range(4)]
    api_exc = genai_errors.APIError(
        code=429, response_json={"error": {"message": "quota exhausted"}}
    )
    mock = AsyncMock(side_effect=api_exc)
    client = _client(mock)

    with caplog.at_level("WARNING", logger="volly.judge"):
        out = await rank(client, "cat", "p", images)

    assert [s.index for s in out.scores] == [0, 1, 2, 3]
    assert all(s.score == 0.5 for s in out.scores)
    assert out.prompt_suggestions == []
    assert out.best_index == 0
    assert out.worst_index == 3
    assert out.critique.startswith("judge degraded:")
    assert any(
        "judge degraded on APIError" in rec.getMessage() for rec in caplog.records
    )


async def test_rank_falls_back_on_api_error_text_mode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    images = [_img() for _ in range(2)]
    texts = ["a", "b"]
    api_exc = genai_errors.APIError(
        code=503, response_json={"error": {"message": "backend busy"}}
    )
    mock = AsyncMock(side_effect=api_exc)
    client = _client(mock)

    with caplog.at_level("WARNING", logger="volly.judge"):
        out = await rank(client, "cat", "p", images, include_images=False, texts=texts)

    assert [s.index for s in out.scores] == [0, 1]
    assert all(s.score == 0.5 for s in out.scores)
    assert out.prompt_suggestions == []
    assert out.critique.startswith("judge degraded:")
    assert any(
        "judge degraded on APIError" in rec.getMessage() for rec in caplog.records
    )


async def test_rank_api_error_fallback_handles_zero_candidates() -> None:
    images: list[PIL.Image.Image] = []
    api_exc = genai_errors.APIError(
        code=429, response_json={"error": {"message": "slow"}}
    )
    mock = AsyncMock(side_effect=api_exc)
    client = _client(mock)

    out = await rank(client, "cat", "p", images)

    assert out.scores == []
    assert out.best_index == 0
    assert out.worst_index == 0
    assert out.critique.startswith("judge degraded:")
