"""Tests for ``volly.rewriter``. ``GeminiClient.text`` is mocked — no network."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

from google.genai import errors as genai_errors

from volly.gemini_client import GeminiClient, Thinking
from volly.judge import CandidateScore, JudgeResult
from volly.rewriter import _ANCHOR, _MAX_LEN, rewrite


def _client(text_mock: AsyncMock) -> GeminiClient:
    client = MagicMock(spec=GeminiClient)
    client.text = text_mock
    return client


def _judge_result(
    critique: str = "needs better whiskers.",
    suggestions: list[str] | None = None,
) -> JudgeResult:
    return JudgeResult(
        scores=[CandidateScore(index=0, score=0.5, why="meh")],
        best_index=0,
        worst_index=0,
        critique=critique,
        prompt_suggestions=suggestions if suggestions is not None else ["use ears"],
    )


async def test_rewrite_returns_text_and_calls_at_high_thinking() -> None:
    new_prompt = f"{_ANCHOR} Use clear silhouettes."
    mock = AsyncMock(return_value=new_prompt)
    client = _client(mock)

    out = await rewrite(client, f"{_ANCHOR} old.", _judge_result(), "cat")

    assert out == new_prompt
    assert mock.await_args.kwargs["thinking"] is Thinking.HIGH


async def test_rewrite_passes_thinking_override() -> None:
    mock = AsyncMock(return_value=f"{_ANCHOR} ok.")
    client = _client(mock)

    await rewrite(
        client, f"{_ANCHOR} old.", _judge_result(), "cat", thinking=Thinking.MEDIUM
    )

    assert mock.await_args.kwargs["thinking"] is Thinking.MEDIUM


async def test_rewrite_system_prompt_substitutes_subject() -> None:
    mock = AsyncMock(return_value=f"{_ANCHOR} ok.")
    client = _client(mock)

    await rewrite(client, f"{_ANCHOR} old.", _judge_result(), "sailboat")

    system = mock.await_args.args[0]
    assert 'draw "sailboat"' in system
    assert 'Mention "sailboat"' in system


async def test_rewrite_user_message_includes_prompt_critique_and_suggestions() -> None:
    mock = AsyncMock(return_value=f"{_ANCHOR} ok.")
    client = _client(mock)

    judge = _judge_result(
        critique="too cluttered.", suggestions=["use negative space", "fewer chars"]
    )
    await rewrite(client, f"{_ANCHOR} draw nicely.", judge, "fish")

    user = mock.await_args.args[1]
    assert "draw nicely." in user
    assert "too cluttered." in user
    assert "- use negative space" in user
    assert "- fewer chars" in user
    assert "fish" in user


async def test_rewrite_handles_empty_suggestions_gracefully() -> None:
    mock = AsyncMock(return_value=f"{_ANCHOR} ok.")
    client = _client(mock)

    judge = _judge_result(suggestions=[])
    await rewrite(client, f"{_ANCHOR} old.", judge, "cat")

    user = mock.await_args.args[1]
    assert "no specific suggestions returned" in user


async def test_rewrite_injects_anchor_when_model_omits_it() -> None:
    mock = AsyncMock(return_value="Use bold shapes and clear lines.")
    client = _client(mock)

    out = await rewrite(client, f"{_ANCHOR} old.", _judge_result(), "cat")

    assert out.startswith(_ANCHOR)
    assert "Use bold shapes" in out


async def test_rewrite_does_not_double_anchor_when_present() -> None:
    body = f"{_ANCHOR} Be careful with proportions."
    mock = AsyncMock(return_value=body)
    client = _client(mock)

    out = await rewrite(client, f"{_ANCHOR} old.", _judge_result(), "cat")

    assert out == body
    assert out.count(_ANCHOR) == 1


async def test_rewrite_truncates_to_max_len(caplog) -> None:
    long_body = _ANCHOR + " " + ("x" * (_MAX_LEN * 2))
    mock = AsyncMock(return_value=long_body)
    client = _client(mock)

    with caplog.at_level(logging.WARNING, logger="volly.rewriter"):
        out = await rewrite(client, f"{_ANCHOR} old.", _judge_result(), "cat")

    assert len(out) == _MAX_LEN
    assert out.startswith(_ANCHOR)
    assert any("truncating" in rec.getMessage() for rec in caplog.records)


async def test_rewrite_warns_on_subject_overfitting(caplog) -> None:
    overfit = f"{_ANCHOR} For a cat, the cat has cat ears and cat whiskers."
    mock = AsyncMock(return_value=overfit)
    client = _client(mock)

    with caplog.at_level(logging.WARNING, logger="volly.rewriter"):
        out = await rewrite(client, f"{_ANCHOR} old.", _judge_result(), "cat")

    assert out == overfit
    assert any("overfitting" in rec.getMessage() for rec in caplog.records)


async def test_rewrite_does_not_warn_when_subject_within_limit(caplog) -> None:
    body = f"{_ANCHOR} A cat needs clear ears. Avoid clutter."
    mock = AsyncMock(return_value=body)
    client = _client(mock)

    with caplog.at_level(logging.WARNING, logger="volly.rewriter"):
        out = await rewrite(client, f"{_ANCHOR} old.", _judge_result(), "cat")

    assert out == body
    assert not any("overfitting" in rec.getMessage() for rec in caplog.records)


async def test_rewrite_subject_count_is_case_insensitive_and_word_bounded(caplog) -> None:
    # "Cat", "cat", "CAT" all count; "catalog" does not. → 3 mentions, warns.
    body = f"{_ANCHOR} A Cat is a cat is a CAT, not a catalog."
    mock = AsyncMock(return_value=body)
    client = _client(mock)

    with caplog.at_level(logging.WARNING, logger="volly.rewriter"):
        await rewrite(client, f"{_ANCHOR} old.", _judge_result(), "cat")

    assert any("overfitting" in rec.getMessage() for rec in caplog.records)


async def test_rewrite_handles_multi_word_subject(caplog) -> None:
    body = f"{_ANCHOR} A coffee cup needs a handle. Coffee cup proportions matter."
    mock = AsyncMock(return_value=body)
    client = _client(mock)

    with caplog.at_level(logging.WARNING, logger="volly.rewriter"):
        out = await rewrite(client, f"{_ANCHOR} old.", _judge_result(), "coffee cup")

    assert out == body
    assert not any("overfitting" in rec.getMessage() for rec in caplog.records)


async def test_rewrite_strips_whitespace_from_model_output() -> None:
    body = f"   \n{_ANCHOR} ok.\n   "
    mock = AsyncMock(return_value=body)
    client = _client(mock)

    out = await rewrite(client, f"{_ANCHOR} old.", _judge_result(), "cat")

    assert out == f"{_ANCHOR} ok."


async def test_rewrite_anchor_injection_on_empty_response() -> None:
    mock = AsyncMock(return_value="")
    client = _client(mock)

    out = await rewrite(client, f"{_ANCHOR} old.", _judge_result(), "cat")

    assert out == _ANCHOR


async def test_rewrite_returns_current_prompt_on_api_error(caplog) -> None:
    prior = f"{_ANCHOR} keep me as-is."
    api_exc = genai_errors.APIError(
        code=429, response_json={"error": {"message": "quota exhausted"}}
    )
    mock = AsyncMock(side_effect=api_exc)
    client = _client(mock)

    with caplog.at_level(logging.INFO, logger="volly.rewriter"):
        out = await rewrite(client, prior, _judge_result(), "cat")

    assert out == prior
    assert any(
        "rewriter degraded" in rec.getMessage()
        and "keeping prior prompt" in rec.getMessage()
        for rec in caplog.records
    )


async def test_rewrite_api_error_skips_invariant_enforcement(caplog) -> None:
    # current_prompt that violates the anchor/length invariants must still
    # come back unchanged on APIError — the rewriter is *keeping* it, not
    # producing a new prompt.
    prior_without_anchor = "no anchor and full of nonsense " * 10
    api_exc = genai_errors.APIError(
        code=503, response_json={"error": {"message": "backend busy"}}
    )
    mock = AsyncMock(side_effect=api_exc)
    client = _client(mock)

    with caplog.at_level(logging.INFO, logger="volly.rewriter"):
        out = await rewrite(client, prior_without_anchor, _judge_result(), "cat")

    assert out == prior_without_anchor
    assert not out.startswith(_ANCHOR)


async def test_rewrite_api_error_log_level_is_info_not_warning(caplog) -> None:
    api_exc = genai_errors.APIError(
        code=429, response_json={"error": {"message": "slow"}}
    )
    mock = AsyncMock(side_effect=api_exc)
    client = _client(mock)

    with caplog.at_level(logging.INFO, logger="volly.rewriter"):
        await rewrite(client, f"{_ANCHOR} old.", _judge_result(), "cat")

    degraded = [
        rec for rec in caplog.records if "rewriter degraded" in rec.getMessage()
    ]
    assert degraded, "expected an INFO rewriter-degraded log line"
    assert all(rec.levelno == logging.INFO for rec in degraded)
