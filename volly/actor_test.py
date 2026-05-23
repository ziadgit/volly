"""Tests for ``volly.actor``. ``GeminiClient.text`` is mocked — no network."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from volly.actor import Candidate, _strip_fences, generate
from volly.gemini_client import GeminiClient, Thinking


def _client(text_mock: AsyncMock) -> GeminiClient:
    client = MagicMock(spec=GeminiClient)
    client.text = text_mock
    return client


async def test_generate_returns_k_candidates_in_order() -> None:
    text = AsyncMock(side_effect=[f"art-{i}" for i in range(4)])
    client = _client(text)

    out = await generate(client, "sys", "cat", k=4)

    assert [c.text for c in out] == ["art-0", "art-1", "art-2", "art-3"]
    assert [c.index for c in out] == [0, 1, 2, 3]
    assert text.await_count == 4


async def test_generate_passes_thinking_and_temperature() -> None:
    text = AsyncMock(return_value="x")
    client = _client(text)

    await generate(client, "sys", "cat", k=1, thinking=Thinking.MEDIUM, temperature=0.3)

    call = text.await_args
    assert call.args[0] == "sys"
    assert "Subject: cat" in call.args[1]
    assert "no code fences" in call.args[1]
    assert call.kwargs["thinking"] is Thinking.MEDIUM
    assert call.kwargs["temperature"] == 0.3


async def test_generate_dispatches_in_parallel() -> None:
    started = 0
    can_finish = asyncio.Event()

    async def slow(*_args: object, **_kwargs: object) -> str:
        nonlocal started
        started += 1
        await can_finish.wait()
        return "ok"

    client = _client(AsyncMock(side_effect=slow))

    async def runner() -> list[Candidate]:
        return await generate(client, "sys", "cat", k=5)

    task = asyncio.create_task(runner())
    # Let the gather schedule every coroutine before any completes.
    for _ in range(20):
        if started >= 5:
            break
        await asyncio.sleep(0)
    assert started == 5
    can_finish.set()
    out = await task
    assert len(out) == 5


async def test_generate_drops_failed_calls() -> None:
    text = AsyncMock(side_effect=["ok-0", RuntimeError("boom"), "ok-2"])
    client = _client(text)

    out = await generate(client, "sys", "cat", k=3)

    assert [c.text for c in out] == ["ok-0", "ok-2"]
    assert [c.index for c in out] == [0, 2]


async def test_generate_returns_empty_when_all_fail() -> None:
    text = AsyncMock(side_effect=[RuntimeError("a"), RuntimeError("b")])
    client = _client(text)

    out = await generate(client, "sys", "cat", k=2)

    assert out == []


async def test_generate_zero_k_short_circuits() -> None:
    text = AsyncMock()
    client = _client(text)

    out = await generate(client, "sys", "cat", k=0)

    assert out == []
    text.assert_not_awaited()


async def test_generate_strips_code_fence_with_language_tag() -> None:
    body = "  /\\_/\\\n ( o.o )\n  > ^ <"
    text = AsyncMock(return_value=f"```ascii\n{body}\n```")
    client = _client(text)

    [out] = await generate(client, "sys", "cat", k=1)

    assert out.text == body
    assert out.raw == f"```ascii\n{body}\n```"


async def test_generate_strips_bare_triple_backtick_fence() -> None:
    body = "===\n |\n==="
    text = AsyncMock(return_value=f"```\n{body}\n```")
    client = _client(text)

    [out] = await generate(client, "sys", "tree", k=1)

    assert out.text == body


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("hello\nworld", "hello\nworld"),
        ("```py\nx = 1\n```", "x = 1"),
        ("```\nonly closing missing", "only closing missing"),
        ("text with ``` inline backticks", "text with ``` inline backticks"),
        ("", ""),
    ],
)
def test_strip_fences_edge_cases(raw: str, expected: str) -> None:
    assert _strip_fences(raw) == expected
