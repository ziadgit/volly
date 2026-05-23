"""Tests for ``volly.gemini_client``. No network — SDK call is mocked."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import PIL.Image
import pytest
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

from volly.gemini_client import GeminiClient, Thinking


class _Schema(BaseModel):
    name: str
    score: float


def _make_client() -> GeminiClient:
    with patch("volly.gemini_client.genai.Client") as ctor:
        ctor.return_value = MagicMock()
        return GeminiClient(api_key="test-key")


def _response(*, text: str = "", parsed: Any = None) -> MagicMock:
    resp = MagicMock(spec=types.GenerateContentResponse)
    resp.text = text
    resp.parsed = parsed
    return resp


def _install_generate(client: GeminiClient, mock: AsyncMock) -> None:
    client._client.aio.models.generate_content = mock


def test_init_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("volly.gemini_client.load_dotenv"):
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            GeminiClient()


def test_init_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    with patch("volly.gemini_client.genai.Client") as ctor:
        GeminiClient()
        ctor.assert_called_once_with(api_key="env-key")


async def test_text_returns_response_text() -> None:
    client = _make_client()
    mock = AsyncMock(return_value=_response(text="hello"))
    _install_generate(client, mock)

    out = await client.text("sys", "hi", thinking=Thinking.LOW, temperature=0.7)

    assert out == "hello"
    call = mock.await_args
    assert call.kwargs["model"] == "gemini-3.5-flash"
    assert call.kwargs["contents"] == ["hi"]
    config = call.kwargs["config"]
    assert config.system_instruction == "sys"
    assert config.temperature == 0.7
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert config.response_mime_type is None


async def test_multimodal_attaches_image_parts() -> None:
    client = _make_client()
    mock = AsyncMock(return_value=_response(text="seen"))
    _install_generate(client, mock)

    img1 = PIL.Image.new("RGB", (4, 4), "white")
    img2 = PIL.Image.new("RGB", (4, 4), "black")
    out = await client.multimodal("sys", "look", [img1, img2], thinking=Thinking.HIGH)

    assert out == "seen"
    contents = mock.await_args.kwargs["contents"]
    assert contents[0] == "look"
    assert len(contents) == 3
    assert all(isinstance(p, types.Part) for p in contents[1:])
    assert all(p.inline_data.mime_type == "image/png" for p in contents[1:])
    config = mock.await_args.kwargs["config"]
    assert config.thinking_config.thinking_level == types.ThinkingLevel.HIGH


async def test_json_returns_parsed_pydantic() -> None:
    client = _make_client()
    parsed = _Schema(name="cat", score=0.8)
    mock = AsyncMock(return_value=_response(parsed=parsed))
    _install_generate(client, mock)

    out = await client.json("sys", "rank", _Schema)

    assert out == parsed
    config = mock.await_args.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is _Schema


async def test_json_falls_back_to_text_parse() -> None:
    client = _make_client()
    mock = AsyncMock(return_value=_response(text='{"name":"cat","score":0.9}', parsed=None))
    _install_generate(client, mock)

    out = await client.json("sys", "rank", _Schema)

    assert out == _Schema(name="cat", score=0.9)


async def test_json_retries_on_validation_error() -> None:
    client = _make_client()
    mock = AsyncMock(
        side_effect=[
            _response(text="not json at all"),
            _response(text='{"name":"cat","score":0.5}'),
        ]
    )
    _install_generate(client, mock)

    out = await client.json("sys", "rank", _Schema, max_retries=1)

    assert out == _Schema(name="cat", score=0.5)
    assert mock.await_count == 2
    second_user = mock.await_args_list[1].kwargs["contents"][0]
    assert "did not match the schema" in second_user
    assert "rank" in second_user


async def test_json_raises_after_max_retries() -> None:
    client = _make_client()
    mock = AsyncMock(return_value=_response(text="garbage"))
    _install_generate(client, mock)

    with pytest.raises(ValidationError):
        await client.json("sys", "rank", _Schema, max_retries=1)

    assert mock.await_count == 2


async def test_generate_retries_on_transient_status(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client()
    transient = genai_errors.APIError(code=503, response_json={"error": {"message": "busy"}})
    ok = _response(text="finally")
    mock = AsyncMock(side_effect=[transient, transient, ok])
    _install_generate(client, mock)
    monkeypatch.setattr("volly.gemini_client._sleep_backoff", AsyncMock(return_value=None))

    out = await client.text("sys", "hi")

    assert out == "finally"
    assert mock.await_count == 3


async def test_generate_does_not_retry_on_400(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client()
    bad = genai_errors.APIError(code=400, response_json={"error": {"message": "nope"}})
    mock = AsyncMock(side_effect=bad)
    _install_generate(client, mock)
    monkeypatch.setattr("volly.gemini_client._sleep_backoff", AsyncMock(return_value=None))

    with pytest.raises(genai_errors.APIError):
        await client.text("sys", "hi")

    assert mock.await_count == 1


async def test_generate_raises_after_max_transport_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    transient = genai_errors.APIError(code=429, response_json={"error": {"message": "slow"}})
    mock = AsyncMock(side_effect=transient)
    _install_generate(client, mock)
    monkeypatch.setattr("volly.gemini_client._sleep_backoff", AsyncMock(return_value=None))

    with pytest.raises(genai_errors.APIError):
        await client.text("sys", "hi")

    assert mock.await_count == 3
