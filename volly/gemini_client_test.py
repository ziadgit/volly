"""Tests for ``volly.gemini_client``. No network — SDK call is mocked."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import PIL.Image
import pytest
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

from volly.gemini_client import (
    GeminiClient,
    Thinking,
    _parse_retry_delay,
    _resolve_rpm,
    _RpmLimiter,
)


class _Schema(BaseModel):
    name: str
    score: float


@pytest.fixture(autouse=True)
def _clean_rpm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate every test from a developer's ``GEMINI_RPM`` env setting."""
    monkeypatch.delenv("GEMINI_RPM", raising=False)


def _make_client(rpm: int | None = None) -> GeminiClient:
    with patch("volly.gemini_client.genai.Client") as ctor:
        ctor.return_value = MagicMock()
        return GeminiClient(api_key="test-key", rpm=rpm)


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


# --- RPM resolution -------------------------------------------------------


def test_resolve_rpm_default() -> None:
    assert _resolve_rpm(None) == 30


def test_resolve_rpm_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_RPM", "5")
    assert _resolve_rpm(None) == 5


def test_resolve_rpm_arg_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_RPM", "5")
    assert _resolve_rpm(900) == 900


def test_resolve_rpm_empty_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_RPM", "")
    assert _resolve_rpm(None) == 30


def test_resolve_rpm_invalid_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_RPM", "fast")
    with pytest.raises(ValueError, match="GEMINI_RPM"):
        _resolve_rpm(None)


def test_resolve_rpm_nonpositive_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_RPM", "0")
    with pytest.raises(ValueError, match="positive"):
        _resolve_rpm(None)


def test_resolve_rpm_nonpositive_arg_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        _resolve_rpm(-1)


def test_client_init_exposes_rpm() -> None:
    client = _make_client(rpm=7)
    assert client.rpm == 7


def test_client_init_picks_env_rpm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_RPM", "5")
    assert _make_client().rpm == 5


def test_client_init_defaults_to_30() -> None:
    assert _make_client().rpm == 30


# --- RPM limiter mechanics ------------------------------------------------


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _build_limiter(rpm: int) -> tuple[_RpmLimiter, _FakeClock, list[float]]:
    clock = _FakeClock()
    sleeps: list[float] = []

    async def fake_sleep(secs: float) -> None:
        sleeps.append(secs)
        clock.t += secs

    return _RpmLimiter(rpm, clock=clock, sleep=fake_sleep), clock, sleeps


async def test_limiter_initial_burst_does_not_sleep() -> None:
    limiter, _clock, sleeps = _build_limiter(rpm=5)
    for _ in range(5):
        await limiter.acquire()
    assert sleeps == []


async def test_limiter_paces_after_burst() -> None:
    limiter, clock, sleeps = _build_limiter(rpm=60)
    # 1 token/sec refill, capacity 60. Burn the bucket.
    for _ in range(60):
        await limiter.acquire()
    assert sleeps == []
    # Sixty-first must wait ~1s for a token to refill.
    await limiter.acquire()
    assert sleeps == pytest.approx([1.0])
    assert clock.t == pytest.approx(1.0)
    # Sixty-second waits another ~1s.
    await limiter.acquire()
    assert sleeps[-1] == pytest.approx(1.0)
    assert clock.t == pytest.approx(2.0)


async def test_limiter_refills_with_wall_time() -> None:
    limiter, clock, sleeps = _build_limiter(rpm=60)
    for _ in range(60):
        await limiter.acquire()
    # Pretend 10s passed externally — bucket should hold ~10 tokens now.
    clock.t = 10.0
    for _ in range(10):
        await limiter.acquire()
    assert sleeps == []  # ten free tokens, no sleep needed


async def test_limiter_capacity_is_capped_at_rpm() -> None:
    limiter, clock, sleeps = _build_limiter(rpm=5)
    # Don't drain. Wait an hour of wall time; refill cannot exceed capacity.
    clock.t = 3600.0
    for _ in range(5):
        await limiter.acquire()
    assert sleeps == []
    await limiter.acquire()  # sixth must wait
    assert sleeps  # at least one sleep recorded


def test_limiter_rejects_nonpositive_rpm() -> None:
    with pytest.raises(ValueError, match="positive"):
        _RpmLimiter(0)


async def test_generate_acquires_one_token_per_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every SDK attempt, including retries, must acquire one RPM token."""
    client = _make_client()
    transient = genai_errors.APIError(code=503, response_json={"error": {"message": "busy"}})
    ok = _response(text="done")
    mock = AsyncMock(side_effect=[transient, ok])
    _install_generate(client, mock)
    monkeypatch.setattr("volly.gemini_client._sleep_backoff", AsyncMock(return_value=None))

    acquired = 0

    async def counting_acquire() -> None:
        nonlocal acquired
        acquired += 1

    client._rpm_limiter.acquire = counting_acquire  # type: ignore[method-assign]

    out = await client.text("sys", "hi")

    assert out == "done"
    assert mock.await_count == 2
    assert acquired == 2


# --- RetryInfo parsing ----------------------------------------------------


def _quota_error(retry_delay: str | None = "44s", *, code: int = 429) -> genai_errors.APIError:
    """Build an ``APIError`` shaped like Gemini's real 429 response body."""
    details: list[dict[str, Any]] = []
    if retry_delay is not None:
        details.append(
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": retry_delay,
            }
        )
    return genai_errors.APIError(
        code=code,
        response_json={
            "error": {
                "code": code,
                "message": "Resource exhausted",
                "status": "RESOURCE_EXHAUSTED",
                "details": details,
            }
        },
    )


def test_parse_retry_delay_seconds() -> None:
    assert _parse_retry_delay(_quota_error("44s")) == 44.0


def test_parse_retry_delay_fractional() -> None:
    assert _parse_retry_delay(_quota_error("44.5s")) == 44.5


def test_parse_retry_delay_returns_none_when_details_absent() -> None:
    exc = genai_errors.APIError(code=429, response_json={"error": {"message": "slow"}})
    assert _parse_retry_delay(exc) is None


def test_parse_retry_delay_returns_none_when_response_unstructured() -> None:
    # SDK occasionally stuffs a plain string into response_json on unusual errors.
    exc = genai_errors.APIError(code=429, response_json="quota exhausted")
    assert _parse_retry_delay(exc) is None


def test_parse_retry_delay_returns_none_for_malformed_delay() -> None:
    assert _parse_retry_delay(_quota_error("44")) is None  # missing "s" suffix
    assert _parse_retry_delay(_quota_error("44ms")) is None  # wrong unit
    assert _parse_retry_delay(_quota_error("")) is None


def test_parse_retry_delay_skips_non_retry_info_details() -> None:
    exc = genai_errors.APIError(
        code=429,
        response_json={
            "error": {
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "RATE_LIMIT"},
                ]
            }
        },
    )
    assert _parse_retry_delay(exc) is None


def test_parse_retry_delay_handles_missing_retry_delay_field() -> None:
    exc = genai_errors.APIError(
        code=429,
        response_json={
            "error": {
                "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo"}]
            }
        },
    )
    assert _parse_retry_delay(exc) is None


# --- 429 retry-delay handling ---------------------------------------------


async def test_generate_honors_server_retry_delay_on_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 429 carrying RetryInfo must sleep ~retryDelay+jitter, then retry."""
    client = _make_client()
    quota = _quota_error("2s")
    ok = _response(text="finally")
    mock = AsyncMock(side_effect=[quota, ok])
    _install_generate(client, mock)
    monkeypatch.setattr("volly.gemini_client._retry_delay_jitter", lambda: 0.5)
    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("volly.gemini_client._sleep_retry_delay", fake_sleep)
    backoff = AsyncMock(return_value=None)
    monkeypatch.setattr("volly.gemini_client._sleep_backoff", backoff)

    out = await client.text("sys", "hi")

    assert out == "finally"
    assert mock.await_count == 2
    assert sleep_calls == [pytest.approx(2.5)]
    assert backoff.await_count == 0  # backoff path skipped when RetryInfo present


async def test_generate_falls_back_to_backoff_without_retry_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 429 with no RetryInfo must use the existing exponential backoff path."""
    client = _make_client()
    quota = genai_errors.APIError(code=429, response_json={"error": {"message": "slow"}})
    ok = _response(text="done")
    mock = AsyncMock(side_effect=[quota, ok])
    _install_generate(client, mock)
    retry_sleep = AsyncMock(return_value=None)
    monkeypatch.setattr("volly.gemini_client._sleep_retry_delay", retry_sleep)
    backoff = AsyncMock(return_value=None)
    monkeypatch.setattr("volly.gemini_client._sleep_backoff", backoff)

    out = await client.text("sys", "hi")

    assert out == "done"
    assert mock.await_count == 2
    assert retry_sleep.await_count == 0
    assert backoff.await_count == 1


async def test_generate_raises_when_retry_delay_exceeds_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retryDelay larger than the 90s cap must surface the APIError to callers."""
    client = _make_client()
    quota = _quota_error("100s")
    mock = AsyncMock(side_effect=quota)
    _install_generate(client, mock)
    monkeypatch.setattr("volly.gemini_client._retry_delay_jitter", lambda: 0.0)
    retry_sleep = AsyncMock(return_value=None)
    monkeypatch.setattr("volly.gemini_client._sleep_retry_delay", retry_sleep)

    with pytest.raises(genai_errors.APIError):
        await client.text("sys", "hi")

    # No sleep — we bailed before waiting, and there was only one transport attempt.
    assert mock.await_count == 1
    assert retry_sleep.await_count == 0


async def test_generate_raises_when_cumulative_retry_delay_exceeds_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two 50s retryDelays sum past 90s; the second attempt must raise."""
    client = _make_client()
    quota = _quota_error("50s")
    mock = AsyncMock(side_effect=[quota, quota, _response(text="never reached")])
    _install_generate(client, mock)
    monkeypatch.setattr("volly.gemini_client._retry_delay_jitter", lambda: 0.0)
    retry_sleep = AsyncMock(return_value=None)
    monkeypatch.setattr("volly.gemini_client._sleep_retry_delay", retry_sleep)

    with pytest.raises(genai_errors.APIError):
        await client.text("sys", "hi")

    # First 429: slept 50s. Second 429: would push total to 100s > 90s cap → raise.
    assert mock.await_count == 2
    assert retry_sleep.await_count == 1


# --- Throttle + 429-retry logging ----------------------------------------


async def test_limiter_logs_throttle_when_sleeping(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """First sleeping acquire must emit an INFO throttled line."""
    limiter, _clock, _sleeps = _build_limiter(rpm=60)
    for _ in range(60):
        await limiter.acquire()  # exhaust bucket

    with caplog.at_level(logging.INFO, logger="volly.gemini_client"):
        await limiter.acquire()  # this one must sleep

    throttled = [r for r in caplog.records if "throttled" in r.getMessage()]
    assert len(throttled) == 1
    msg = throttled[0].getMessage()
    assert "rpm=60" in msg
    assert "queued=1" in msg
    assert "eta" in msg
    assert throttled[0].levelno == logging.INFO


async def test_limiter_does_not_log_when_token_available(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Acquires that do not sleep must not emit a throttle line."""
    limiter, _clock, _sleeps = _build_limiter(rpm=5)
    with caplog.at_level(logging.INFO, logger="volly.gemini_client"):
        for _ in range(5):
            await limiter.acquire()
    throttled = [r for r in caplog.records if "throttled" in r.getMessage()]
    assert throttled == []


async def test_limiter_squelches_throttle_logs_within_one_second(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A second sleeping acquire inside the 1s window must be squelched.

    rpm=120 → 2 tokens/sec → each token costs 0.5s. After exhausting the
    bucket, three back-to-back acquires fire at clock=0.0, 0.5, 1.0. With
    a 1s squelch window the first and third log; the middle is dropped.
    """
    limiter, _clock, _sleeps = _build_limiter(rpm=120)
    for _ in range(120):
        await limiter.acquire()  # exhaust bucket, no logs (no sleep yet)

    with caplog.at_level(logging.INFO, logger="volly.gemini_client"):
        await limiter.acquire()  # logs at clock=0
        await limiter.acquire()  # at clock=0.5 → squelched (Δ=0.5 < 1.0)
        await limiter.acquire()  # at clock=1.0 → logs (Δ=1.0 ≥ 1.0)

    throttled = [r for r in caplog.records if "throttled" in r.getMessage()]
    assert len(throttled) == 2


async def test_generate_logs_server_retry_delay_at_info(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Honoring a server retryDelay must emit one INFO line, not WARNING."""
    client = _make_client()
    mock = AsyncMock(side_effect=[_quota_error("3s"), _response(text="ok")])
    _install_generate(client, mock)
    monkeypatch.setattr("volly.gemini_client._retry_delay_jitter", lambda: 0.5)
    monkeypatch.setattr(
        "volly.gemini_client._sleep_retry_delay", AsyncMock(return_value=None)
    )

    with caplog.at_level(logging.INFO, logger="volly.gemini_client"):
        out = await client.text("sys", "hi")

    assert out == "ok"
    retry_logs = [r for r in caplog.records if "429 retry in" in r.getMessage()]
    assert len(retry_logs) == 1
    assert "3.5s" in retry_logs[0].getMessage()
    assert "(server)" in retry_logs[0].getMessage()
    assert retry_logs[0].levelno == logging.INFO


async def test_generate_does_not_log_retry_when_exceeding_cap(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If we bail without sleeping, no 429-retry line should appear."""
    client = _make_client()
    mock = AsyncMock(side_effect=_quota_error("100s"))
    _install_generate(client, mock)
    monkeypatch.setattr("volly.gemini_client._retry_delay_jitter", lambda: 0.0)
    monkeypatch.setattr(
        "volly.gemini_client._sleep_retry_delay", AsyncMock(return_value=None)
    )

    with caplog.at_level(logging.INFO, logger="volly.gemini_client"):
        with pytest.raises(genai_errors.APIError):
            await client.text("sys", "hi")

    retry_logs = [r for r in caplog.records if "429 retry in" in r.getMessage()]
    assert retry_logs == []
