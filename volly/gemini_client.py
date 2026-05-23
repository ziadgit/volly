"""Async wrapper around ``google-genai`` for Gemini 3.5 Flash.

The only module in Volly that touches the SDK directly. Every other module
(actor, judge, rewriter) goes through :class:`GeminiClient`. See
``specs/03-gemini-client.md`` for the contract.
"""

from __future__ import annotations

import asyncio
import io
import os
import random
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import TypeVar

import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

_DEFAULT_MODEL = "gemini-3.5-flash"
_DEFAULT_RPM = 30
_RETRY_STATUSES = frozenset({429, 500, 503})
_MAX_TRANSPORT_ATTEMPTS = 3

T = TypeVar("T", bound=BaseModel)


def _resolve_rpm(rpm: int | None) -> int:
    """Resolve effective RPM: explicit arg → ``GEMINI_RPM`` env → default 30."""
    if rpm is not None:
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        return rpm
    env = os.environ.get("GEMINI_RPM")
    if env is None or env == "":
        return _DEFAULT_RPM
    try:
        parsed = int(env)
    except ValueError as exc:
        raise ValueError(f"GEMINI_RPM={env!r} is not an integer") from exc
    if parsed <= 0:
        raise ValueError(f"GEMINI_RPM must be positive, got {parsed}")
    return parsed


class _RpmLimiter:
    """Token-bucket async rate limiter, FIFO via ``asyncio.Lock``.

    Bucket starts full (capacity = ``rpm`` tokens) and refills at
    ``rpm / 60`` tokens/sec. Every acquire takes exactly one token; when
    empty, the caller sleeps inside the lock until a token is available,
    which serializes waiters and gives FIFO ordering for free.
    """

    def __init__(
        self,
        rpm: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        self.rpm = rpm
        self._capacity = float(rpm)
        self._refill_per_sec = rpm / 60.0
        self._clock = clock
        self._sleep = sleep
        self._tokens = float(rpm)
        self._last_refill = clock()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = self._clock()
                elapsed = now - self._last_refill
                if elapsed > 0:
                    self._tokens = min(
                        self._capacity,
                        self._tokens + elapsed * self._refill_per_sec,
                    )
                    self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._refill_per_sec
                await self._sleep(wait)


class Thinking(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_THINKING_MAP: dict[Thinking, types.ThinkingLevel] = {
    Thinking.LOW: types.ThinkingLevel.LOW,
    Thinking.MEDIUM: types.ThinkingLevel.MEDIUM,
    Thinking.HIGH: types.ThinkingLevel.HIGH,
}


def _image_part(image: PIL.Image.Image) -> types.Part:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return types.Part.from_bytes(mime_type="image/png", data=buf.getvalue())


def _build_config(
    system: str,
    *,
    thinking: Thinking,
    temperature: float,
    response_mime_type: str | None = None,
    response_schema: type[BaseModel] | None = None,
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        thinking_config=types.ThinkingConfig(thinking_level=_THINKING_MAP[thinking]),
        response_mime_type=response_mime_type,
        response_schema=response_schema,
    )


def _build_contents(user: str, images: list[PIL.Image.Image] | None) -> list[types.PartUnion]:
    parts: list[types.PartUnion] = [user]
    if images:
        parts.extend(_image_part(img) for img in images)
    return parts


async def _sleep_backoff(attempt: int) -> None:
    # 0.5s, 1.0s base with full jitter.
    base = 0.5 * (2**attempt)
    await asyncio.sleep(base * (0.5 + random.random()))


class GeminiClient:
    """Thin async wrapper over ``google.genai`` for Gemini 3.5 Flash."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        *,
        rpm: int | None = None,
    ) -> None:
        load_dotenv()
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Export it or put it in a .env file."
            )
        self.model = model
        self._client = genai.Client(api_key=key)
        self._rpm_limiter = _RpmLimiter(_resolve_rpm(rpm))

    @property
    def rpm(self) -> int:
        """Effective requests-per-minute ceiling enforced by the limiter."""
        return self._rpm_limiter.rpm

    async def _generate(
        self,
        contents: list[types.PartUnion],
        config: types.GenerateContentConfig,
    ) -> types.GenerateContentResponse:
        last_exc: BaseException | None = None
        for attempt in range(_MAX_TRANSPORT_ATTEMPTS):
            await self._rpm_limiter.acquire()
            try:
                return await self._client.aio.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
            except genai_errors.APIError as exc:
                if exc.code not in _RETRY_STATUSES or attempt == _MAX_TRANSPORT_ATTEMPTS - 1:
                    raise
                last_exc = exc
                await _sleep_backoff(attempt)
        assert last_exc is not None  # unreachable; keeps the type-checker honest
        raise last_exc

    async def text(
        self,
        system: str,
        user: str,
        *,
        thinking: Thinking = Thinking.LOW,
        temperature: float = 1.0,
    ) -> str:
        config = _build_config(system, thinking=thinking, temperature=temperature)
        response = await self._generate(_build_contents(user, None), config)
        return response.text or ""

    async def multimodal(
        self,
        system: str,
        user: str,
        images: list[PIL.Image.Image],
        *,
        thinking: Thinking = Thinking.HIGH,
        temperature: float = 1.0,
    ) -> str:
        config = _build_config(system, thinking=thinking, temperature=temperature)
        response = await self._generate(_build_contents(user, images), config)
        return response.text or ""

    async def json(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        images: list[PIL.Image.Image] | None = None,
        thinking: Thinking = Thinking.HIGH,
        temperature: float = 1.0,
        max_retries: int = 2,
    ) -> T:
        attempts = max_retries + 1
        current_user = user
        last_error: ValidationError | None = None
        for attempt in range(attempts):
            config = _build_config(
                system,
                thinking=thinking,
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=schema,
            )
            response = await self._generate(_build_contents(current_user, images), config)
            parsed = response.parsed
            if isinstance(parsed, schema):
                return parsed
            raw = response.text or ""
            try:
                return schema.model_validate_json(raw)
            except ValidationError as exc:
                last_error = exc
                if attempt == attempts - 1:
                    raise
                current_user = (
                    f"{user}\n\nYour previous response did not match the schema. "
                    f"Parser error:\n{exc}\nReturn valid JSON only."
                )
        assert last_error is not None  # unreachable
        raise last_error
