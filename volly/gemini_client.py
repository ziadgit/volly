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
from enum import StrEnum
from typing import TypeVar

import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

_DEFAULT_MODEL = "gemini-3.5-flash"
_RETRY_STATUSES = frozenset({429, 500, 503})
_MAX_TRANSPORT_ATTEMPTS = 3

T = TypeVar("T", bound=BaseModel)


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

    def __init__(self, model: str = _DEFAULT_MODEL, api_key: str | None = None) -> None:
        load_dotenv()
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Export it or put it in a .env file."
            )
        self.model = model
        self._client = genai.Client(api_key=key)

    async def _generate(
        self,
        contents: list[types.PartUnion],
        config: types.GenerateContentConfig,
    ) -> types.GenerateContentResponse:
        last_exc: BaseException | None = None
        for attempt in range(_MAX_TRANSPORT_ATTEMPTS):
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
