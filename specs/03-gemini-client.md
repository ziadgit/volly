# 03 — Gemini Client

Thin async wrapper around `google-genai` (the modern Gemini SDK) for
Flash 3.5. The only module that touches the SDK directly. Every other
module imports from here.

## Why a wrapper

- One place to set thinking level (Low/Medium/High) per call.
- One place to do JSON-mode + pydantic parse + retry on malformed output.
- One place to send multimodal inputs (PIL images + text).
- One place to retry on transient 429/500.
- Trivially mockable in tests — no network in `*_test.py`.

## Public surface

```python
class Thinking(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class GeminiClient:
    def __init__(self, model: str = "gemini-3.5-flash", api_key: str | None = None): ...

    async def text(
        self,
        system: str,
        user: str,
        *,
        thinking: Thinking = Thinking.LOW,
        temperature: float = 1.0,
    ) -> str: ...

    async def multimodal(
        self,
        system: str,
        user: str,
        images: list[PIL.Image.Image],
        *,
        thinking: Thinking = Thinking.HIGH,
    ) -> str: ...

    async def json[T: BaseModel](
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        images: list[PIL.Image.Image] | None = None,
        thinking: Thinking = Thinking.HIGH,
        max_retries: int = 2,
    ) -> T: ...
```

## Behavior

- **Thinking config:** map `Thinking.LOW/MEDIUM/HIGH` to
  `google.genai.types.ThinkingConfig(thinking_level=ThinkingLevel.LOW
  /MEDIUM/HIGH)`, set on `GenerateContentConfig.thinking_config`. The
  SDK also exposes `ThinkingLevel.MINIMAL`; we do not use it.
- **Async call shape:** `client.aio.models.generate_content(
  model=..., contents=..., config=GenerateContentConfig(
  system_instruction=..., temperature=..., thinking_config=...))`.
  No `ChatSession` — each call is independent.
- **JSON mode:** set `response_mime_type="application/json"` and
  `response_schema=<pydantic model class>` on `GenerateContentConfig`.
  The SDK accepts pydantic v2 model classes directly for `response_schema`.
- **Validation:** parse the returned text with the provided pydantic
  schema. On `ValidationError`, retry up to `max_retries` with the parser
  error appended to the user message.
- **Retries:** retry with jittered exponential backoff on 429, 500, 503.
  Three attempts total. Surface anything else. Use a small handwritten
  retry helper — no `tenacity` dep.
- **Image encoding:** accept `PIL.Image.Image`; convert internally to
  `types.Part.from_bytes(mime_type="image/png", data=<PNG bytes via
  BytesIO>)`. User content is then a list mixing the user-text string
  and image parts.

## Auth

- Reads `GEMINI_API_KEY` from environment if `api_key` is None.
- Loads `.env` via `python-dotenv` if it exists in CWD.

## Concurrency

- Each call creates a new SDK session — safe to `asyncio.gather` hundreds.
- No global state. Tests can construct independent clients.

## Test surface

- `gemini_client_test.py` mocks the SDK call layer at the transport boundary.
- No tests hit the network. Live calls go in a separate `tests/live/` (out
  of scope for v1).
