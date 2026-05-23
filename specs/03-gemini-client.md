# 03 — Gemini Client

Thin async wrapper around `google-generativeai` for Flash 3.5. The only
module that touches the SDK directly. Every other module imports from here.

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

- **Thinking config:** map `Thinking.LOW/MEDIUM/HIGH` to the Gemini SDK's
  thinking-budget config. Verify exact field name from the SDK at impl
  time — that's a known unknown.
- **JSON mode:** use the SDK's `response_mime_type="application/json"` and
  `response_schema=` if available; otherwise wrap with an explicit "respond
  ONLY with JSON conforming to:" preamble.
- **Validation:** parse with the provided pydantic schema. On
  `ValidationError`, retry up to `max_retries` with the parser error
  appended to the user message.
- **Retries:** tenacity-style retry with jittered exponential backoff on
  429, 500, 503. Three attempts total. Surface anything else.
- **Image encoding:** accept `PIL.Image.Image`; convert internally to the
  SDK's expected format (PNG bytes via `BytesIO`).

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
