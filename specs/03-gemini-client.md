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
        temperature: float = 1.0,
    ) -> str: ...

    async def json[T: BaseModel](
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        images: list[PIL.Image.Image] | None = None,
        thinking: Thinking = Thinking.HIGH,
        temperature: float = 1.0,
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

## Rate limiting

Free-tier Gemini Flash 3.5 is **5 RPM** per model. The default iteration
of this app fires ~19 parallel calls. The client must throttle, not the
callers — actor/judge/rewriter remain free to dispatch as many concurrent
calls as they want.

### Design

- One `asyncio.Lock` + monotonic token-bucket state per `GeminiClient`
  instance. Bucket capacity = `rpm`; refill rate = `rpm / 60` tokens/sec.
- Every `_generate` call awaits `_acquire_token()` before issuing the
  SDK call. Bursting callers are queued (FIFO via the lock); no caller
  is refused.
- `rpm` resolution at construction time:
  `arg → GEMINI_RPM env → 30 (default)`. `30` suits paid tier without
  surprises; free-tier operators use `--tier free` on the loop CLI,
  which sets rpm=4 — one below the 5 RPM Gemini ceiling as a safety
  margin against burst/clock drift (see spec 02 §"Tier presets").

### 429 handling

Existing exponential backoff is **insufficient** because it sleeps
1/2/4s while Gemini's `RetryInfo.retryDelay` is typically 44–60s. The
new rule:

1. On 429, parse `RetryInfo.retryDelay` from the JSON error body —
   `error.details[*]["@type"] == "type.googleapis.com/google.rpc.RetryInfo"`,
   field `retryDelay` shaped like `"44s"` / `"44.5s"`.
2. Sleep `retryDelay + random(0, 2)` seconds (jitter to avoid herd).
3. Retry. Repeat up to a **total wall-time cap of 90s per call**; after
   that, raise `APIError` so callers can fall back gracefully.
4. If `RetryInfo` is missing or unparseable, fall back to the existing
   exponential backoff (1s, 2s, 4s).

### Patient mode (`--max-retry-wait`)

The 90s default cap is wrong for two real scenarios: sponsorship keys that
reset hourly, and free-tier daily caps that reset at UTC midnight. The
client takes a `max_retry_wait_s: float` constructor arg (default 90,
overridable per-`GeminiClient`) wired to the `--max-retry-wait` CLI flag:

- If `retryDelay > max_retry_wait_s`, do NOT raise. Log WARNING:
  `gemini: quota locked, eta=Xs > max_retry_wait=Ys, pausing; Ctrl-C to abort`
- Sleep `retryDelay + jitter`. Repeat once per server-stated `retryDelay`
  window until a non-429 response arrives or the operator hits Ctrl-C.
- Emit a once-per-30s INFO heartbeat: `gemini: still waiting, eta≈Xs`.
  This is the operator's signal that the process is alive and just
  patient — not hung.

Rationale: an unattended Ralph or rehearsal run should outlive a daily
quota cap and resume itself. The 90s default keeps interactive demos from
silently hanging when the operator just expected a fast failure.

### Logging contract

- INFO `gemini: throttled rpm=R queued=Q eta≈Ts` when the limiter holds a
  call. At most one per second per caller (use a `_last_throttle_log_at`
  monotonic timer keyed by caller id; squelch the rest).
- INFO `gemini: 429 retry in Xs (server)` when honoring `retryDelay`.
- WARNING is reserved for genuinely surprising errors (5xx after retry
  cap, schema validation after retries). Day-to-day rate limiting is INFO.

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
