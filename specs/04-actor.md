# 04 — Actor

Generates N ASCII candidates in parallel from a given system prompt.

## Public surface

```python
@dataclass(frozen=True)
class Candidate:
    text: str          # raw model output, post-fence-stripped
    index: int         # 0..N-1
    raw: str           # original model response, untouched

async def generate(
    client: GeminiClient,
    system_prompt: str,
    subject: str,
    *,
    k: int = 8,
    thinking: Thinking = Thinking.LOW,
) -> list[Candidate]: ...
```

## Behavior

- Constructs `k` independent calls with **identical** system + user messages.
  Diversity comes from sampling temperature, not prompt variation.
- `asyncio.gather` with `return_exceptions=True`. Failed calls are dropped
  silently; if `len(successes) < k`, log and continue with fewer candidates.
  The loop has a separate pad-from-prior-best fallback.
- Temperature defaults to 1.0; can be raised to widen exploration.
- **Fence stripping:** if the model wraps output in
  ```` ``` ```` or ```` ```ascii ````, strip the fences before storing
  `.text`. Keep the un-stripped `.raw` for the judge if needed (it usually
  isn't, because the judge sees the *rendered image*, not the text).
- Subject sanitization happens UPSTREAM — actor trusts its input.

## User message format

```
Subject: {subject}

Draw it as ASCII art. Respond with only the ASCII drawing, no commentary,
no code fences.
```

The "no commentary, no code fences" is belt-and-suspenders — the fence
stripper handles violations gracefully.

## Test surface

- Mocks `GeminiClient.text` to return canned strings.
- Verifies parallelism (all `k` calls dispatched before any awaited).
- Verifies fence stripping for triple-backtick variants.
- Verifies exception swallowing and partial-result behavior.
