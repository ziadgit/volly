# 06 — Judge

The single most important module. **Always renders ASCII to an image
before judging.** Vision judgment dominates text-grid judgment on spatial
tasks, and this is the implementation detail the whole thesis rides on.

## Public surface

```python
class CandidateScore(BaseModel):
    index: int
    score: float = Field(ge=0.0, le=1.0)
    why: str  # ≤ 1 sentence

class JudgeResult(BaseModel):
    scores: list[CandidateScore]
    best_index: int
    worst_index: int
    critique: str          # 2-4 sentences, prose
    prompt_suggestions: list[str]  # concrete, actionable bullets

async def rank(
    client: GeminiClient,
    subject: str,
    system_prompt: str,
    images: list[PIL.Image.Image],
    *,
    history: "RunHistory | None" = None,
    thinking: Thinking = Thinking.HIGH,
) -> JudgeResult: ...
```

## Behavior

- Sends ALL `N` images in a SINGLE call. The judge has to see them
  side-by-side to rank relatively. Do not call N times.
- Includes:
  - The target subject.
  - The current system prompt the actor used.
  - The history of previous iterations' best images + their judge scores +
    their critiques. The judge uses this to track progress and avoid
    repeating prior advice. (Trim history to the last 4 iterations if
    context gets large — Flash 3.5's 1M window makes this easy.)
- Returns JSON conforming to `JudgeResult` via `client.json(schema=...)`.
- On `ValidationError` after retries, fall back to text-judge mode
  (`prompt_suggestions=[]`, `scores=uniform`) and log the fallback. The
  loop continues; we'd rather degrade than crash mid-demo.

## System prompt for the judge

```
You are evaluating ASCII art renderings of "{subject}".
You will see {N} candidate images.
Score each 0.0–1.0 on: recognizability of the subject, composition,
proportions, and use of negative space.
Identify the best and worst.
Then suggest 1–3 concrete improvements to the artist's system prompt —
the prompt is included below. Be specific, not generic.

Current artist system prompt:
---
{system_prompt}
---
```

## Why suggestions live here

The rewriter (`07-rewriter.md`) needs concrete material. The judge has all
the context (images + prompt + history); having it propose deltas is
strictly cheaper than calling a second model.

## Test surface

- `judge_test.py` mocks `GeminiClient.json` to return canned
  `JudgeResult` objects.
- Verifies the judge call includes all `N` images and the history
  truncation to ≤ 4 prior iterations.
- Verifies the fallback path on persistent `ValidationError`.

## Ablation hook

P2 backlog: text-only judge mode (`include_images=False`) for the
ablation talking point. Same schema, no images attached, subject + raw
ASCII text in the user message.
