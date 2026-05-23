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

Text-only judge mode for the ablation talking point: same schema, no
images attached, subject + raw ASCII text in the user message.

```python
await rank(
    client, subject, system_prompt, images,
    include_images=False, texts=[c.text for c in cands],
)
```

`texts` is required when `include_images=False`; its length must match
`len(images)` because the loop calls vision-judge and text-judge on the
same candidate set. The text-only system prompt drops the "candidate
images" phrasing and says "candidate drawings as raw ASCII text (no
images attached)". Prior-iteration image attachments are also omitted in
text-only mode, but their textual critique summaries are kept — the
ablation is about *this* iteration's candidates, not history.

Invoked from `volly.loop` when `--ablate-judge` is set. After the vision
judge runs, the loop calls text-judge on the same candidates and logs a
single line per arm per iteration:

```
ablation iter N arm <evolving|control>: vision_top3=A.AAA text_top3=B.BBB delta=±C.CCC
```

Failures in the text-judge call are logged + swallowed (`_log.exception`)
— the live loop never crashes on an ablation degradation.
