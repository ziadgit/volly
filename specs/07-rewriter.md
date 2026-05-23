# 07 — Rewriter

Takes the judge's critique and produces a new system prompt for the next
iteration. Runs at High thinking. Pure text-in, text-out.

## Public surface

```python
async def rewrite(
    client: GeminiClient,
    current_prompt: str,
    judge_result: JudgeResult,
    subject: str,
    *,
    thinking: Thinking = Thinking.HIGH,
) -> str: ...
```

## Behavior

- Calls `GeminiClient.text` with a system prompt that explains the
  rewriter's role and a user message that includes:
  - The current system prompt (verbatim, fenced).
  - The judge's critique (`judge_result.critique`).
  - The judge's `prompt_suggestions` as bullets.
  - The subject (so the new prompt can stay specific without overfitting).
- Returns a single new system prompt string.
- **Invariants on the returned prompt:**
  - Length ≤ 4000 characters. If longer, the rewriter must collapse;
    enforce post-hoc with a hard truncate as a safety net.
  - Does not contain the literal subject more than twice (avoid overfitting
    to one drawing).
  - Always starts with "You are an ASCII artist." (sanity anchor).
- Diff is computed by `state.py`, not here.

## System prompt for the rewriter

```
You are improving a system prompt used by an ASCII artist.
The artist's task is to draw "{subject}" — but the new prompt should
generalize to other subjects too.

You will receive:
- the current system prompt
- a critique from a vision judge
- specific suggestions from the judge

Produce the new system prompt. Rules:
- Start with "You are an ASCII artist."
- ≤ 4000 chars.
- Mention "{subject}" at most twice.
- Prefer concrete, transferable techniques over subject-specific recipes.
- Keep useful instructions from the current prompt; integrate the new ones.
- Output ONLY the new system prompt. No commentary, no fences.
```

## Test surface

- `rewriter_test.py` mocks `GeminiClient.text` with canned outputs.
- Verifies the invariants are enforced post-hoc:
  - Length truncation.
  - Leading anchor injection if the model omits it.
  - Subject-mention counter (warn + log, do not rewrite — overfitting is a
    soft constraint).
