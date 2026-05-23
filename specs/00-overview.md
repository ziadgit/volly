# 00 — Overview

## Goals

1. **Demonstrate system prompt learning** — Karpathy's "third paradigm." A
   single LLM iteratively rewrites its own system prompt from natural-language
   critique. No gradient updates, no fine-tuning, no training data.
2. **Show visible improvement in <1 minute on stage.** ASCII art is the
   vehicle: failure mode is visually obvious, no gold answer, improvement
   feels visceral.
3. **Exhibit Gemini 3.5 Flash's unique capabilities** in concert:
   - Parallel agentic reasoning (N=8 candidates concurrently)
   - Dynamic thinking levels (Low for actor, High for judge & rewriter)
   - Preserved thought context across iterations (judge remembers prior rounds)
   - Multimodality (judge sees rendered candidates as images, not text)
   - Long context (judge can see full history of attempts + prompt versions)
4. **One Flash model wears three hats** — actor, vision-judge, prompt-editor —
   in a single closed loop. The novelty.

## Non-goals

- Beating SOTA ASCII art generators (irrelevant — improvement curve is the point).
- Arbitrary subjects. A curated/sanitized list — cat, house, fish, coffee
  cup, smiley, sailboat, tree, heart, star, **capybara, owl, mushroom**.
  Faces and complex scenes are out of scope. The animal subjects
  (cat / capybara / owl / fish) are deliberately included to exercise the
  detail/shading axis of the judge rubric (spec 06); silhouettes are
  legible at the renderer's default canvas while still rewarding
  multi-character tonal shading.
- Persistent learning across runs. Each subject restarts from the seed prompt.
  (Could be added later; not for the demo.)
- Multi-model comparisons. Pure Flash-3.5 story.
- Cost optimization. Hackathon credits cover the loop comfortably.

## Personas

- **The judge in the audience** — sees a flat seed-prompt drawing, then sees
  the same model producing a recognizable subject after ~30 seconds. Their
  question to answer: *"Did the model actually learn, or did you cherry-pick?"*
  Static control arm exists for them.
- **The demo operator (us)** — types a subject from a known-safe list, hits
  Run, narrates the four panels: prompt evolving, best-of-N climbing,
  critique getting more specific.
- **The skeptic** — looks for the trick. We say: *"One model, three roles,
  preserved thoughts, vision judging, no weights changed."*

## Demo mode (rehearsal safety net)

The default run starts both arms from `SEED_PROMPT` — that is the on-stage
pitch ("watch it learn from nothing"). For rehearsal and operator-side
sanity checks, `python -m volly.loop --subject <s> --demo` swaps the
evolving arm's iteration-1 system prompt for a curated per-subject
"rehearsed" prompt (`volly.loop.DEMO_PROMPTS`) — one entry per curated
subject, each shaped like a prompt the rewriter would converge toward
after a few iterations (subject-specific structure, character set,
centering note). The control arm still starts on `SEED_PROMPT`, so the
two-arm comparison stays meaningful — control shows what a flat seed
prompt produces, and evolving shows how a known-good starting point
behaves against it. Use this when you need a predictable demo, not the
"true zero" pitch.

## Rate-limit constraints (operator-visible)

Gemini 3.5 Flash free tier is **5 requests / minute / model**. One iteration
of the default config bursts ~19 calls (16 actor + 2 judge + 1 rewriter).
That is fundamentally incompatible with free tier unless throttled. We
handle this *inside the loop*, not by changing the demo shape:

- `volly/gemini_client.py` owns a global token-bucket **RPM limiter**
  (spec 03 §"Rate limiting"). All `_generate` calls acquire a token first;
  bursting actors are queued, not refused.
- `--rpm` CLI flag and `GEMINI_RPM` env override the default. Default is
  **30** (suits a paid tier without surprises); free-tier operators set
  `--rpm 5`.
- On a 429 the client honors Gemini's `RetryInfo.retryDelay` (typically
  44–60s) instead of guessing — see spec 03.

Operator playbook:

- **Paid tier** (any positive-RPS plan; what hackathon sponsors usually
  hand out): keep defaults, iterations finish in ~10 s each.
- **Free tier**: pass `--rpm 5`. Each iteration takes ~4 minutes; do not
  attempt a live demo on free tier — pre-record and replay via the UI's
  read-only mode (see spec 10).

## Pitch (60 seconds)

> LLMs are famously bad at ASCII art — they can't reason spatially in a token
> stream. We're showing that by letting Gemini 3.5 Flash judge its own
> attempts visually and rewrite its own instructions, it teaches itself to
> draw in under a minute. No fine-tuning, no training data, no gradient
> updates. One model — Flash 3.5 — playing actor, vision-judge, and
> prompt-editor in parallel, with preserved thought context across
> iterations. This is Karpathy's system prompt learning, applied to a task
> nobody thought prompting could fix.
