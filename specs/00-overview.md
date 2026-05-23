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
  cup, smiley, sailboat, tree, heart, star. Faces and complex scenes
  are out of scope.
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

## Pitch (60 seconds)

> LLMs are famously bad at ASCII art — they can't reason spatially in a token
> stream. We're showing that by letting Gemini 3.5 Flash judge its own
> attempts visually and rewrite its own instructions, it teaches itself to
> draw in under a minute. No fine-tuning, no training data, no gradient
> updates. One model — Flash 3.5 — playing actor, vision-judge, and
> prompt-editor in parallel, with preserved thought context across
> iterations. This is Karpathy's system prompt learning, applied to a task
> nobody thought prompting could fix.
