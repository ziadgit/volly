# 02 — Loop

The orchestration layer. Owns the iteration cycle and the two-arm
(evolving vs. static control) comparison.

## Inputs

- `subject: str` — user-chosen subject from the curated list
- `iterations: int` — default 8, configurable
- `candidates_per_iter: int` — default 8
- `seed_prompt: str` — `SEED_PROMPT` constant: `"You are an ASCII artist.
  Draw the requested subject."`

## Outputs

- `runs/<timestamp>-<subject>/state.json` — full `RunHistory`
- `runs/<timestamp>-<subject>/iter-NN/cand-MM.png` — every rendered candidate
- `runs/<timestamp>-<subject>/iter-NN/best.png` — judge-selected best
- `runs/<timestamp>-<subject>/iter-NN/prompt.txt` — system prompt used

## Algorithm

```
seed_prompt = SEED_PROMPT
evolving_prompt = seed_prompt
control_prompt  = seed_prompt   # never mutates

for iter in 1..N:
    # Evolving arm
    cands_e   = actor.generate(evolving_prompt, subject, k=8, thinking=Low)
    images_e  = [renderer.render(c) for c in cands_e]
    ranked_e  = judge.rank(subject, evolving_prompt, images_e, history=run.history)
    evolving_prompt = rewriter.update(evolving_prompt, ranked_e.critique, thinking=High)

    # Control arm (parallel)
    cands_c   = actor.generate(control_prompt, subject, k=8, thinking=Low)
    images_c  = [renderer.render(c) for c in cands_c]
    ranked_c  = judge.rank(subject, control_prompt, images_c, history=run.history)
    # control_prompt is NOT updated

    state.record(iter, evolving=ranked_e, control=ranked_c, new_prompt=evolving_prompt)
```

Evolving and control arms run **in parallel** via `asyncio.gather`, so each
iteration takes ~max(evolving, control), not sum.

## Win-rate metric

```
win_rate(iteration) = mean(top_3_judge_scores)
```

Smoother than `max(scores)`, which jumps. Both arms publish their own
`win_rate` per iteration. Demo chart shows two lines.

## CLI

```
python -m volly.loop --subject cat --iterations 8 --candidates 8
```

Flags:
- `--subject` (required, validated against curated list)
- `--iterations` (default 8)
- `--candidates` (default 8)
- `--no-control` (skip the control arm, half the API spend)
- `--ablate-judge` (rerun a text-only judge per iter and log the top-3 delta vs. the vision judge)
- `--demo` (pre-warm the evolving arm with `DEMO_PROMPTS[subject]`; control arm
  stays on `SEED_PROMPT` so the comparison is still meaningful — see
  spec 00 §"Demo mode")
- `--out` (override `VOLLY_RUN_DIR`)

## Concurrency budget

Per iteration, with both arms on: `2 * 8` actor calls + `2` judge calls +
`1` rewriter call = ~19 API calls in parallel. Fits Flash 3.5 quota
comfortably. If we hit a rate limit, the failure mode is a 429 — wrap each
call in a `tenacity`-style retry with jittered backoff inside the
`gemini_client` module, not here.

## Plateau

We expect a plateau after ~6 iterations. The loop does not stop early —
predictable timing matters more than saved tokens on stage.

## Failure handling

- Judge returns malformed JSON twice in a row → fall back to text-only
  judge for that iteration only, log the fallback, continue.
- Actor produces fewer than `k` parseable candidates → pad with the prior
  iteration's best.
- Any unhandled exception → still write `state.json` to disk so the UI can
  render up to the failure point.

## Module split

- `volly/loop.py` owns this orchestration only.
- Actor, renderer, judge, rewriter, state are imported — see their specs.
- Client wrapper is `volly/gemini_client.py` — see spec 03.
