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
- `--rpm N` (cap requests/minute across actor+judge+rewriter; default 30, set 5 for free tier — see spec 03 §"Rate limiting")
- `--max-retry-wait N` (per-call ceiling on honoring server `retryDelay`, in seconds; default 90; set ~3700 to wait through an hourly reset — see spec 03 §"Patient mode")
- `--resume <run-dir>` (continue an existing run from its last completed iteration; preserves `run_dir`, `state.json`, and the evolved prompt — see §"Resumable runs" below)
- `--tier {free,paid}` (preset bundle — see §"Tier presets" below)
- `--ablate-judge` (rerun a text-only judge per iter and log the top-3 delta vs. the vision judge)
- `--demo` (pre-warm the evolving arm with `DEMO_PROMPTS[subject]`; control arm
  stays on `SEED_PROMPT` so the comparison is still meaningful — see
  spec 00 §"Demo mode")
- `--out` (override `VOLLY_RUN_DIR`)

## Concurrency budget

Per iteration, with both arms on: `2 * 8` actor calls + `2` judge calls +
`1` rewriter call = **~19 API calls in parallel**. On paid tier this
finishes in ~10 s; on free tier (5 RPM) the same iteration takes ~4
minutes because the in-client limiter (spec 03) queues bursts. The
client honors Gemini's `RetryInfo.retryDelay` on 429s, so the loop never
crashes from rate limits — it just runs slower. The orchestration code
here does NOT do its own throttling; that lives entirely in
`gemini_client`.

## Iteration-1 wedge handling

Iteration 1 is uniquely fragile: there's no "prior best" to pad from when
the actor returns fewer than `k` candidates. If the actor returns 0
candidates (full free-tier wedge before the limiter warmed up), do NOT
call the judge with an empty image list — the SDK rejects that. Instead:

1. Sleep `min(60s, last_observed_retryDelay)`.
2. Re-run iteration 1, up to **2 retries**.
3. If still empty after 2 retries, exit with a clear banner:
   `iter 1 wedged — likely rate-limited; try --rpm=<lower> or upgrade tier`.

Iteration ≥ 2 with shortfall continues to pad from the prior iteration's
best, as before.

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

## Tier presets

`--tier` is a convenience bundle for the two real operational modes.
Explicit flags override the preset (e.g. `--tier free --candidates 4`).

| Flag                 | `--tier free` (sponsorship / no billing)  | `--tier paid` (Tier 1+)       |
| -------------------- | ----------------------------------------- | ----------------------------- |
| `--rpm`              | 4                                         | 900                           |
| `--candidates`       | 3                                         | 8                             |
| `--no-control`       | True (skip control arm to halve API cost) | False                         |
| `--max-retry-wait`   | 3700 (wait through hourly reset)          | 90                            |

When `--tier` is omitted, **no preset is applied** — raw argparse defaults
stand (`rpm=None` → `GEMINI_RPM` env or 30, `candidates=8`, `no_control=False`,
`max-retry-wait=90`). Unchanged invocations keep working bit-for-bit and a
shell-level `GEMINI_RPM` still wins. Operators on sponsorship/free-tier
scenarios add `--tier free` once; operators on Tier 1+ can pass `--tier paid`
to opt into the higher 900 RPM ceiling.

## Resumable runs

A run that crashes mid-loop (quota lockout, killed process, etc.) MUST be
resumable without losing work. `state.json` is the single source of
truth; on `--resume <run-dir>`:

1. Load `state.json` from the run-dir. Reject if missing or malformed.
2. Find the last iteration `N` for which BOTH arms have a complete
   `IterationRecord` (or the only arm in `--no-control` mode).
3. Set `evolving_prompt = state.iterations[N].evolving.system_prompt`
   (the prompt the rewriter produced AFTER iter N — i.e., what would
   have been used for iter N+1).
4. Set `control_prompt = SEED_PROMPT` (always).
5. Reuse the existing `run_dir` — do NOT create a new timestamped dir.
   Per-iteration files (`iter-NN/...`) for iter ≤ N stay on disk untouched.
6. Continue the main loop at iteration `N+1`.

If `state.json` has a partial iteration `N` (e.g. evolving arm completed
but control didn't), discard the partial half from in-memory state and
re-run iteration `N` cleanly. Do NOT delete the partial half from disk —
overwrite on the next save.

If `state.iterations` is empty (resume of a never-progressed run),
behave identically to a fresh run but with the existing run-dir.

## Module split

- `volly/loop.py` owns this orchestration only.
- Actor, renderer, judge, rewriter, state are imported — see their specs.
- Client wrapper is `volly/gemini_client.py` — see spec 03.
