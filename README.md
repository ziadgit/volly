# Volly

**ASCII art that teaches itself to draw — no fine-tuning, no training data, no gradient updates.**

Volly is a working demonstration of Karpathy's proposed third learning paradigm — *system prompt learning* — built on a single Gemini 3.5 Flash model. The model plays three roles in a closed loop: it draws ASCII art (Actor), looks at its own drawings as images and ranks them (Vision Judge), and rewrites its own instructions based on the critique (Prompt Editor). Each iteration the system gets visibly better at drawing the requested subject. No weights change.

## The thesis

LLMs are famously bad at ASCII art — they can't reason spatially in a token stream. That makes ASCII art an unusually honest demo target: failure is visually obvious to anyone, there's no single gold answer, and improvement is visceral.

The mechanism is inspired by [OpenPipe's RULER](https://openpipe.ai/blog/ruler) (LLM-as-judge giving *group-relative* rankings — better than absolute scoring because the judge only has to compare, not calibrate), but applied without any RL. RULER is a reward function for training; here we're not training. We're letting the judge's output drive *prompt rewriting* in plain English, iteration over iteration, while the model's weights stay frozen.

The result is a small, legible system that produces a curve any audience can read: a flat control arm (frozen seed prompt) vs. a climbing evolving arm (prompt rewritten each round).

## How the loop works

```
Initial system prompt: "You are an ASCII artist. Draw the requested subject."
Subject: "capybara"

For iteration in 1..N:
    1. GENERATE   — N candidates in parallel (Flash, Low thinking)
    2. RENDER     — each ASCII text → PNG via Pillow, fixed canvas, monospace
    3. JUDGE      — single multimodal call ranks all N images relative to each
                    other, returns scores + prose critique + concrete prompt-
                    edit suggestions (Flash, High thinking, sees prior iters'
                    best images + critiques in-context)
    4. REWRITE    — apply the judge's critique to produce the next iteration's
                    system prompt (Flash, High thinking)

A static control arm runs in parallel on the same subject with the seed
prompt frozen forever. Its win-rate line is the demo's epistemic baseline.
```

The whole loop is async — actor candidates fan out via `asyncio.gather`, evolving and control arms run concurrently, and the in-client RPM limiter throttles the burst to whatever your Gemini tier allows.

## Why Gemini 3.5 Flash specifically

Volly leans on four Flash 3.5 features that, taken together, make this demo possible:

- **Dynamic thinking levels** (Low / Medium / High) — the actor runs at Low for fast parallel candidate generation; the judge and rewriter run at High for careful ranking and editing. One model, asymmetric compute per role.
- **Multimodality** — the judge sees rendered candidates as images, not as text grids. Vision judgment is dramatically more reliable than text-grid judgment for spatial tasks. This is the single most important implementation detail.
- **Parallel agentic reasoning** — fan out 8 candidates per iteration in parallel, finish iterations in seconds rather than minutes.
- **1M context window + preserved thought context** — the judge sees a window of prior iterations' best images, scores, and critiques. The rewriter benefits from the judge's accumulated reasoning instead of cold-starting each round.

## Quick start

```sh
# 1. Install
uv venv
uv pip install -e ".[dev]"

# 2. Configure
echo "GEMINI_API_KEY=AIza..." > .env

# 3. Run a small first probe (paid tier)
.venv/bin/python -m volly.loop --subject capybara --iterations 4 --candidates 3 --no-control

# 4. Open the dashboard
.venv/bin/streamlit run volly/ui/app.py
```

The dashboard opens with four panels: the evolving system prompt (with diff highlighting between iterations), the current best rendered candidate, a win-rate chart with evolving vs. control series, and the judge's latest critique.

## CLI reference

```
python -m volly.loop --subject <subject> [flags]
```

| Flag | Default | Purpose |
|---|---|---|
| `--subject` | required | One of the curated subjects (case-insensitive). See list below. |
| `--iterations` | 8 | How many loop iterations to run. |
| `--candidates` | 8 | Candidates per arm per iteration. |
| `--no-control` | off | Skip the control arm (halves API cost; loses the baseline line). |
| `--rpm` | env `GEMINI_RPM` or 30 | Cap on requests per minute across actor + judge + rewriter. Set 5 for free tier. |
| `--max-retry-wait` | 90 | Per-call ceiling on honoring server `retryDelay`. Set ~3700 to patiently wait through an hourly quota reset. |
| `--resume <run-dir>` | — | Continue an existing run from its last completed iteration. Preserves `run_dir`, `state.json`, and the evolved prompt. |
| `--tier {free,paid}` | — | Convenience preset bundle. `free` → rpm=4, candidates=3, no-control, max-retry-wait=3700. `paid` → rpm=900, candidates=8, control on, max-retry-wait=90. Explicit flags override. |
| `--demo` | off | Pre-warm the evolving arm with a rehearsed per-subject prompt; control arm stays on seed. For predictable on-stage runs. |
| `--ablate-judge` | off | Rerun a text-only judge per iter and log the top-3 delta vs. the vision judge. Demo talking point. |
| `--out` | `runs/` | Run directory root (also overridable via `VOLLY_RUN_DIR`). |

### Curated subjects

`cat`, `house`, `fish`, `coffee cup`, `smiley`, `sailboat`, `tree`, `heart`, `star`, `capybara`. Faces and complex scenes are out of scope — the canvas can't render that much spatial detail legibly.

## What gets produced

Every run writes a self-contained directory:

```
runs/20260523T133000-capybara/
├── state.json                          # full RunHistory, atomically written per iter
├── iter-01/
│   ├── evolving/
│   │   ├── cand-00.png ... cand-07.png # every rendered candidate
│   │   ├── best.png                    # judge-picked best of the N
│   │   └── prompt.txt                  # system prompt used this iteration
│   └── control/
│       └── ...
├── iter-02/...
```

The UI reads `state.json` directly, so dropping a saved run into `runs/` and reloading the dashboard is the read-only replay path.

## Architecture

```
volly/
├── gemini_client.py    # async google-genai wrapper: thinking levels, JSON mode,
│                       # multimodal, global RPM token bucket, retryDelay honoring
├── actor.py            # parallel candidate generation (Low thinking)
├── renderer.py         # ASCII text → PNG via Pillow, fixed canvas, monospace
├── judge.py            # multimodal ranking with history, returns
│                       # scores + critique + prompt-edit suggestions
├── rewriter.py         # apply judge critique → new system prompt (High thinking)
├── state.py            # RunHistory, win-rate metric, prompt diffs, JSON persistence
├── loop.py             # orchestration: evolving + control arms, parallel per-iter,
│                       # resumable from state.json
└── ui/app.py           # Streamlit four-panel dashboard
```

Module-adjacent tests (`foo.py` → `foo_test.py`) cover the pipeline with stubbed clients. Run `pytest -x -q`.

## On RULER

We took the *idea* from RULER (group-relative LLM-as-judge ranking) but did not use the library, for five reasons specific to this demo:

| RULER assumes | Volly needs |
|---|---|
| Text-only candidates | Vision judging — the whole hill the project is fought on |
| You're doing RL / GRPO | No training; just prompt rewriting at inference time |
| Returns scalar rewards | Scores + prose critique + concrete prompt-edit suggestions (the rewriter needs language, not numbers) |
| Stateless per-call | Cross-iteration memory of prior best images + critiques |
| External SDK + training stack | Single-file `volly/judge.py`, ~200 lines |

The principled takeaway from RULER — *relative ranking beats absolute scoring* — is preserved. The framework is not.

## Costs and tiers

Per 8-iteration run with both arms (~152 API calls, ~155K input / ~170K output tokens):

- **Paid (Tier 1+):** ~$0.50–$1.50 per run, finishes in seconds.
- **Free tier:** Flash 3.5 free quota is 5 RPM and **20 RPD** — one iteration burns most of a day's allowance. Use `--tier free` to patient-wait through hourly/daily resets; not viable for a live demo, fine for tinkering.

A typical hackathon weekend's worth of rehearsal + demo is well under $30 on paid tier.

## Status

Hackathon-day code. The full pipeline runs end-to-end against mocked clients; the rate-limit hardening landed in the final hours and has not been extensively battle-tested against live Gemini quotas. Treat this as a working proof-of-concept, not a production tool.

See `fix_plan.md` for the active backlog and `AGENT.md` for accumulated implementation gotchas.
