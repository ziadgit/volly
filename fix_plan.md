# fix_plan.md

Prioritized backlog. Ralph picks the topmost unchecked item per loop.

Format: `- [ ] P0/P1/P2 — <what> — spec: <path>`

## P0 — rate-limit hardening (discovered 2026-05-23 from a real Gemini run; demo unrunnable until these land)

- [x] P0 — Add a global async **RPM limiter** to `volly/gemini_client.py`: token bucket shared across actor/judge/rewriter so a 19-call iteration doesn't burst past the per-minute ceiling. Configurable via `--rpm` CLI flag (default), `GEMINI_RPM` env (override), constructor arg (override env). Free-tier value is **5**. Limiter is per-`GeminiClient` instance and lives in the client module — every `_generate` acquires a token before the SDK call — spec: specs/03-gemini-client.md
- [x] P0 — Honor Gemini's `RetryInfo.retryDelay` on 429. Parse the JSON error body (`details[*]["@type"] == "type.googleapis.com/google.rpc.RetryInfo"` → `retryDelay: "44s"`), sleep that long ±jitter, then retry. Cap **total** wait per call at 90s; after that, raise. Update `_RETRY_STATUSES` path in `gemini_client.py` accordingly. Falls back to existing exponential backoff if `retryDelay` is missing — spec: specs/03-gemini-client.md
- [x] P0 — Judge AND Rewriter must catch `google.genai.errors.APIError` (and subclasses `ClientError`/`ServerError`) so the loop survives any transient/quota API failure. Judge → uniform fallback `JudgeResult` with `prompt_suggestions=[]` and `critique="judge degraded: <reason>"`. Rewriter → return the unchanged `current_prompt` and log INFO `rewriter degraded: <reason>; keeping prior prompt`. Today judge only catches `ValidationError` and rewriter catches nothing — both have crashed the run against real Gemini (judge: `volly/loop.py:259`; rewriter: `volly/loop.py:442 → rewriter.py:106`). The actor's `return_exceptions=True` already handles this for its layer; do the same shape here — spec: specs/06-judge.md and specs/07-rewriter.md
- [x] P0 — Loop: when actor returns < k candidates AND no prior best exists (iteration 1 shortfall), do NOT call the judge with an empty image list — that triggers the SDK's "contents must not be empty" error path and a confusing crash. Instead, log a clear "iteration 1 produced N/k; retrying iter 1 in T seconds (RPM=R)" and re-run iteration 1 up to 2 times. After 2 retries, abort the run cleanly with a non-zero exit and a banner that says "iter 1 wedged — likely rate-limited; try --rpm=<lower> or upgrade tier" — spec: specs/02-loop.md
- [x] P0 — Surface throttling in operator-visible logs instead of a 429 wall. When the limiter holds a call, log INFO `gemini: throttled (rpm=R, queued=Q, eta≈Ts)` once per second max per caller. When a 429 retry is scheduled, log INFO `gemini: 429 retry in Xs (per server)` — NOT WARNING — to keep the noise floor low — spec: specs/03-gemini-client.md
- [x] P0 — **Patient quota handling** for sponsorship/hourly-reset scenarios. Add `--max-retry-wait N` CLI flag (default 90s, operator can set 3700s to wait through a per-hour reset). When the server's `retryDelay` exceeds `max_retry_wait`, do NOT crash — log WARNING `gemini: quota locked, eta=Xs > max_retry_wait=Ys, pausing; Ctrl-C to abort` and keep waiting in a polite once-per-30s heartbeat loop. The user explicitly wants the loop to outlast a daily-quota window rather than die — sponsorship key resets on the hour, and we want the iteration to just heal — spec: specs/03-gemini-client.md
- [x] P0 — **Resumable runs.** Add `--resume <run-dir>` flag to `volly/loop.py`. When set: load `state.json`, find the last fully-completed iteration N (both arms recorded), restore `evolving_prompt` from that iteration's `system_prompt`, restore `control_prompt = SEED_PROMPT`, and continue at iteration N+1 with the SAME `run_dir` (do not create a new timestamped dir). A run that crashed mid-iteration (state.json has iter N evolving but not control) gets that iter re-run cleanly. Critical for hourly-reset workflows: a crashed run can be picked up an hour later without losing iterations 1..N-1. Test: synthesize a 3-iter state.json, resume, assert iter 4 is the next one and prompts match — spec: specs/02-loop.md
- [x] P0 — **Tier-aware preset flags.** Add `--tier {free,paid}` to `volly/loop.py` that sets defaults: `free` → `rpm=4, candidates=3, no_control=True, max_retry_wait=3700`; `paid` → `rpm=900, candidates=8, no_control=False, max_retry_wait=90`. Explicit flags override the preset (e.g. `--tier free --candidates 4` works). When `--tier` is omitted, no preset is applied (raw argparse defaults stand — preserves existing behavior and `GEMINI_RPM` env). Operators on sponsorship/hourly-reset run `--tier free`; once they have a billed key they swap to `--tier paid` with no other changes. Update `AGENT.md` and `specs/02-loop.md` with the preset table — spec: specs/02-loop.md

## P0 — bootstrap

- [x] P0 — Scaffold `pyproject.toml` with deps (google-generativeai, Pillow, streamlit, pytest, ruff, anyio) and the `volly` package skeleton (`volly/__init__.py`) — spec: specs/01-stack.md
- [x] P0 — Implement `volly/gemini_client.py` — async wrapper for Gemini Flash 3.5 with thinking-level control (Low/Medium/High), JSON-mode helper, multimodal helper — spec: specs/03-gemini-client.md
- [x] P0 — Implement `volly/renderer.py` — ASCII text → PNG via PIL, fixed canvas, monospace font, black-on-white — spec: specs/05-renderer.md
- [x] P0 — Implement `volly/actor.py` — generate N candidates in parallel via `asyncio.gather` at Low thinking — spec: specs/04-actor.md
- [x] P0 — Implement `volly/judge.py` — multimodal call: subject + N rendered images + current prompt → ranking with per-candidate scores + critique + suggested prompt deltas — spec: specs/06-judge.md
- [x] P0 — Implement `volly/rewriter.py` — current prompt + judge critique → new prompt, at High thinking — spec: specs/07-rewriter.md
- [x] P0 — Implement `volly/state.py` — `IterationState`, `RunHistory`, win-rate metric, prompt diffs, JSON persistence — spec: specs/08-state.md
- [x] P0 — Implement `volly/loop.py` — orchestrate generate→render→judge→rewrite, run evolving arm + static control arm side-by-side — spec: specs/02-loop.md
- [x] P0 — End-to-end smoke test: `python -m volly.loop --subject cat --iterations 2` writes a valid run-history JSON and prints a best-of-8 image path — spec: specs/02-loop.md

## P1 — UI & control

- [x] P1 — Implement `volly/ui/app.py` Streamlit dashboard with the four panels (evolving prompt + diffs, current best, win-rate chart, judge reasoning) — spec: specs/10-ui.md
- [x] P1 — Wire the static control arm into the UI chart so evolving vs. flat is visually obvious — spec: specs/09-control.md
- [x] P1 — Subject sanitizer / curated list (cat, house, fish, coffee cup, smiley, sailboat, tree, heart, star) — spec: specs/00-overview.md

## P1 — capybara + detailed-animal pass (audience asked, 2026-05-23)

- [x] P1 — Add `"capybara"` to `CURATED_SUBJECTS` in `volly/loop.py` and to the curated list in `specs/00-overview.md`. Also add `"owl"` and `"mushroom"` — silhouettes that reward shaded ASCII. Update `volly/ui/app.py` selectbox source if it's not already pulling from `CURATED_SUBJECTS` — spec: specs/00-overview.md
- [x] P1 — Add `DEMO_PROMPTS["capybara"]` (and `"owl"`, `"mushroom"`) entries — each ~10 lines, mentions the subject ≤2×, suggests a shaded character palette `(. , : ; - = + * # @)` for tonal range. These are the "rehearsed" prompts the operator uses with `--demo` for predictable on-stage output — spec: specs/00-overview.md §"Demo mode"
- [x] P1 — Upgrade the judge rubric in `volly/judge.py` to weight **shading depth / texture / level of detail / character variety** alongside the existing recognizability/composition/proportions/negative-space axes. Sum-of-axes still normalizes to [0, 1]. Update the judge system prompt to mention these explicitly so its critique vocabulary feeds the rewriter — spec: specs/06-judge.md
- [x] P1 — Renderer: bump default `canvas=(1024, 768)` and lower `_MIN_FONT_SIZE` to 8 so the larger drawings the rewriter will start producing can actually fit. Update `renderer_test.py` overflow case to match the new defaults — spec: specs/05-renderer.md

## P2 — polish

- [x] P2 — Add `ralph.sh` runner with `MAX_ITERATIONS` and `RALPH_MODEL` env support — spec: specs/00-overview.md
- [x] P2 — Ablation: text-judge vs. vision-judge on the same candidates, log delta — spec: specs/06-judge.md
- [ ] P2 — Tune iteration count + candidate count based on observed plateau — spec: specs/02-loop.md (blocked-by: needs win-rate data from a real Gemini-backed run; not actionable from code alone — operator picks values after watching the curve flatten in rehearsal)
- [x] P2 — Demo-mode CLI flag that pre-warms with a known-good prompt for the rehearsed subject — spec: specs/00-overview.md

## Discovered

<!-- Ralph appends here -->
- [x] P2 — `volly` script entrypoint in `pyproject.toml` points at `volly.loop:main`; `loop.py` must expose a synchronous `main()` callable (calls `asyncio.run`) — spec: specs/02-loop.md
- [x] P0 — Swap SDK from deprecated `google-generativeai` to modern `google-genai` (legacy SDK has no `ThinkingConfig` / `thinking_level`; thinking-level control is core to the actor/judge split). Updated `pyproject.toml`, `specs/01-stack.md`, `specs/03-gemini-client.md`, `AGENT.md` — spec: specs/03-gemini-client.md
- [x] P2 — Replace `Image.getdata()` in `renderer_test.py` with a non-deprecated accessor (Pillow 14 emits `DeprecationWarning`; swap for `img.tobytes()` chunked by mode, or `img.load()` indexed access) — spec: specs/05-renderer.md
- [x] P1 — UI subject input: add free-text entry with closest-match sanitization to `CURATED_SUBJECTS` ahead of the existing dropdown — spec 10 §"Subject input" says "Free-text input sanitized to a curated subject (closest match) before the loop receives it", but `volly/ui/app.py` currently exposes only `st.selectbox(SUBJECTS)`. Sanitizer should be a pure helper (`difflib.get_close_matches` or similar) with its own test in `app_test.py`; off-list strings with no plausible match fall back to the dropdown default and surface a `st.warning` — spec: specs/10-ui.md
