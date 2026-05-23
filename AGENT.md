# AGENT.md

## Project

**Volly** — self-improving ASCII art via system prompt learning. One Gemini
3.5 Flash model plays actor + vision-judge + prompt-editor in a loop,
rewriting its own system prompt from natural-language critique. Karpathy's
"third paradigm," applied to a task LLMs are famously bad at.

## Stack

- **Language:** Python 3.11+
- **LLM:** Gemini 3.5 Flash (`gemini-3.5-flash`) via `google-genai`
  - Low thinking → actor (fast parallel candidate generation)
  - High thinking → judge & rewriter (careful ranking and editing)
- **Async:** `asyncio` — all model calls are parallel via `asyncio.gather`
- **Rendering:** Pillow (`PIL`) — ASCII text → PNG with monospace font
- **UI:** Streamlit — four-panel dashboard
- **Test/lint:** `pytest`, `ruff`

## Bootstrap (iteration 1 only — if dependencies missing)

```sh
# uv is preferred; fall back to venv + pip if uv isn't available
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Required env var
export GEMINI_API_KEY=...   # the user's hackathon key
```

If `pyproject.toml` does not exist yet, the first P0 fix_plan item is to
scaffold it. Do not pip-install anything before the toolchain is in place.

## Commands

| Purpose       | Command                                  |
| ------------- | ---------------------------------------- |
| dev (UI)      | `streamlit run volly/ui/app.py`          |
| dev (CLI)     | `python -m volly.loop --subject cat`     |
| typecheck/lint| `ruff check .`                           |
| test          | `pytest -x -q`                           |
| build sanity  | `python -c "import volly"`               |

## Layout (target)

```
volly/
├── volly/                         # Python package
│   ├── __init__.py
│   ├── gemini_client.py           # async Gemini Flash 3.5 wrapper
│   ├── gemini_client_test.py
│   ├── actor.py                   # parallel candidate generation
│   ├── actor_test.py
│   ├── renderer.py                # ASCII → PNG via PIL
│   ├── renderer_test.py
│   ├── judge.py                   # multimodal ranking
│   ├── judge_test.py
│   ├── rewriter.py                # apply critique → new system prompt
│   ├── rewriter_test.py
│   ├── state.py                   # IterationState, win rates, history
│   ├── state_test.py
│   ├── loop.py                    # orchestration (evolving + control)
│   ├── loop_test.py
│   └── ui/
│       └── app.py                 # Streamlit four-panel dashboard
├── specs/                         # source of truth for each surface
├── tests/                         # cross-module integration tests (if any)
├── pyproject.toml
├── PROMPT.md
├── AGENT.md
├── fix_plan.md
└── ralph.sh
```

## Gotchas

<!-- Ralph: append discoveries here, oldest at top. Never rewrite. -->
- The project venv lives at `.venv/` (created via `uv venv`). Use
  `.venv/bin/ruff`, `.venv/bin/pytest`, `.venv/bin/python` from the project
  root — no `source` needed. `uv pip install -e ".[dev]"` is the bootstrap.
- `pytest -x -q` returns exit code 5 (no tests collected) until the first
  `*_test.py` lands. Treat 5 as a pass during the scaffold-era loops; once
  any test file exists, 5 means real breakage.
- SDK is `google-genai` (modern, package import `google.genai`), NOT
  `google-generativeai` (deprecated, no `ThinkingConfig`). The original
  spec named the legacy package — corrected because the deprecated SDK
  lacks `thinking_budget` / `thinking_level` and we need per-call
  Low/Medium/High control for the actor vs. judge split. Async entry
  point is `client.aio.models.generate_content(model, contents, config)`.
- `google-genai` 2.6.0: `types.ThinkingConfig(thinking_level=...)` accepts
  the snake_case kwarg (pydantic alias). `GenerateContentResponse.parsed`
  is populated when both `response_mime_type="application/json"` AND
  `response_schema=<pydantic class>` are set — fall back to
  `schema.model_validate_json(response.text)` if `.parsed` is `None`.
  `APIError.code` is the HTTP status; retry only 429/500/503.
- `volly.renderer` locates a monospace font by scanning a small list of
  well-known paths (DejaVuSansMono on Linux, Menlo.ttc on macOS, Consolas
  on Windows). If none exist it tries `matplotlib.font_manager` (soft
  dep, not in `pyproject`), then falls back to `ImageFont.load_default()`
  — which is proportional, so grid alignment degrades. On this Mac,
  `/System/Library/Fonts/Menlo.ttc` is the resolved font.
- Pillow 12.2 deprecates `Image.Image.getdata` (removal in Pillow 14,
  2027-10-15). The drop-in replacement is `img.get_flattened_data()` —
  same shape for RGB (tuple of `(r, g, b)` tuples, length `w*h`). Used
  throughout `renderer_test.py`. The deprecation message itself names
  `get_flattened_data` as the migration target.
- `volly.actor.generate` returns a list whose length may be `< k` when
  some `GeminiClient.text` calls raise — `return_exceptions=True` on
  `asyncio.gather` swallows them by design. Each surviving `Candidate`
  carries its original dispatch `index` (0..k-1), so an index gap means
  that slot failed. The loop is expected to pad from prior best.
- `volly.judge.rank` never raises on a malformed model response — on
  persistent `ValidationError` from `GeminiClient.json` it logs a warning
  and returns a uniform fallback `JudgeResult` (all scores `0.5`, empty
  `prompt_suggestions`). The loop should read a flat-0.5 iteration as a
  judge degradation signal, not a real plateau. `HistoryEntry` is defined
  in `judge.py` (not `state.py`) so the judge stays independent of state;
  `volly.state.RunHistory` will project its records into `HistoryEntry`
  when wired.
- `volly.rewriter.rewrite` enforces the spec invariants post-hoc on
  whatever the model returns: strips surrounding whitespace, injects the
  "You are an ASCII artist." anchor when missing (logs INFO), hard-
  truncates to 4000 chars (logs WARNING), and counts subject mentions
  case-insensitively with word boundaries — a count over 2 logs a
  WARNING but does NOT rewrite. Word-boundary regex means "cat" matches
  "cat"/"Cat"/"CAT" but not "catalog", and multi-word subjects like
  "coffee cup" work the same way.
- `volly.loop.run` orchestrates evolving + control arms in parallel via
  `asyncio.gather` per iteration, then runs the rewriter sequentially on
  the evolving arm's `JudgeResult`. Run-dir layout is
  `iter-NN/<arm>/{cand-MM.png, best.png, prompt.txt}` plus `state.json`
  at the run-dir root, persisted atomically after every iteration. CLI
  subject is normalized + validated against `CURATED_SUBJECTS`; off-list
  subjects exit 2. When the actor returns < k candidates, missing slots
  are padded from the prior iteration's best `Candidate` for that arm —
  iteration 1 with shortfall just records a shorter list. Judge history
  is per-arm and capped at the last 4 iterations.
- Cross-module integration tests live under `tests/` (configured via
  `testpaths = ["volly", "tests"]` in `pyproject.toml`); `tests/smoke_test.py`
  drives `loop.main` in-process with a stubbed `GeminiClient` (patched as
  `volly.loop.GeminiClient` since `main` constructs it inside `run`). No
  network, no API key required — covers the full CLI → state.json →
  printed best-image-path pipeline for both `--no-control` and the default
  evolving+control configuration.
- `volly.ui.app` is the Streamlit four-panel dashboard. Run with
  `.venv/bin/streamlit run volly/ui/app.py`. It picks the most-recently-
  modified subdir under `VOLLY_RUN_DIR` (default `./runs/`) that contains
  `state.json` — so dropping a fresh `runs/...` dir on disk and reloading
  is the read-only smoke test path. The Run button spawns
  `asyncio.run(loop.run(config))` inside a module-cached single-worker
  `ThreadPoolExecutor` (`@st.cache_resource`), stores the `Future` in
  `st.session_state["future"]`, and polls every `POLL_SECONDS` (1s) via
  `time.sleep` + `st.rerun()` until done. Stop is best-effort:
  `future.cancel()` is a no-op once the worker has picked the task up,
  but `state.json` is already persisted per-iteration by `loop.run`, so
  partial state is on disk regardless. UI is intentionally not unit-
  tested per spec; only the pure helpers (`latest_run_dir`,
  `load_history`, `winrate_chart_data`) have tests in `app_test.py`.
  `winrate_chart_data` returns an empty dict when no iterations exist
  (caller shows "waiting"), omits the `"control"` key entirely on
  `--no-control` runs, and pads the shorter arm with `None` so
  `st.line_chart` accepts the dict.
- `volly.ui.app` Panel 3 (win-rate) uses `st.altair_chart`, not
  `st.line_chart` — Streamlit 1.57's `st.line_chart` has no `y_min`/
  `y_max`, and spec 10-ui requires y-axis [0, 1] so a flat control noise
  band visually reads flat. `winrate_chart_data` returns LONG-form
  records `[{iteration, arm, win_rate}, ...]` (not the prior wide-form
  dict — altair handles ragged series natively, so no `None` padding is
  needed). `_winrate_chart` builds the spec with explicit `ARM_COLORS`
  (evolving #16a34a green, control #9ca3af gray) so the demo's
  "learning vs. baseline" story is legible without a legend lookup; the
  color-scale domain is filtered to arms actually present, so a
  `--no-control` run shows no phantom legend entry. `altair` and
  `pandas` are not in `pyproject.toml` — they ship as transitive deps
  of `streamlit`.
- `volly.judge.rank(..., include_images=False, texts=[...])` is the
  text-judge ablation mode (spec 06 §Ablation hook). `texts` is required
  and its length must match `len(images)` — the loop passes the same
  rendered candidates so vision-judge and text-judge agree on `n`. In
  text mode no images are attached (neither candidates nor prior bests),
  the system prompt swaps "candidate images" for "candidate drawings as
  raw ASCII text", and the user message inlines each ASCII block between
  `~~~` fences. History critiques are still included as text.
  `volly.loop --ablate-judge` enables it: after vision-judge, the loop
  reruns text-judge on the same candidates per arm and logs
  `ablation iter N arm X: vision_top3=A text_top3=B delta=±C` at INFO.
  Text-judge failures are caught and logged via `_log.exception` so the
  live demo never dies on ablation noise.
- `volly.state.RunHistory.prompt_versions` returns the evolving-arm
  `system_prompt` for every evolving iteration in order, with no
  dedup — iteration N's prompt is what the rewriter produced from
  iteration N-1 (or the seed for N=1). `diff(i)` returns the unified
  diff from versions[i-1] to versions[i] with headers `prompt v{i-1}`
  / `prompt v{i}`; `i < 1` or `i >= len(versions)` returns "", and so
  do consecutive identical prompts. `save()` writes `state.json.tmp`
  under `run_dir` then `os.replace` to `state.json` so Streamlit (which
  re-reads on every redraw) never observes a half-written file. Scores
  flow through `judge.CandidateScore` so the [0.0, 1.0] range is
  validated at construction; tests that synthesize scores must clamp.
- `volly.loop --demo` pre-warms the evolving arm's iteration-1 system
  prompt with `DEMO_PROMPTS[subject]` (one rehearsed prompt per
  `CURATED_SUBJECTS` entry, hard-coded in `loop.py`). Control stays on
  `SEED_PROMPT` regardless — the comparison would be meaningless if both
  arms started ahead. CLI flag is `--demo`, `LoopConfig.demo_mode: bool`
  is the runtime knob; `RunHistory.seed_prompt` is still `SEED_PROMPT`
  in both modes (it represents the canonical baseline, not the evolving
  arm's actual starting point). Test invariant
  `test_demo_prompts_cover_every_curated_subject` guards against
  forgetting a `DEMO_PROMPTS` entry when adding a subject — the loop
  would otherwise `KeyError` at iteration 1 in demo mode.
- `volly.ui.app.sanitize_subject(text, curated=CURATED_SUBJECTS)` is the
  pure free-text → curated-subject helper backing the header text input
  (spec 10 §"Subject input"). It strips/lowercases, returns the exact
  match if present, else `difflib.get_close_matches(..., n=1,
  cutoff=0.6)` — empty/whitespace and unrelated nouns ("airplane",
  "dragon") return `None`. The cutoff was tuned against realistic typos:
  "sailbot"→"sailboat", "coffe cup"/"coffeecup"→"coffee cup",
  "heeart"→"heart", "tre"→"tree". Note "startup"→"star" is an accepted
  false positive — the dropdown is always present as explicit override.
  `_resolve_subject(free_text, dropdown_value)` returns
  `(subject, notice)` where `notice` is `"warn:..."` for
  fallback-to-dropdown, `"info:..."` for a sanitization caption, or
  `None` when canonical/empty. The header parses the prefix and dispatches
  to `st.warning` or `st.caption` accordingly.
- **Rate-limit reality (real Gemini run on 2026-05-23):** Free tier Flash
  3.5 is **5 RPM/model**. Default loop iter = 19 parallel calls → wall of
  429s, current retry sleeps 1/2/4s while Gemini asks for 44s, judge
  exhausts retries and crashes the whole run. The existing
  `_sleep_backoff` is per-call exponential backoff — there is no global
  RPM limiter and no honoring of `RetryInfo.retryDelay`. See the new P0
  "rate-limit hardening" section at the top of `fix_plan.md`. Concrete
  trace: `volly/loop.py:259 → judge.rank → gemini_client.json →
  google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED`. Judge's
  `except ValidationError` block does NOT catch `APIError`, so the run
  dies instead of degrading.
- **Free tier on Flash 3.5 has TWO ceilings, both via `FreeTier` quotas:**
  per-minute (`GenerateRequestsPerMinutePerProjectPerModel-FreeTier`,
  quotaValue 5) and per-day (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`,
  quotaValue 20). The per-day ceiling is the killer — a single failed run
  burns it. Resets at UTC midnight by default; sponsorship/credit setups
  sometimes reset hourly. Diagnostic: the `quotaId` string in the 429
  response body literally contains `FreeTier` vs. (paid tier has no such
  marker). The dashboard at https://aistudio.google.com/ shows the
  currently-selected project's tier; if the dashboard says Tier 1 (1K RPM
  / 10K RPD on Flash 3.5) but API errors say `FreeTier`, the API key in
  `.env` was created against a *different* (free-tier) project than the
  one currently selected in AI Studio.
- **Operational modes are bundled in `--tier`** (after fix_plan P0
  "Tier-aware preset flags" lands): `--tier free` for sponsorship/no-
  billing/hourly-reset scenarios (rpm=4, candidates=3, --no-control,
  max-retry-wait=3700 so the loop sits through an hourly reset);
  `--tier paid` for Tier 1+ (rpm=900, candidates=8, control on,
  max-retry-wait=90). Explicit flags override the preset. `--resume
  <run-dir>` continues a crashed/quota-blocked run from `state.json`'s
  last fully-completed iteration without losing the evolved prompt or
  per-iter PNGs — critical for the "iterate, get quota-locked, wait an
  hour, pick up where you left off" workflow.
- `volly.gemini_client._RpmLimiter` is a per-instance token bucket
  (capacity = `rpm`, refill = `rpm/60` tokens/sec) acquired inside
  `_generate` ONCE PER ATTEMPT — a 429/503 transport retry re-acquires.
  Resolution at construction: explicit `rpm=` arg → `GEMINI_RPM` env →
  default `30`. CLI plumbing is `--rpm N` → `LoopConfig.rpm` →
  `GeminiClient(rpm=...)`. Limiter exposes `client.rpm` (read-only) for
  log lines. FIFO ordering comes from a single `asyncio.Lock` held
  across the refill+wait loop, so callers serialize cleanly even at
  rpm=5 (one token / 12s). Limiter accepts injectable `clock=` and
  `sleep=` kwargs purely for tests — production code should never pass
  them. Tests that construct `GeminiClient` instances must
  `monkeypatch.delenv("GEMINI_RPM", raising=False)` (there's an autouse
  fixture in `gemini_client_test.py` that does this for the whole
  module) or a developer's shell value will leak into the test. Throttle
  logs and the operator-configurable `max_retry_wait_s` / patient mode
  are still separate P0 items in `fix_plan.md`.
- `volly.gemini_client._parse_retry_delay(exc)` extracts the
  server-supplied `RetryInfo.retryDelay` (e.g. `"44s"` or `"44.5s"`) from a
  `google.genai.errors.APIError`. The error body sits on
  `exc.details["error"]["details"]`, a list — RetryInfo is the entry whose
  `@type` ends `/google.rpc.RetryInfo`. Returns `None` on any structural
  miss (unstructured `response_json`, no `error` dict, no `details` list,
  no RetryInfo entry, missing/malformed `retryDelay`). When `_generate`
  hits a 429 it asks the parser first; on a hit it sleeps
  `retryDelay + _retry_delay_jitter()` (jitter ∈ [0, 2)s) via
  `_sleep_retry_delay`, and tracks cumulative wait across retries against
  `_MAX_RETRY_WAIT_S` (90s). If the next planned wait would push the
  cumulative over the cap the bare `APIError` is re-raised — callers see
  the same exception they would have without the retry, so judge/rewriter
  fallback paths can degrade gracefully. Tests monkeypatch
  `_retry_delay_jitter` (sync, returns float) and `_sleep_retry_delay`
  (async, takes seconds) at module level; both are deliberate seams.
  Falls back to `_sleep_backoff` when RetryInfo is absent — the existing
  `test_generate_raises_after_max_transport_attempts` still exercises
  that path (its 429s carry no RetryInfo).
- `volly.loop` handles the **iteration-1 wedge** (zero candidates from any
  arm) by retrying iter 1 up to 2 times before aborting. Spec 02
  §"Iteration-1 wedge handling" is the source of truth. Retry trigger:
  ANY arm in iter 1 produces an empty candidate list (iter ≥ 2 still pads
  from prior best — only iter 1 has no fallback). Logic lives inline in
  `run()` as a `while True` inside the per-iteration `for`; the task list
  is rebuilt each attempt by the module-level `_build_arm_tasks` helper
  (extracted to avoid ruff B023 closure-over-loop-variable). Sleep between
  attempts is `_ITER_ONE_RETRY_SLEEP_S` (60s) via `_iter_one_retry_sleep`
  — module-level async seam, tests monkeypatch it to skip the wait. After
  `_ITER_ONE_MAX_RETRIES + 1` (= 3) attempts with any arm still empty,
  raises `IterationOneWedgedError` (subclass of `RuntimeError`); `main`
  catches it, prints `_ITER_ONE_WEDGED_BANNER` to stderr, and exits **rc=3**.
  Partial shortfall (1..k-1 cands) does NOT retry — the judge still ranks
  what we have. The `run()` `except Exception:` block re-saves `state.json`
  on the wedge path, so a wedged run leaves an empty-iterations state.json
  on disk plus whatever iter-01/ artifacts the final failed attempt wrote.
- `volly.gemini_client` emits operator-visible throttle/retry logs at INFO
  (NOT WARNING — rate limiting is normal, the noise floor matters for the
  demo). Two log lines, both formatted with `_log = logging.getLogger(
  __name__)`: (1) `gemini: throttled rpm=R queued=Q eta≈Ts` from inside
  `_RpmLimiter.acquire` on any iteration that has to sleep waiting for a
  token; squelched to once per `_THROTTLE_LOG_SQUELCH_S` (=1.0) per limiter
  instance via `_last_throttle_log_at` initialized to `-math.inf` so the
  first sleeping acquire always logs. `queued` is a self-inclusive counter
  (`_queued` incremented on acquire entry, decremented on exit), so a
  solo throttled caller still reads `queued=1`. (2) `gemini: 429 retry in
  Xs (server)` from `_generate` when a server `RetryInfo.retryDelay` is
  honored; logged before `_sleep_retry_delay` so an operator sees the
  wait coming. The exponential-backoff fallback (`_sleep_backoff`,
  triggered when no RetryInfo is present) deliberately does NOT log —
  it's a sub-second sleep and would just be noise. Patient-mode entries
  (single retryDelay > `max_retry_wait_s`) deliberately do NOT emit the
  INFO line either — they use the WARNING `quota locked` line instead
  (`test_generate_logs_quota_locked_warning_in_patient_mode` pins both:
  the WARNING fires once, the INFO `429 retry in` never does). Throttle
  squelching is tested with a `_FakeClock` that
  honestly ticks: rpm=120 → 0.5s/token, three back-to-back sleeping
  acquires at clock=0/0.5/1.0 produce exactly two log lines.
- **Judge + rewriter both swallow `google.genai.errors.APIError`** (and
  its subclasses `ClientError`/`ServerError`, by `except APIError`) so
  the loop survives quota/transient failures the client retry could not
  paper over. Judge path: `_api_fallback_result(n, str(exc))` returns
  uniform 0.5 scores, `prompt_suggestions=[]`, and `critique="judge
  degraded: <reason>"` (distinct critique from the existing
  ValidationError fallback so an operator can tell the two failure modes
  apart in `state.json`); logs WARNING `judge degraded on APIError: ...`.
  Rewriter path: returns `current_prompt` UNCHANGED — invariants are NOT
  re-enforced because we're keeping the prior prompt, which already
  passed them; logs INFO (not WARNING — keeps demo noise floor low)
  `rewriter degraded: <reason>; keeping prior prompt`. Tests construct
  `genai_errors.APIError(code=429, response_json={"error": {"message":
  "..."}})` directly; the bare-class case covers all subclasses.
- `volly.loop --resume <run-dir>` (spec 02 §"Resumable runs") loads
  `state.json` from the run-dir, computes the last fully-completed iter N via
  `_last_complete_iter(history, expects_control=not config.no_control)`,
  truncates any partial half from in-memory `history.iterations` (the
  on-disk PNGs stay — they'll be overwritten on next save), restores
  `evolving_prompt` from iter N's recorded `system_prompt`, and continues
  at iter N+1 inside the SAME `run_dir` (no new timestamped child). When
  N==0 (state.json present but no complete iters), behavior matches a fresh
  run but reuses `run_dir`, including `demo_mode` pre-warming. Missing or
  malformed `state.json` raises `ValueError` → `main` rc=2.
  `--subject` becomes optional once `--resume` is set; if passed anyway it
  must match state.json's subject (else `ValueError` → rc=2). Subject is
  always loaded from state.json on resume (never from CLI). `--out` is
  ignored on resume — the spec is explicit that the run_dir is the resume
  argument. Edge: a partial iter N where evolving completed but rewriter
  also already ran — the rewriter's output for N+1 was never persisted, so
  resume re-uses iter N's prompt for iter N+1 (one rewriter step is "lost"
  — acceptable per spec). Tests cover: 3-iter resume continuing to iter 4
  with iter 3's prompt threading through, run_dir reuse (no timestamped
  child added), partial-half discard + clean re-run of iter N, empty-
  iterations resume = fresh iter 1, missing/malformed state.json →
  ValueError, CLI `--subject` optional + mismatch detection, and
  `_last_complete_iter` arm-set semantics in both modes.
- `volly.gemini_client` **patient mode** keeps the loop alive through long
  quota windows (sponsorship-key hourly resets, free-tier daily resets at
  UTC midnight). Trigger: a single server-supplied `retryDelay` exceeds the
  per-client `max_retry_wait_s` cap (default 90s, override via
  `GeminiClient(max_retry_wait_s=...)` or `--max-retry-wait N` on the loop
  CLI). Behavior in `_generate`: log WARNING `gemini: quota locked, eta=Xs
  > max_retry_wait=Ys, pausing; Ctrl-C to abort`, call `_patient_sleep(
  retryDelay + jitter)` which sleeps in 30s chunks and emits INFO `gemini:
  still waiting, eta≈Xs` after each non-final chunk, then `continue` the
  retry loop — **bypassing both the `_MAX_TRANSPORT_ATTEMPTS=3` cap and the
  cumulative-wait cap**. The unbounded patient retry only exits on a
  non-429 response or KeyboardInterrupt; back-to-back quota locks just keep
  waiting. Normal-path 429s (where `retryDelay ≤ max_retry_wait_s`) still
  use the existing `_sleep_retry_delay` + cumulative-cap path and the INFO
  `429 retry in Xs (server)` line. Operators on `--tier free`-style flows
  will run `--max-retry-wait 3700` so a server saying "wait 60 minutes for
  the hourly reset" pauses politely instead of crashing. `_patient_sleep`
  is a module-level seam — tests monkeypatch the whole helper (skipping
  the heartbeat) when exercising `_generate`; the heartbeat itself is
  covered by direct `_patient_sleep` tests that monkeypatch
  `asyncio.sleep`. Heartbeat constant is `_PATIENT_HEARTBEAT_S=30.0`; a
  sleep that lands exactly on a chunk boundary (e.g. 30s, 60s) emits no
  trailing heartbeat because `remaining > 0` is the gate.
- **Curated subjects: shading-palette convention.** The "detailed-animal" set
  (`capybara`, `owl`, `mushroom`) all use the same tonal palette
  `. , : ; - = + * # @` (light→dark) in their `DEMO_PROMPTS` entries —
  deliberate so the rewriter sees consistent palette vocabulary across
  shaded subjects and the (pending) judge-rubric upgrade can score
  "texture / character variety" against a stable palette. When adding a
  new silhouette/shaded subject, reuse this palette verbatim in the
  rehearsed prompt; the lighter punctuation-only set used by `cat`/
  `heart`/`star` is the right choice for line-art subjects. Either way,
  the two test invariants `test_curated_subjects_match_overview_list` and
  `test_demo_prompts_cover_every_curated_subject` lock CURATED_SUBJECTS
  and DEMO_PROMPTS keys to the same set — adding to one without the other
  fails the suite, so the two `fix_plan.md` items must land together.
  `volly.ui.app.SUBJECTS = tuple(sorted(CURATED_SUBJECTS))` is computed
  at import, so the Streamlit dropdown picks up new subjects automatically
  with no UI edit.
- `volly.loop --tier {free,paid}` is the operational-mode preset bundle
  from spec 02 §"Tier presets". Preset dict `_TIER_PRESETS` lives at
  module top: `free` = `{rpm: 4, candidates: 3, no_control: True,
  max_retry_wait_s: 3700.0}`, `paid` = `{rpm: 900, candidates: 8,
  no_control: False, max_retry_wait_s: 90.0}`. Resolution lives in
  `_resolve_loop_config(args)` (called from `main` after `_parse_args`):
  explicit flags (anything non-None at the parser level) always win; if
  `--tier` selects a preset, unset flags fall back to the preset value;
  if `--tier` is omitted (`args.tier is None`), unset flags fall back to
  the legacy raw defaults (rpm=None→env/30, candidates=8,
  no_control=False, max_retry_wait_s=90.0) — preserving pre-tier
  behavior and `GEMINI_RPM` env. The four tier-controlled flags
  (`--candidates`, `--no-control`, `--rpm`, `--max-retry-wait`) all
  parse to `None` when unset so the resolver can tell "unspecified" from
  "explicitly set"; `--no-control` keeps `action="store_true"` so
  passing it still yields `True` (just with `default=None`). Spec deviates
  from fix_plan's "default preset is paid" claim because paid's rpm=900
  would silently bypass `GEMINI_RPM` env — keeping the no-tier path
  pass-through is the cleanest backward-compat story. `--tier paid`
  remains the explicit opt-in for the 900 RPM ceiling.
