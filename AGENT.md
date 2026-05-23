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
- Pillow 14 deprecates `Image.Image.getdata`; renderer tests still use
  it (warns, doesn't fail). Tracked in fix_plan.md under P2.
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
