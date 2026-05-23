# fix_plan.md

Prioritized backlog. Ralph picks the topmost unchecked item per loop.

Format: `- [ ] P0/P1/P2 — <what> — spec: <path>`

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
- [ ] P1 — UI subject input: add free-text entry with closest-match sanitization to `CURATED_SUBJECTS` ahead of the existing dropdown — spec 10 §"Subject input" says "Free-text input sanitized to a curated subject (closest match) before the loop receives it", but `volly/ui/app.py` currently exposes only `st.selectbox(SUBJECTS)`. Sanitizer should be a pure helper (`difflib.get_close_matches` or similar) with its own test in `app_test.py`; off-list strings with no plausible match fall back to the dropdown default and surface a `st.warning` — spec: specs/10-ui.md
