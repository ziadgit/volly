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
