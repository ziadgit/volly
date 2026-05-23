# 01 — Stack

## Language: Python 3.11+

Async-native, mature multimodal SDKs, fast iteration, Streamlit for the UI.
Anything else is a distraction at a hackathon.

## Core dependencies

| Package              | Why                                                  |
| -------------------- | ---------------------------------------------------- |
| `google-generativeai`| Official Gemini SDK, async client, multimodal in     |
| `Pillow`             | ASCII text → PNG rendering (fixed-grid, monospace)   |
| `streamlit`          | Four-panel UI in minutes, hot-reload friendly        |
| `anyio`              | Structured concurrency on top of asyncio             |
| `pydantic`           | Typed schemas for judge JSON output                  |
| `python-dotenv`      | Load `GEMINI_API_KEY` from `.env` locally            |

## Dev dependencies

| Package  | Why                                       |
| -------- | ----------------------------------------- |
| `pytest` | Tests beside code (`foo.py` → `foo_test.py`) |
| `ruff`   | Lint + format in one tool                 |
| `pytest-asyncio` | Async test support                |

## Layout

Single package `volly/` with module-adjacent tests (Go-style: `foo.py` and
`foo_test.py` in the same directory). See AGENT.md for the full tree.

## pyproject.toml shape

```toml
[project]
name = "volly"
version = "0.0.1"
requires-python = ">=3.11"
dependencies = [
  "google-generativeai>=0.8",
  "Pillow>=10",
  "streamlit>=1.30",
  "anyio>=4",
  "pydantic>=2",
  "python-dotenv>=1",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "ruff>=0.5"]

[tool.pytest.ini_options]
python_files = ["*_test.py"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
```

## Environment

- `GEMINI_API_KEY` — required, read at module load by `gemini_client.py`.
- `VOLLY_MODEL` — defaults to `gemini-3.5-flash`. Overridable for ablations.
- `VOLLY_RUN_DIR` — defaults to `./runs/`. Each loop run writes a
  `runs/<timestamp>-<subject>/` directory with state.json + best-of-N PNGs.

## Why these choices

- **`google-generativeai` over LangChain/etc.** — direct path to Flash 3.5's
  thinking-level config and parallel calls. No abstraction tax at hackathon
  speed.
- **Streamlit over Next.js** — four panels in a single Python file, hot
  reload, no JS toolchain to debug under stage lights.
- **Pillow over Skia/cairo** — ASCII rendering is trivial; Pillow ships with
  a default monospace font. One dep, zero surprises.
- **`pydantic` for judge output** — judge returns JSON; we want validation
  + retries built in.
