# 08 — State

Per-run state: iteration history, win rates, prompt versions, best
candidates, persistence to disk. No model calls.

## Public surface

```python
@dataclass
class IterationRecord:
    iter_index: int                # 1-based
    arm: Literal["evolving", "control"]
    system_prompt: str
    candidates: list[Candidate]
    judge: JudgeResult
    best_image_path: Path          # iter-NN/<arm>/best.png
    win_rate: float                # mean(top-3 scores)

@dataclass
class RunHistory:
    subject: str
    started_at: datetime
    run_dir: Path
    seed_prompt: str
    iterations: list[IterationRecord]   # interleaved both arms

    def add(self, record: IterationRecord) -> None: ...
    def latest(self, arm: Literal["evolving","control"]) -> IterationRecord | None: ...
    def win_rates(self, arm: str) -> list[float]: ...
    def prompt_versions(self) -> list[str]: ...
    def diff(self, i: int) -> str: ...   # unified diff of prompt_versions[i-1] → [i]
    def save(self) -> Path: ...          # writes state.json
    @classmethod
    def load(cls, path: Path) -> "RunHistory": ...
```

## Win rate

```python
win_rate = mean(sorted(scores, reverse=True)[:3])
```

Top-3, not max — smoother curve, matches the demo claim in
`specs/02-loop.md`.

## Persistence

- `state.json` serialized via `pydantic`/`dataclass-json`. Images live as
  separate files (path references only).
- File layout under `run_dir`:

```
runs/2026-05-23T15-04-cat/
├── state.json
├── iter-01/
│   ├── evolving/
│   │   ├── cand-00.png ... cand-07.png
│   │   ├── best.png
│   │   └── prompt.txt
│   └── control/
│       ├── cand-00.png ... cand-07.png
│       └── best.png
├── iter-02/...
```

- Atomicity: write `state.json.tmp` then `os.replace` to `state.json`.
- Streamlit reads `state.json` on every redraw; the loop calls `.save()`
  at the end of every iteration (after both arms).

## Diff format

`difflib.unified_diff` between consecutive evolving-arm prompts. The UI
renders this with simple `+`/`-` line coloring.

## Test surface

- `state_test.py` constructs a synthetic `RunHistory`, runs through 3
  iterations, asserts save/load round-trip and win-rate math.
- Verifies diff output between two prompt versions.
