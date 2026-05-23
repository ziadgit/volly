"""Run-level state, win rates, prompt diffs, JSON persistence.

Pure data: no model calls. Owns the on-disk layout under ``run_dir``,
including atomic ``state.json`` writes and consecutive evolving-arm prompt
diffs. See ``specs/08-state.md``.
"""

from __future__ import annotations

import difflib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from volly.actor import Candidate
from volly.judge import JudgeResult

Arm = Literal["evolving", "control"]

_TOP_K = 3


def win_rate(scores: list[float]) -> float:
    """Mean of the top-:data:`_TOP_K` scores (smoother than ``max``).

    Returns ``0.0`` on empty input. When fewer than ``_TOP_K`` scores
    exist, averages whatever is there.
    """
    if not scores:
        return 0.0
    top = sorted(scores, reverse=True)[:_TOP_K]
    return sum(top) / len(top)


@dataclass
class IterationRecord:
    """One arm's outcome for one iteration."""

    iter_index: int
    arm: Arm
    system_prompt: str
    candidates: list[Candidate]
    judge: JudgeResult
    best_image_path: Path
    win_rate: float

    def to_dict(self) -> dict:
        return {
            "iter_index": self.iter_index,
            "arm": self.arm,
            "system_prompt": self.system_prompt,
            "candidates": [asdict(c) for c in self.candidates],
            "judge": self.judge.model_dump(),
            "best_image_path": str(self.best_image_path),
            "win_rate": self.win_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IterationRecord:
        return cls(
            iter_index=data["iter_index"],
            arm=data["arm"],
            system_prompt=data["system_prompt"],
            candidates=[Candidate(**c) for c in data["candidates"]],
            judge=JudgeResult.model_validate(data["judge"]),
            best_image_path=Path(data["best_image_path"]),
            win_rate=data["win_rate"],
        )


@dataclass
class RunHistory:
    """All iterations of one run, interleaved across both arms."""

    subject: str
    started_at: datetime
    run_dir: Path
    seed_prompt: str
    iterations: list[IterationRecord] = field(default_factory=list)

    def add(self, record: IterationRecord) -> None:
        self.iterations.append(record)

    def latest(self, arm: Arm) -> IterationRecord | None:
        for record in reversed(self.iterations):
            if record.arm == arm:
                return record
        return None

    def win_rates(self, arm: Arm) -> list[float]:
        return [r.win_rate for r in self.iterations if r.arm == arm]

    def prompt_versions(self) -> list[str]:
        """Evolving-arm system prompts in iteration order.

        One entry per evolving-arm iteration, even if the rewriter
        produced an identical prompt — callers can dedupe via ``diff``,
        which returns an empty string for no-op revisions.
        """
        return [r.system_prompt for r in self.iterations if r.arm == "evolving"]

    def diff(self, i: int) -> str:
        """Unified diff from ``prompt_versions()[i-1]`` to ``[i]``.

        Returns an empty string when ``i`` is out of range (``i < 1`` or
        ``i >= len(versions)``) or when the two versions are identical.
        """
        versions = self.prompt_versions()
        if not (1 <= i < len(versions)):
            return ""
        prev = versions[i - 1].splitlines(keepends=True)
        curr = versions[i].splitlines(keepends=True)
        return "".join(
            difflib.unified_diff(
                prev,
                curr,
                fromfile=f"prompt v{i - 1}",
                tofile=f"prompt v{i}",
            )
        )

    def save(self) -> Path:
        """Atomically write ``state.json`` under ``run_dir``.

        Writes ``state.json.tmp`` first then ``os.replace`` — Streamlit
        reads ``state.json`` on every redraw and must never see a half-
        written file.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        target = self.run_dir / "state.json"
        tmp = self.run_dir / "state.json.tmp"
        payload = {
            "subject": self.subject,
            "started_at": self.started_at.isoformat(),
            "run_dir": str(self.run_dir),
            "seed_prompt": self.seed_prompt,
            "iterations": [r.to_dict() for r in self.iterations],
        }
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, target)
        return target

    @classmethod
    def load(cls, path: Path) -> RunHistory:
        data = json.loads(Path(path).read_text())
        return cls(
            subject=data["subject"],
            started_at=datetime.fromisoformat(data["started_at"]),
            run_dir=Path(data["run_dir"]),
            seed_prompt=data["seed_prompt"],
            iterations=[IterationRecord.from_dict(r) for r in data["iterations"]],
        )
