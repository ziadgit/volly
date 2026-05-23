"""Print a timeline of prompt evolution + judge critique per iter.

Usage:
    python scripts/chain_dump.py runs/20260523T223453-capybara
    python scripts/chain_dump.py                  # auto-picks latest run
    python scripts/chain_dump.py --arm control    # control arm instead of evolving
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path


def latest_run() -> Path:
    runs = [r for r in Path("runs").glob("*") if (r / "state.json").exists()]
    if not runs:
        sys.exit("no runs found under ./runs/")
    return max(runs, key=lambda p: p.stat().st_mtime)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", help="path to runs/<...>/ (default: latest)")
    ap.add_argument("--arm", default="evolving", choices=["evolving", "control"])
    args = ap.parse_args()

    run = Path(args.run_dir) if args.run_dir else latest_run()
    state = json.loads((run / "state.json").read_text())

    iters = [it for it in state.get("iterations", []) if it.get("arm") == args.arm]

    print(f"# Volly chain dump")
    print(f"run:     {run}")
    print(f"subject: {state.get('subject', '?')}")
    print(f"arm:     {args.arm}")
    print(f"iters:   {len(iters)}")
    print()

    prior_prompt = ""
    for it in iters:
        n = it.get("iter_index", 0)
        prompt = it.get("system_prompt", "")
        judge = it.get("judge") or {}
        critique = (judge.get("critique") or "").strip()
        suggestions = judge.get("prompt_suggestions") or []
        win = it.get("win_rate")
        best = run / f"iter-{n:02d}" / args.arm / "best.png"

        print("=" * 72)
        print(f"ITER {n}    win_rate={win}")
        print("=" * 72)
        print(f"best image: {best}")
        print()
        print("--- prompt used this iter ---")
        print(prompt)
        print()
        print("--- judge critique ---")
        snippet = critique if len(critique) <= 600 else critique[:600] + "..."
        print(snippet or "(none)")
        print()
        if suggestions:
            print("--- judge prompt_suggestions ---")
            for s in suggestions:
                print(f"  - {s}")
            print()
        if n > 1:
            if prior_prompt == prompt:
                print("--- diff vs iter " f"{n-1}" " ---")
                print("(prompt unchanged — rewriter degraded or no-op)")
            else:
                print(f"--- diff vs iter {n-1} ---")
                for line in difflib.unified_diff(
                    prior_prompt.splitlines(),
                    prompt.splitlines(),
                    fromfile=f"iter-{n-1:02d}",
                    tofile=f"iter-{n:02d}",
                    lineterm="",
                    n=1,
                ):
                    print(line)
            print()
        prior_prompt = prompt

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
