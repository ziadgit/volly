"""Stitch iter best.png images side-by-side with labels into one wide PNG.

Usage:
    python scripts/image_strip.py runs/20260523T223453-capybara
    python scripts/image_strip.py runs/<...> --iters 1,3,6
    python scripts/image_strip.py --arm control --out control-strip.png
    python scripts/image_strip.py                # latest run, all iters, evolving arm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def latest_run() -> Path:
    runs = [r for r in Path("runs").glob("*") if (r / "state.json").exists()]
    if not runs:
        sys.exit("no runs found under ./runs/")
    return max(runs, key=lambda p: p.stat().st_mtime)


def load_label_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:\\Windows\\Fonts\\consola.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", help="path to runs/<...>/ (default: latest)")
    ap.add_argument("--arm", default="evolving", choices=["evolving", "control"])
    ap.add_argument("--iters", help="comma-separated iter indices, e.g. 1,3,6 (default: all)")
    ap.add_argument("--out", default="strip.png", help="output filename (default: strip.png)")
    ap.add_argument("--label-h", type=int, default=56, help="label band height in px")
    ap.add_argument("--gap", type=int, default=16, help="gap between panels in px")
    args = ap.parse_args()

    run = Path(args.run_dir) if args.run_dir else latest_run()

    if args.iters:
        wanted = [int(x.strip()) for x in args.iters.split(",") if x.strip()]
    else:
        wanted = sorted(
            int(p.name.removeprefix("iter-"))
            for p in run.glob("iter-*")
            if p.is_dir() and p.name.startswith("iter-")
        )

    panels: list[Image.Image] = []
    for n in wanted:
        best = run / f"iter-{n:02d}" / args.arm / "best.png"
        if not best.exists():
            print(f"  skip iter {n}: no best.png at {best}", file=sys.stderr)
            continue

        img = Image.open(best).convert("RGB")
        panel = Image.new("RGB", (img.width, img.height + args.label_h), "white")
        panel.paste(img, (0, args.label_h))
        draw = ImageDraw.Draw(panel)
        font = load_label_font(28)
        label = f"iter {n}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((panel.width - tw) // 2, 12), label, fill="black", font=font)
        panels.append(panel)

    if not panels:
        sys.exit("no images to stitch")

    height = max(p.height for p in panels)
    panels = [p if p.height == height else p.resize((p.width, height)) for p in panels]

    total_w = sum(p.width for p in panels) + args.gap * (len(panels) - 1)
    strip = Image.new("RGB", (total_w, height), "white")
    x = 0
    for p in panels:
        strip.paste(p, (x, 0))
        x += p.width + args.gap

    out_path = Path(args.out)
    strip.save(out_path)
    print(f"wrote {out_path}  ({strip.width}x{strip.height})  panels={len(panels)} arm={args.arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
