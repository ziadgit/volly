"""Renderer — ASCII text → PNG via PIL, fixed canvas, monospace font.

See ``specs/05-renderer.md``. The judge must see images, not text — this
module exists to make that possible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES: tuple[str, ...] = (
    "/Library/Fonts/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "C:\\Windows\\Fonts\\consola.ttf",
    "C:\\Windows\\Fonts\\CascadiaMono.ttf",
)

_MIN_FONT_SIZE = 6


def _find_font_path() -> str | None:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return path
    try:
        from matplotlib import font_manager

        for name in ("DejaVu Sans Mono", "Menlo", "Consolas"):
            try:
                found = font_manager.findfont(
                    font_manager.FontProperties(family=name),
                    fallback_to_default=False,
                )
            except Exception:
                continue
            if found and Path(found).exists():
                return found
    except Exception:
        pass
    return None


_FONT_PATH = _find_font_path()
_FONT_CACHE: dict[int, Any] = {}


def _load_font(size: int) -> Any:
    cached = _FONT_CACHE.get(size)
    if cached is not None:
        return cached
    if _FONT_PATH:
        font = ImageFont.truetype(_FONT_PATH, size=size)
    else:
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:
            font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _measure_cell(font: Any) -> tuple[int, int]:
    # "M" is a stable representative for monospace cell width.
    bbox = font.getbbox("M")
    cw = bbox[2] - bbox[0]
    try:
        ascent, descent = font.getmetrics()
        ch = ascent + descent
    except AttributeError:
        ch = bbox[3] - bbox[1]
    return max(cw, 1), max(ch, 1)


def _trim(ascii_text: str) -> list[str]:
    lines = [ln.rstrip() for ln in ascii_text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _fits(lines: list[str], font: Any, canvas: tuple[int, int]) -> bool:
    cw, ch = _measure_cell(font)
    max_cols = max((len(ln) for ln in lines), default=0)
    return max_cols * cw <= canvas[0] and len(lines) * ch <= canvas[1]


def _best_fitting_size(
    lines: list[str], canvas: tuple[int, int], font_size: int
) -> int:
    """Largest size in [_MIN_FONT_SIZE, font_size] whose render fits the canvas.

    Falls through to ``_MIN_FONT_SIZE`` if even the minimum overflows (we
    render anyway — spec says "never crop", best-effort).
    """
    hi = max(font_size, _MIN_FONT_SIZE)
    if _fits(lines, _load_font(hi), canvas):
        return hi
    lo = _MIN_FONT_SIZE
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if _fits(lines, _load_font(mid), canvas):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def render(
    ascii_text: str,
    *,
    canvas: tuple[int, int] = (640, 640),
    font_size: int = 14,
    bg: str = "white",
    fg: str = "black",
) -> Image.Image:
    """Render ``ascii_text`` to a centered, fixed-canvas RGB image.

    Trailing blank lines and trailing whitespace on each line are stripped
    before measurement. If the drawing overflows ``canvas``, ``font_size``
    is binary-searched down to 6pt. Empty input yields a blank canvas.
    """
    img = Image.new("RGB", canvas, color=bg)
    lines = _trim(ascii_text)
    if not lines or all(not ln for ln in lines):
        return img

    size = _best_fitting_size(lines, canvas, font_size)
    font = _load_font(size)
    cw, ch = _measure_cell(font)
    max_cols = max((len(ln) for ln in lines), default=0)
    draw_w = max_cols * cw
    draw_h = len(lines) * ch
    x0 = max((canvas[0] - draw_w) // 2, 0)
    y0 = max((canvas[1] - draw_h) // 2, 0)

    draw = ImageDraw.Draw(img)
    for row, line in enumerate(lines):
        y = y0 + row * ch
        for col, char in enumerate(line):
            if char == " ":
                continue
            draw.text((x0 + col * cw, y), char, fill=fg, font=font)
    return img
