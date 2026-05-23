# 05 — Renderer

ASCII text → PNG. Pure local code, no LLM. Existence justified by one
critical claim in `specs/06-judge.md`: **the judge must see images, not
text**.

## Public surface

```python
def render(
    ascii_text: str,
    *,
    canvas: tuple[int, int] = (640, 640),  # pixels
    font_size: int = 14,
    bg: str = "white",
    fg: str = "black",
) -> PIL.Image.Image: ...
```

## Behavior

- Splits `ascii_text` into lines. Computes a uniform monospace cell from
  `font_size` and renders each character at `(col*cw, row*ch)`.
- Centers the drawing inside `canvas` (compute bounding box, offset).
- Uses a bundled monospace font (PIL's default is not great — prefer
  `DejaVuSansMono` from `matplotlib.font_manager.findfont` or fall back to
  `ImageFont.load_default()` if not present).
- Trims trailing blank lines and trailing spaces before measuring.
- If the ASCII overflows the canvas, **scale down `font_size` until it
  fits** (binary search between 6pt and the requested size). Never crop.

## Why fixed canvas

Judge compares 8 images side-by-side. They must be the same size or the
judge's spatial cues degrade. Same canvas, same font, same colors — only
the *drawing* differs.

## Output format

PNG via `Image.save(buf, format="PNG")`. Returned as a `PIL.Image.Image`;
the loop persists each candidate to `iter-NN/cand-MM.png`.

## Test surface

- `renderer_test.py` renders a 3-line known string, asserts non-blank
  output and correct dimensions.
- Verifies overflow → downscale path with a deliberately huge input.
- Verifies a single empty string renders a blank canvas (no crash).
