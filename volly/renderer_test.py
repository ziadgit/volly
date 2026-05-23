"""Tests for ``volly.renderer`` — see ``specs/05-renderer.md``."""

from __future__ import annotations

from PIL import Image

from volly.renderer import render


def _has_non_bg_pixel(img: Image.Image, bg: tuple[int, int, int] = (255, 255, 255)) -> bool:
    return any(px != bg for px in img.get_flattened_data())


def test_render_three_lines_produces_non_blank_image_at_requested_size() -> None:
    img = render("ABC\nDEF\nGHI", canvas=(320, 320), font_size=20)
    assert img.size == (320, 320)
    assert img.mode == "RGB"
    assert _has_non_bg_pixel(img)


def test_render_empty_string_returns_blank_canvas() -> None:
    img = render("", canvas=(64, 64))
    assert img.size == (64, 64)
    assert not _has_non_bg_pixel(img)


def test_render_whitespace_only_returns_blank_canvas() -> None:
    img = render("   \n\n  \n", canvas=(64, 64))
    assert not _has_non_bg_pixel(img)


def test_render_overflow_downscales_and_does_not_crash() -> None:
    # 200 lines × 200 cols at 14pt vastly exceeds 64×64 px — must shrink.
    huge = "\n".join("M" * 200 for _ in range(200))
    img = render(huge, canvas=(64, 64), font_size=14)
    assert img.size == (64, 64)
    assert _has_non_bg_pixel(img)


def test_render_trailing_blank_lines_do_not_shift_centering() -> None:
    base = render("X", canvas=(64, 64), font_size=14)
    padded = render("X\n\n\n", canvas=(64, 64), font_size=14)
    assert base.get_flattened_data() == padded.get_flattened_data()


def test_render_respects_fg_color() -> None:
    img = render("XXXX\nXXXX", canvas=(64, 64), font_size=18, fg="red", bg="white")
    has_red_dominant = any(r > g and r > b for (r, g, b) in img.get_flattened_data())
    assert has_red_dominant


def test_render_respects_bg_color() -> None:
    img = render("", canvas=(8, 8), bg="black")
    assert all(px == (0, 0, 0) for px in img.get_flattened_data())
