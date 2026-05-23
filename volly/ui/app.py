"""Streamlit four-panel dashboard for Volly.

See ``specs/10-ui.md``. Reads ``state.json`` from the most recent run
under ``VOLLY_RUN_DIR`` and rerenders ~1Hz while a background loop runs
in a single-worker ``ThreadPoolExecutor``.
"""

from __future__ import annotations

import asyncio
import difflib
import os
import time
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import altair as alt
import streamlit as st

from volly.loop import CURATED_SUBJECTS, LoopConfig
from volly.loop import run as loop_run
from volly.state import RunHistory

POLL_SECONDS = 1.0
SUBJECTS: tuple[str, ...] = tuple(sorted(CURATED_SUBJECTS))

# Cutoff for ``difflib.get_close_matches`` when sanitizing free-text subjects.
# 0.6 is the library default — it catches the realistic operator typos
# ("sailbot", "coffe cup", "heeart") while rejecting unrelated nouns
# ("airplane", "dragon"). Tighter cutoffs start dropping useful typo fixes.
_SUBJECT_MATCH_CUTOFF = 0.6

# Evolving = the system learning (green, foreground). Control = the frozen
# baseline (gray, recedes). Matches specs/09-control.md: the demo lives or
# dies on whether the audience reads the two curves at a glance.
ARM_COLORS: dict[str, str] = {"evolving": "#16a34a", "control": "#9ca3af"}


def sanitize_subject(
    text: str, curated: Iterable[str] = CURATED_SUBJECTS
) -> str | None:
    """Map free-text input to the closest curated subject, or ``None`` if none fit.

    Spec ``specs/10-ui.md`` §"Subject input" requires that free-text typed into
    the UI be sanitized to a curated subject before the loop receives it — the
    loop's ``validate_subject`` only accepts members of ``CURATED_SUBJECTS``.
    Matching is case-insensitive and tolerates surrounding whitespace; the
    ``difflib`` cutoff is tuned so realistic typos survive but unrelated nouns
    return ``None`` (UI falls back to the dropdown default + a warning).
    """
    normalized = text.strip().lower()
    if not normalized:
        return None
    curated_list = sorted(curated)
    if normalized in curated_list:
        return normalized
    matches = difflib.get_close_matches(
        normalized, curated_list, n=1, cutoff=_SUBJECT_MATCH_CUTOFF
    )
    return matches[0] if matches else None


def run_root() -> Path:
    return Path(os.environ.get("VOLLY_RUN_DIR", "runs"))


def latest_run_dir(root: Path) -> Path | None:
    """Most-recently-modified subdirectory of ``root`` containing ``state.json``."""
    if not root.exists():
        return None
    candidates = [
        p for p in root.iterdir() if p.is_dir() and (p / "state.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "state.json").stat().st_mtime)


def load_history(run_dir: Path) -> RunHistory | None:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        return None
    try:
        return RunHistory.load(state_path)
    except (OSError, ValueError, KeyError):
        return None


def winrate_chart_data(history: RunHistory) -> list[dict[str, float | int | str]]:
    """Long-form win-rate records per (arm, iteration), suitable for altair.

    Returns ``[]`` when no iteration has been recorded so the caller can show
    a "waiting" placeholder instead of an empty chart. Iterations are 1-based
    to match how ``RunHistory`` indexes them; arms with no data are simply
    absent from the output (no padding — altair handles ragged series).
    """
    records: list[dict[str, float | int | str]] = []
    for arm in ("evolving", "control"):
        for i, w in enumerate(history.win_rates(arm), start=1):
            records.append({"iteration": i, "arm": arm, "win_rate": w})
    return records


def _winrate_chart(data: list[dict[str, float | int | str]]) -> alt.Chart:
    """Altair line chart with bounded y-axis [0, 1] and explicit arm colors.

    Y-axis bounds are non-negotiable: an autoscaled chart makes a flat 0.18–0.22
    control noise band look like big swings, which destroys the demo's
    "evolving climbs, control stays flat" story. Spec: ``specs/09-control.md``.
    """
    arms_present = [arm for arm in ARM_COLORS if any(r["arm"] == arm for r in data)]
    color_scale = alt.Scale(
        domain=arms_present,
        range=[ARM_COLORS[a] for a in arms_present],
    )
    return (
        alt.Chart(alt.Data(values=data))
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X(
                "iteration:Q",
                axis=alt.Axis(title="iteration", tickMinStep=1, format="d"),
            ),
            y=alt.Y(
                "win_rate:Q",
                scale=alt.Scale(domain=[0.0, 1.0], clamp=True),
                axis=alt.Axis(title="win rate", format=".1f"),
            ),
            color=alt.Color(
                "arm:N",
                scale=color_scale,
                legend=alt.Legend(title="arm"),
            ),
        )
    )


@st.cache_resource
def _executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1)


def _run_loop_blocking(config: LoopConfig) -> RunHistory:
    return asyncio.run(loop_run(config))


def _start_run(config: LoopConfig) -> Future[RunHistory]:
    return _executor().submit(_run_loop_blocking, config)


def _running_future() -> Future[RunHistory] | None:
    future = st.session_state.get("future")
    if future is None or future.done():
        return None
    return future


def _resolve_subject(free_text: str, dropdown_value: str) -> tuple[str, str | None]:
    """Pick the effective subject from header inputs and the user-facing notice.

    Free-text takes precedence when it sanitizes to a curated subject; an
    empty box defers to the dropdown silently. A non-empty box with no
    plausible match falls back to the dropdown and asks the caller to
    surface a warning so the operator sees why their typed input was
    ignored. Returns ``(subject, notice)`` where ``notice`` is ``None`` if
    no message should be shown, ``"info:..."`` for a sanitization hint, or
    ``"warn:..."`` for the unmatched-fallback warning.
    """
    typed = free_text.strip()
    if not typed:
        return dropdown_value, None
    matched = sanitize_subject(typed)
    if matched is None:
        return dropdown_value, (
            f"warn:'{typed}' isn't on the curated list and has no close match — "
            f"using '{dropdown_value}' from the dropdown instead."
        )
    if matched == typed.lower():
        return matched, None
    return matched, f"info:Sanitized '{typed}' → '{matched}'."


def _render_header() -> tuple[str, int, bool, bool]:
    col_subject, col_iters, col_control, col_button = st.columns([2, 1, 1, 1])
    with col_subject:
        free_text = st.text_input(
            "Subject (free text — fuzzy-matched to the curated list)",
            value="",
            placeholder="e.g. cat, sailboat, coffee cup",
            key="subject_text_input",
        )
        dropdown = st.selectbox(
            "…or pick from the curated list", SUBJECTS, key="subject_input"
        )
    with col_iters:
        iterations = st.number_input(
            "Iterations", min_value=1, max_value=20, value=8, step=1, key="iter_input"
        )
    with col_control:
        no_control = st.checkbox("Skip control arm", value=False, key="no_control_input")
    running = _running_future() is not None
    with col_button:
        st.write("")  # vertical alignment with the input labels
        if running:
            if st.button("Stop", type="secondary", use_container_width=True):
                future = st.session_state.get("future")
                if future is not None:
                    future.cancel()
                st.session_state.pop("future", None)
                st.rerun()
            start = False
        else:
            start = st.button("Run", type="primary", use_container_width=True)
    subject, notice = _resolve_subject(str(free_text), str(dropdown))
    if notice is not None:
        kind, _, body = notice.partition(":")
        (st.warning if kind == "warn" else st.caption)(body)
    return subject, int(iterations), bool(no_control), bool(start)


def _render_prompt_panel(history: RunHistory) -> None:
    st.markdown("### Panel 1 — Evolving Prompt")
    versions = history.prompt_versions()
    if not versions:
        st.info("Waiting for the first evolving iteration…")
        return
    latest_idx = len(versions) - 1
    st.code(versions[latest_idx], language=None)
    diff = history.diff(latest_idx)
    if diff:
        st.caption(f"Diff: prompt v{latest_idx - 1} → v{latest_idx}")
        st.code(diff, language="diff")
    elif latest_idx > 0:
        st.caption("Prompt unchanged from the previous iteration.")
    else:
        st.caption("Seed prompt — no prior iteration to diff against yet.")


def _render_best_image_panel(history: RunHistory) -> None:
    st.markdown("### Panel 2 — Current Best")
    latest = history.latest("evolving")
    if latest is None:
        st.info("Waiting for the first best-of-N image…")
        return
    if not latest.best_image_path.exists():
        st.warning(f"Best image missing on disk: {latest.best_image_path}")
        return
    st.image(str(latest.best_image_path), use_container_width=True)
    st.caption(
        f"Iteration {latest.iter_index} · win rate {latest.win_rate:.2f}"
    )


def _render_winrate_panel(history: RunHistory) -> None:
    st.markdown("### Panel 3 — Win Rate")
    data = winrate_chart_data(history)
    if not data:
        st.info("Waiting for the first iteration…")
        return
    st.altair_chart(_winrate_chart(data), use_container_width=True)


def _render_critique_panel(history: RunHistory) -> None:
    st.markdown("### Panel 4 — Judge Reasoning")
    latest = history.latest("evolving")
    if latest is None:
        st.info("Waiting for the first judge critique…")
        return
    st.markdown(latest.judge.critique or "_(empty critique)_")
    if latest.judge.prompt_suggestions:
        st.markdown("**Suggested prompt edits:**")
        for suggestion in latest.judge.prompt_suggestions:
            st.markdown(f"- {suggestion}")


def _render_panels(history: RunHistory) -> None:
    row1_left, row1_right = st.columns(2)
    with row1_left:
        _render_prompt_panel(history)
    with row1_right:
        _render_best_image_panel(history)
    row2_left, row2_right = st.columns(2)
    with row2_left:
        _render_winrate_panel(history)
    with row2_right:
        _render_critique_panel(history)


def _consume_finished_future() -> None:
    future = st.session_state.get("future")
    if future is None or not future.done():
        return
    err = future.exception()
    if err is not None:
        st.session_state["error"] = f"{type(err).__name__}: {err}"
    else:
        st.session_state.pop("error", None)
    st.session_state.pop("future", None)


def main() -> None:
    st.set_page_config(page_title="Volly", layout="wide")
    st.title("Volly — system prompt learning for ASCII art")

    subject, iterations, no_control, start = _render_header()

    if start:
        config = LoopConfig(
            subject=subject,
            iterations=iterations,
            no_control=no_control,
        )
        st.session_state["future"] = _start_run(config)
        st.session_state.pop("error", None)
        st.rerun()

    _consume_finished_future()
    if "error" in st.session_state:
        st.error(f"Run failed: {st.session_state['error']}")

    run_dir = latest_run_dir(run_root())
    if run_dir is None:
        st.info("No runs found yet — pick a subject above and press Run.")
        return

    history = load_history(run_dir)
    if history is None:
        st.warning(f"state.json under {run_dir} is missing or unreadable.")
        return

    running = _running_future() is not None
    status = "running" if running else "finished"
    st.caption(
        f"Run: `{run_dir.name}` · subject: **{history.subject}** · status: {status}"
    )
    _render_panels(history)

    if running:
        time.sleep(POLL_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()
