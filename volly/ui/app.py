"""Streamlit four-panel dashboard for Volly.

See ``specs/10-ui.md``. Reads ``state.json`` from the most recent run
under ``VOLLY_RUN_DIR`` and rerenders ~1Hz while a background loop runs
in a single-worker ``ThreadPoolExecutor``.
"""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import streamlit as st

from volly.loop import CURATED_SUBJECTS, LoopConfig
from volly.loop import run as loop_run
from volly.state import RunHistory

POLL_SECONDS = 1.0
SUBJECTS: tuple[str, ...] = tuple(sorted(CURATED_SUBJECTS))


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


def winrate_chart_data(history: RunHistory) -> dict[str, list[float | None]]:
    """Pad evolving + control win-rate series to the same length for ``st.line_chart``.

    Returns an empty dict when neither arm has recorded any iteration so the
    caller can show a "waiting" placeholder instead of an empty chart.
    """
    evolving = history.win_rates("evolving")
    control = history.win_rates("control")
    n = max(len(evolving), len(control))

    def pad(xs: list[float]) -> list[float | None]:
        return list(xs) + [None] * (n - len(xs))

    out: dict[str, list[float | None]] = {}
    if evolving:
        out["evolving"] = pad(evolving)
    if control:
        out["control"] = pad(control)
    return out


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


def _render_header() -> tuple[str, int, bool, bool]:
    col_subject, col_iters, col_control, col_button = st.columns([2, 1, 1, 1])
    with col_subject:
        subject = st.selectbox("Subject", SUBJECTS, key="subject_input")
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
    return str(subject), int(iterations), bool(no_control), bool(start)


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
    st.line_chart(data)


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
