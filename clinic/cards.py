"""Render every topic as a full example card (used by All Topics · By module and Home)."""

from __future__ import annotations

from collections import OrderedDict

import pandas as pd
import streamlit as st

from clinic.quiz import flashcard
from clinic.worked import load_worked


@st.cache_data(show_spinner="Loading topic examples…")
def cached_worked(seed: int, n_enc: int):
    return load_worked()


def topic_example_card(
    t: dict,
    *,
    key: str,
    result: pd.DataFrame | None = None,
    interview: bool = True,
) -> None:
    """One topic: concept, full clinic example, live result, interview Q."""
    st.markdown(f"**{t['topic']}**")
    st.caption(str(t["where"]))
    st.markdown(str(t["what"]))
    expl = str(t.get("explain") or "").strip()
    if expl:
        with st.expander("Explain with examples (same style as acuity_weight)", expanded=False):
            st.markdown(expl)
    st.markdown("**Example**")
    st.code(str(t["example"]), language="python")
    if result is not None and len(result):
        st.caption("Result on this clinic")
        st.dataframe(result, hide_index=True, width="stretch")
    if interview:
        flashcard(
            key,
            str(t["interview_q"]),
            str(t["interview_a"]),
            example=None,
            where=str(t["where"]),
        )


def render_by_module(
    topics: list[dict],
    *,
    key_prefix: str,
    results: dict[tuple[str, str], pd.DataFrame] | None = None,
    interview: bool = True,
    expand_first: bool = False,
    expand_all: bool = False,
) -> None:
    """Every topic under its module, with the example in full — not a truncated table."""
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for t in topics:
        grouped.setdefault(str(t["area"]), []).append(t)
    results = results or {}
    for i, (area, items) in enumerate(grouped.items()):
        with st.expander(
            f"{area} · {len(items)} topics with examples",
            expanded=expand_all or (expand_first and i == 0),
        ):
            for j, t in enumerate(items):
                topic_example_card(
                    t,
                    key=f"{key_prefix}_{area}_{j}_{t['topic']}",
                    result=results.get((str(t["area"]), str(t["topic"]))),
                    interview=interview,
                )
                if j < len(items) - 1:
                    st.markdown("---")


def module_explainers(area: str, *, expanded: bool = False) -> None:
    """Lab pages: every topic in this module with the worked-example writeup."""
    from clinic.topics import TOPICS

    items = [t for t in TOPICS if t["area"] == area]
    if not items:
        return
    st.markdown("### Explain with examples")
    st.caption("Same walkthrough style as `acuity_weight`: English, numbered calls, what it is not.")
    for t in items:
        with st.expander(f"{t['topic']}", expanded=expanded):
            expl = str(t.get("explain") or "").strip()
            if expl:
                st.markdown(expl)
            st.code(str(t["example"]), language="python")
