"""Sidebar: every topic with a clinic example + interview Q — Dotnet-InterviewQuestions layout."""

import streamlit as st

from clinic.cards import cached_worked, render_by_module, topic_example_card
from clinic.interview import live_interview
from clinic.topics import CATALOG_COLS, TOPICS, areas, catalog_frame
from clinic.ui import header, inject, load_clinic, warn

inject()
clinic = load_clinic()

header(
    "All topics",
    "Every topic · full example · result on this clinic",
    "By module lists every topic with the Northshore snippet and the table it produces — not a truncated summary.",
)
warn("Synthetic educational data. Not medical advice.")

st.markdown(
    """
**By module** below is the full catalog: concept, example code with numbered call traces, live result, interview question.
Open **Explain with examples** on any card — same walkthrough style as `acuity_weight` (English, numbered calls, picture, what it is not).
Drill mode (flashcards / multiple choice) is **Interview questions** in the sidebar.
"""
)


results = cached_worked(int(clinic.meta.get("seed", 0)), int(clinic.meta.get("n_encounters", 0)))

area_opts = areas() + ["All"]
pick = st.selectbox("Area", area_opts)
pool = TOPICS if pick == "All" else [t for t in TOPICS if t["area"] == pick]
st.caption(f"{len(pool)} topics with examples" + ("" if pick == "All" else f" in {pick}"))

st.markdown("### By module")
st.markdown("Open a module — **every topic** is there with the example in full and the result on these tables.")
expand_all = st.checkbox("Expand all modules", value=pick != "All")
render_by_module(
    pool,
    key_prefix=f"mod_{pick}",
    results=results,
    interview=True,
    expand_first=True,
    expand_all=expand_all,
)

st.markdown("---")
st.markdown("### Index (scan)")
cat = catalog_frame()
show = cat if pick == "All" else cat[cat["area"] == pick]
st.dataframe(show[CATALOG_COLS], hide_index=True, width="stretch", height=280)

st.markdown("### Open one topic")
labels = [f"{t['area']} · {t['topic']}" for t in pool]
choice = st.selectbox("Topic", labels)
item = pool[labels.index(choice)]
topic_example_card(
    item,
    key=f"one_{item['area']}_{item['topic']}",
    result=results.get((str(item["area"]), str(item["topic"]))),
)

st.markdown("---")
st.markdown("### Live extras")
st.caption("Same idea as `GET /api/interview/topics` then a demo endpoint.")
live = live_interview(clinic)
live_pick = st.selectbox("Run", list(live))
st.dataframe(live[live_pick], hide_index=True, width="stretch")
