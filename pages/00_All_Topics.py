"""Sidebar: every topic with a clinic example + interview Q — Dotnet-InterviewQuestions layout."""

import pandas as pd
import streamlit as st

from clinic.interview import live_interview
from clinic.quiz import flashcard
from clinic.topics import CATALOG_COLS, TOPICS, areas, catalog_frame
from clinic.ui import header, inject, load_clinic, warn

inject()
clinic = load_clinic()

header(
    "All topics",
    "Concept · example · interview question",
    "Same map as the .NET pizza-store interview APIs: every topic has a Northshore example you can read, and a question you should be able to answer.",
)
warn("Synthetic educational data. Not medical advice.")

st.markdown(
    """
Each row is **Topic / Concept / Example / Where** — then open a topic for the full snippet and the interview answer.
Drill mode lives on **Interview questions** in the sidebar.
"""
)

area_opts = ["All"] + areas()
pick = st.selectbox("Area", area_opts)
cat = catalog_frame()
show = cat if pick == "All" else cat[cat["area"] == pick]
st.caption(f"{len(show)} topics" + ("" if pick == "All" else f" in {pick}"))
st.dataframe(show[CATALOG_COLS], hide_index=True, width="stretch", height=420)

pool = TOPICS if pick == "All" else [t for t in TOPICS if t["area"] == pick]
labels = [f"{t['area']} · {t['topic']}" for t in pool]
choice = st.selectbox("Open a topic (example + interview)", labels)
item = pool[labels.index(choice)]

st.markdown(f"### {item['topic']}")
st.caption(f"{item['area']} · {item['where']}")
st.markdown(item["what"])
st.markdown("**Example (Northshore)**")
st.code(item["example"], language="python")
flashcard(
    f"topic_card_{item['area']}_{item['topic']}",
    str(item["interview_q"]),
    str(item["interview_a"]),
    example=str(item["example"]),
    where=str(item["where"]),
)

st.markdown("---")
st.markdown("### Live examples")
st.caption("Runnable on this clinic — same idea as `GET /api/interview/topics` then hitting a demo endpoint.")
live = live_interview(clinic)
live_pick = st.selectbox("Run", list(live))
st.dataframe(live[live_pick], hide_index=True, width="stretch")

st.markdown("### By module")
for area, block in pd.DataFrame(pool).groupby("area", sort=False):
    with st.expander(f"{area} ({len(block)})", expanded=False):
        st.dataframe(block[list(CATALOG_COLS)], hide_index=True, width="stretch")
