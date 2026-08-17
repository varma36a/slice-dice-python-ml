"""Sidebar page: every topic on the site in one place."""

import pandas as pd
import streamlit as st

from clinic.topics import TOPICS
from clinic.ui import header, inject, warn

inject()

header(
    "All topics",
    "Everything this site covers",
    "Python through the diagnostic agent — one catalog, same Northshore tables.",
)
warn("Synthetic educational data. Not medical advice.")

df = pd.DataFrame(TOPICS)
areas = ["All"] + list(dict.fromkeys(df["area"].tolist()))
pick = st.selectbox("Area", areas)
show = df if pick == "All" else df[df["area"] == pick]
st.caption(f"{len(show)} topics" + ("" if pick == "All" else f" in {pick}"))
st.dataframe(show, hide_index=True, width="stretch", height=720)

st.markdown("### By module")
for area, block in df.groupby("area", sort=False):
    with st.expander(f"{area} ({len(block)})", expanded=False):
        st.dataframe(block[["topic", "what", "where"]], hide_index=True, width="stretch")
