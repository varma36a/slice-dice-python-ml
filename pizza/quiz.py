"""Tiny quiz helper — no backend, just session state."""

from __future__ import annotations

import streamlit as st


def ask(qid: str, question: str, options: list[str], answer: str, explain: str) -> None:
    st.markdown(f"**Check:** {question}")
    pick = st.radio("Your answer", options, key=qid, index=None, label_visibility="collapsed")
    if pick is None:
        return
    if pick == answer:
        st.success(explain)
    else:
        st.error(f"Not that one. {explain}")
