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


def flashcard(qid: str, question: str, answer: str, example: str | None = None, where: str | None = None) -> None:
    """Interview card: question first, reveal answer + clinic example (Dotnet-InterviewQuestions style)."""
    with st.expander(question, expanded=False):
        if where:
            st.caption(str(where))
        st.markdown(answer)
        if example:
            st.code(example, language="python")


def drill(
    qid: str,
    question: str,
    options: list[str],
    answer: str,
    explain: str,
    example: str | None = None,
) -> None:
    ask(qid, question, options, answer, explain)
    if example:
        with st.expander("Clinic example", expanded=False):
            st.code(example, language="python")
