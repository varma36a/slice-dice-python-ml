"""Interview drill — questions with clinic examples, plus live demos."""

import streamlit as st

from clinic.interview import all_questions, live_interview
from clinic.quiz import drill, flashcard
from clinic.topics import areas
from clinic.ui import header, inject, load_clinic, warn

inject()
clinic = load_clinic()

header(
    "Interview questions",
    "Python · NumPy · Pandas · ML · agents",
    "Same style as Dotnet-InterviewQuestions: a topic map, a clinic example, a question you should answer out loud.",
)
warn("Synthetic educational data. Not medical advice. Not a clinical exam.")

mode = st.radio("Mode", ["Live examples", "Flashcards (say this)", "Multiple choice"], horizontal=True)
area_opts = ["All"] + areas() + ["High-yield extras"]
area = st.selectbox("Area", area_opts)

qs = all_questions()
if area == "High-yield extras":
    qs = [q for q in qs if q.get("source") == "bank"]
elif area != "All":
    qs = [q for q in qs if q["area"] == area]

if mode == "Multiple choice" and area == "All":
    st.info("Multiple choice is per area so the page stays usable. Showing **high-yield extras**. Or pick a module above.")
    qs = [q for q in all_questions() if q.get("source") == "bank"]

st.caption(f"{len(qs)} questions")

if mode == "Live examples":
    st.markdown(
        """
These are the clinic equivalents of the pizza-store interview endpoints
(`GET /api/interview/di-lifetimes`, LINQ demos, strategy examples): **run the math on Northshore tables**.
"""
    )
    live = live_interview(clinic)
    for title, frame in live.items():
        st.markdown(f"**{title}**")
        st.dataframe(frame, hide_index=True, width="stretch")
    st.markdown("**Agent planner (deterministic)**")
    st.code(
        """if state.chart is None:        → get_chart
elif state.lab_flags is None:  → flag_labs
elif state.admit_score is None:→ score_admit
elif state.protocol is None:   → nearest_condition → retrieve_protocol
else:                          → stop""",
        language="text",
    )

elif mode == "Flashcards (say this)":
    st.markdown("Open a question, say the answer out loud, then reveal.")
    for i, q in enumerate(qs):
        flashcard(
            f"fc_{area}_{i}_{q['topic']}",
            f"{q['topic']} — {q['interview_q']}",
            str(q["interview_a"]),
            example=str(q["example"]),
            where=str(q["where"]),
        )

else:
    st.markdown("Pick an answer. The explanation is what you should say in the room.")
    for i, q in enumerate(qs):
        choices = list(q["choices"])
        drill(
            f"mc_{area}_{i}_{q['topic']}",
            f"{q['topic']}: {q['interview_q']}",
            choices,
            str(q["answer"]),
            str(q["interview_a"]),
            example=str(q["example"]),
        )
