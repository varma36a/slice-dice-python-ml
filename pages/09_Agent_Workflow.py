"""Diagnostic agent lab — tools, planner, trace."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from clinic.agent import build_tools, run_agent
from clinic.ml import admit_transformer, encounter_model_frame, time_split
from clinic.cards import module_explainers
from clinic.quiz import ask
from clinic.ui import header, inject, load_clinic, ok, pitfall, warn, why

inject()
clinic = load_clinic()

header(
    "Module 09 · Agent workflow",
    "The model is a tool. The product is a loop.",
    "Observe chart → flag labs (NumPy) → score admit (sklearn) → retrieve protocol → stop. No LLM required to learn the architecture.",
)
warn("This agent is a teaching orchestrator on synthetic data. It is not a diagnostic system and must not be used in care.")
why(
    "Production 'AI doctors' fail at tool design, state, and stop conditions — not at picking GPT vs Llama. "
    "If you can write this loop, swapping in an LLM planner later is a one-function change."
)
module_explainers("Agent")

st.markdown("### Architecture")
st.code(
    """for step in range(max_steps):
    thought, tool, args = plan(state)      # policy (here: deterministic)
    if tool == "stop":
        return recommend(state)
    obs = tools[tool].fn(**args)           # get_chart | flag_labs | score_admit | retrieve
    state = update(state, obs)             # memory
""",
    language="python",
)

cols = [
    "age", "esi_n", "hour", "spo2", "hr", "temp_c", "sbp", "wbc",
    "lactate_f", "troponin_f", "rush", "site", "arrival", "season",
]


@st.cache_resource(show_spinner="Fitting admit pipeline for the agent…")
def _admit_pipe():
    cl = load_clinic()
    df = encounter_model_frame(cl)
    tr, _ = time_split(df, "date", 0.75)
    pipe = Pipeline(
        [
            ("prep", admit_transformer()),
            ("model", RandomForestClassifier(n_estimators=160, min_samples_leaf=4, class_weight="balanced", random_state=0, n_jobs=-1)),
        ]
    )
    pipe.fit(df.loc[tr, cols], df.loc[tr, "admit_int"])
    return pipe


pipe = _admit_pipe()
tools = build_tools(clinic, pipe)

st.markdown("### Tool registry")
st.dataframe(
    pd.DataFrame([{"tool": t.name, "contract": t.description} for t in tools.values()]),
    hide_index=True,
    width="stretch",
)

st.markdown("### Run the agent on a held-out-style case")
# Prefer later dates so it feels like "tonight"
pool = clinic.encounters.sort_values("ts").tail(400)
eid = st.selectbox("Encounter", pool["encounter_id"].tolist())
preview = pool.loc[pool.encounter_id == eid, ["encounter_id", "site", "age", "esi", "arrival", "symptoms", "spo2", "hr"]].iloc[0]
st.dataframe(pd.DataFrame([preview.to_dict()]), hide_index=True, width="stretch")
st.caption("Gold labels are hidden from the planner. They show up only in the eval row after stop.")

if st.button("Run agent", type="primary"):
    state, trace, ev = run_agent(clinic, pipe, int(eid))
    st.session_state["agent_trace"] = trace
    st.session_state["agent_eval"] = ev
    st.session_state["agent_rec"] = state.recommendation

if "agent_trace" in st.session_state:
    tr = st.session_state["agent_trace"]
    st.markdown("### Trace (thought → tool → observation)")
    st.dataframe(
        pd.DataFrame([{"step": r.step, "thought": r.thought, "tool": r.tool, "observation": r.observation} for r in tr]),
        hide_index=True,
        width="stretch",
    )
    st.success(st.session_state["agent_rec"])
    st.markdown("### Eval vs gold (for the lab, not for the model)")
    st.dataframe(pd.DataFrame(st.session_state["agent_eval"]), hide_index=True, width="stretch")

st.markdown("### What the planner actually does")
st.code(
    """if state.chart is None:        return get_chart
if state.lab_flags is None:    return flag_labs
if state.admit_score is None:  return score_admit
if state.protocol is None:     return nearest_condition → retrieve_protocol
return stop
""",
    language="text",
)
ok(
    "That is an agent: a policy over tools, with memory, and a stop rule. "
    "Replace `plan()` with an LLM that must emit `{tool, args}` JSON and you have the industry pattern — with the same guardrail: clinician confirms."
)
pitfall(
    "Letting the planner see `gold_label_condition` is leakage dressed as RAG. "
    "Passing `patient_id` into a prompt is the same bug as one-hot encoding MRN."
)

st.markdown("### Manual tool call (you are the planner)")
tool_name = st.selectbox("Tool", list(tools))
if tool_name == "retrieve_protocol":
    cond = st.selectbox("condition", clinic.condition_names, key="proto_cond")
    if st.button("Call retrieve_protocol"):
        st.json(tools[tool_name].fn(cond))
else:
    if st.button(f"Call {tool_name}"):
        st.json(tools[tool_name].fn(int(eid)))

ask(
    "q9_agent",
    "The admit model returns P=0.62. The agent should:",
    [
        "Print 'admit this patient' as a diagnosis",
        "Treat it as one observation, combine with red flags + protocol, stop with a suggested plan for the clinician",
        "Retry the model until P>0.9",
    ],
    "Treat it as one observation, combine with red flags + protocol, stop with a suggested plan for the clinician",
    "A probability is an observation, not an action. The workflow owns the action, with a human in the loop.",
)
