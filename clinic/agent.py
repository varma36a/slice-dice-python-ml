"""Diagnostic agent: tools + orchestrator. No LLM required.

The lesson is the workflow ML sits inside:
  observe chart → flag labs (NumPy) → score admit (sklearn) → retrieve protocol → stop.

Educational only. Not a medical device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from clinic.data import Clinic
from clinic.ml import lab_vector


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., dict[str, Any]]


@dataclass
class AgentState:
    encounter_id: int
    chart: dict[str, Any] | None = None
    lab_flags: dict[str, Any] | None = None
    admit_score: dict[str, Any] | None = None
    protocol: dict[str, Any] | None = None
    red_flags: list[str] = field(default_factory=list)
    steps: int = 0
    done: bool = False
    recommendation: str = ""


@dataclass
class TraceRow:
    step: int
    thought: str
    tool: str
    observation: str


def _chart(clinic: Clinic, encounter_id: int) -> dict[str, Any]:
    row = clinic.encounters.loc[clinic.encounters.encounter_id == encounter_id]
    if row.empty:
        return {"error": f"encounter {encounter_id} not found"}
    r = row.iloc[0]
    p = clinic.patients.loc[clinic.patients.patient_id == r.patient_id]
    comorbid = []
    if len(p):
        pr = p.iloc[0]
        comorbid = [k for k in ("htn", "dm", "copd", "cad") if bool(pr[k])]
    return {
        "encounter_id": int(r.encounter_id),
        "patient_id": str(r.patient_id),
        "age": int(r.age),
        "site": str(r.site),
        "arrival": str(r.arrival),
        "esi": int(r.esi),
        "symptoms": str(r.symptoms).split("|"),
        "wait_min": float(r.wait_min),
        "season": str(r.season),
        "comorbid": comorbid,
        "gold_label_condition": str(r.condition),  # hidden from planner; used in eval only
        "gold_admit": bool(r.admit),
    }


def _flag_labs(clinic: Clinic, encounter_id: int) -> dict[str, Any]:
    row = clinic.encounters.loc[clinic.encounters.encounter_id == encounter_id]
    if row.empty:
        return {"error": "not found"}
    r = row.iloc[0]
    vec = lab_vector(r, clinic.biomarker_names)
    low = vec < clinic.ref_low
    high = vec > clinic.ref_high
    missing = np.isnan(vec)
    flags = []
    for i, name in enumerate(clinic.biomarker_names):
        if missing[i]:
            flags.append({"biomarker": name, "value": None, "flag": "missing"})
        elif low[i]:
            flags.append({"biomarker": name, "value": float(vec[i]), "flag": "low"})
        elif high[i]:
            flags.append({"biomarker": name, "value": float(vec[i]), "flag": "high"})
    z = (vec - clinic.ref_low) / np.clip(clinic.ref_high - clinic.ref_low, 1e-6, None)
    return {
        "n_abnormal": int((low | high)[~missing].sum()) if (~missing).any() else 0,
        "n_missing": int(missing.sum()),
        "flags": flags[:12],
        "spo2": None if np.isnan(vec[clinic.biomarker_names.index("spo2")]) else float(vec[clinic.biomarker_names.index("spo2")]),
        "max_abs_z": None if np.all(np.isnan(z)) else float(np.nanmax(np.abs(z))),
    }


def _score_admit(pipe, clinic: Clinic, encounter_id: int) -> dict[str, Any]:
    from clinic.ml import encounter_model_frame

    frame = encounter_model_frame(clinic)
    row = frame.loc[frame.encounter_id == encounter_id]
    if row.empty:
        return {"error": "not found"}
    cols = ["age", "esi_n", "hour", "spo2", "hr", "temp_c", "sbp", "wbc", "lactate_f", "troponin_f", "rush", "site", "arrival", "season"]
    proba = float(pipe.predict_proba(row[cols])[0, 1])
    return {
        "p_admit": round(proba, 3),
        "band": "high" if proba >= 0.45 else ("mid" if proba >= 0.20 else "low"),
        "esi": int(row.esi_n.iloc[0]),
        "spo2": float(row.spo2.iloc[0]),
    }


def _retrieve_protocol(clinic: Clinic, condition: str) -> dict[str, Any]:
    proto = clinic.protocols.get(condition)
    if not proto:
        return {"error": f"no protocol for {condition}", "known": clinic.condition_names}
    return {"condition": condition, **proto}


def _guess_condition(clinic: Clinic, encounter_id: int) -> str:
    """Cheap nearest-mean in lab space — the agent's 'retriever', not the gold label."""
    row = clinic.encounters.loc[clinic.encounters.encounter_id == encounter_id]
    if row.empty:
        return clinic.condition_names[0]
    vec = lab_vector(row.iloc[0], clinic.biomarker_names)
    means = clinic.lab_means.copy()
    # impute missing with column means of the atlas
    for i, v in enumerate(vec):
        if np.isnan(v):
            vec[i] = means[:, i].mean()
    dist = np.linalg.norm(means - vec, axis=1)
    return clinic.condition_names[int(np.argmin(dist))]


def build_tools(clinic: Clinic, admit_pipe) -> dict[str, Tool]:
    def get_chart(encounter_id: int) -> dict[str, Any]:
        out = _chart(clinic, encounter_id)
        out.pop("gold_label_condition", None)
        out.pop("gold_admit", None)
        return out

    def flag_labs(encounter_id: int) -> dict[str, Any]:
        return _flag_labs(clinic, encounter_id)

    def score_admit(encounter_id: int) -> dict[str, Any]:
        return _score_admit(admit_pipe, clinic, encounter_id)

    def retrieve_protocol(condition: str) -> dict[str, Any]:
        return _retrieve_protocol(clinic, condition)

    def nearest_condition(encounter_id: int) -> dict[str, Any]:
        guess = _guess_condition(clinic, encounter_id)
        return {"guess": guess, "method": "nearest lab-mean (NumPy L2)"}

    return {
        "get_chart": Tool("get_chart", "Load demographics, ESI, arrival, symptoms, comorbidities.", get_chart),
        "flag_labs": Tool("flag_labs", "Compare labs/vitals to reference ranges (NumPy masks).", flag_labs),
        "score_admit": Tool("score_admit", "P(admit) from the sklearn pipeline (fit on past encounters).", score_admit),
        "nearest_condition": Tool("nearest_condition", "Nearest condition prototype in lab space.", nearest_condition),
        "retrieve_protocol": Tool("retrieve_protocol", "Fetch next tests / red flags / disposition hint.", retrieve_protocol),
    }


def plan(state: AgentState) -> tuple[str, str, dict[str, Any]]:
    """Deterministic planner — the workflow is the curriculum, not a hidden LLM."""
    eid = state.encounter_id
    if state.chart is None:
        return ("Need the chart before anything else.", "get_chart", {"encounter_id": eid})
    if state.lab_flags is None:
        return ("Chart in hand. Flag out-of-range labs/vitals.", "flag_labs", {"encounter_id": eid})
    if state.admit_score is None:
        return ("Labs flagged. Score admit risk with the fitted model.", "score_admit", {"encounter_id": eid})
    if state.protocol is None:
        return ("Need a protocol for the nearest lab prototype.", "nearest_condition", {"encounter_id": eid})
    return ("Enough context. Stop and write a suggested plan for the clinician.", "stop", {})


def run_agent(clinic: Clinic, admit_pipe, encounter_id: int, max_steps: int = 8) -> tuple[AgentState, list[TraceRow], list[dict[str, Any]]]:
    tools = build_tools(clinic, admit_pipe)
    state = AgentState(encounter_id=encounter_id)
    trace: list[TraceRow] = []
    gold = _chart(clinic, encounter_id)

    for step in range(1, max_steps + 1):
        thought, tool_name, args = plan(state)
        state.steps = step
        if tool_name == "stop":
            rec = _recommend(state)
            state.done = True
            state.recommendation = rec
            trace.append(TraceRow(step, thought, "stop", rec))
            break
        tool = tools[tool_name]
        obs = tool.fn(**args)
        obs_s = _short(obs)
        trace.append(TraceRow(step, thought, tool_name, obs_s))
        if tool_name == "get_chart":
            state.chart = obs
        elif tool_name == "flag_labs":
            state.lab_flags = obs
            state.red_flags = [
                f"{f['biomarker']} {f['flag']}"
                for f in obs.get("flags", [])
                if f["flag"] in ("high", "low") and f["biomarker"] in {"spo2", "lactate", "troponin", "sbp"}
            ]
        elif tool_name == "score_admit":
            state.admit_score = obs
        elif tool_name == "nearest_condition":
            guess = obs.get("guess")
            state.protocol = tools["retrieve_protocol"].fn(guess)
            trace.append(
                TraceRow(
                    step,
                    f"Retrieve protocol for guess={guess}.",
                    "retrieve_protocol",
                    _short(state.protocol),
                )
            )
    else:
        state.recommendation = _recommend(state)
        state.done = True

    eval_rows = [
        {
            "gold_condition": gold.get("gold_label_condition"),
            "guess": (state.protocol or {}).get("condition"),
            "gold_admit": gold.get("gold_admit"),
            "p_admit": (state.admit_score or {}).get("p_admit"),
            "match": (state.protocol or {}).get("condition") == gold.get("gold_label_condition"),
        }
    ]
    return state, trace, eval_rows


def _recommend(state: AgentState) -> str:
    band = (state.admit_score or {}).get("band", "unknown")
    cond = (state.protocol or {}).get("condition", "unspecified")
    nxt = (state.protocol or {}).get("next_tests", [])
    flags = ", ".join(state.red_flags) if state.red_flags else "none"
    disp = (state.protocol or {}).get("disposition_hint", "reassess")
    return (
        f"Suggested (not a diagnosis): nearest prototype {cond}. Admit-risk band {band}. "
        f"Red flags: {flags}. Next tests: {', '.join(nxt) or 'none listed'}. "
        f"Disposition hint: {disp}. Clinician must confirm."
    )


def _short(obs: dict[str, Any], n: int = 220) -> str:
    s = str({k: v for k, v in obs.items() if k != "flags"})
    if "flags" in obs:
        s += f" | flags={obs.get('flags', [])[:6]}"
    return s if len(s) <= n else s[: n - 1] + "…"
