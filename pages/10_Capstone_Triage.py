"""Triage command center — NumPy + Pandas + models + agent on one screen."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clinic.agent import run_agent
from clinic.ml import (
    admit_transformer,
    daily_model_frame,
    daily_transformer,
    encounter_model_frame,
    patient_feature_frame,
    time_split,
)
from clinic.ui import header, inject, load_clinic, ok, warn, why

inject()
clinic = load_clinic()

header(
    "Module 10 · Capstone",
    "Triage command center",
    "Census · reagent pull · admit risk · diagnostic agent. This is the exam: can the pieces talk to each other?",
)
warn("Synthetic clinic. Not for operations or care.")
why("A real system is a decision: who to see next, which assay to run, whether to escalate. The math is NumPy + Pandas + a Pipeline + a tool loop.")

admit_cols = [
    "age", "esi_n", "hour", "spo2", "hr", "temp_c", "sbp", "wbc",
    "lactate_f", "troponin_f", "rush", "site", "arrival", "season",
]
daily_cols = ["site", "season", "dow", "is_weekend", "flu_wave", "is_heat", "enc_lag7", "enc_roll7"]


@st.cache_resource(show_spinner="Training clinic models (cached)…")
def _models():
    cl = load_clinic()
    daily = daily_model_frame(cl)
    tr, te = time_split(daily, "date", 0.75)
    demand = Pipeline(
        [
            ("prep", daily_transformer()),
            ("model", RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0, n_jobs=-1)),
        ]
    )
    demand.fit(daily.loc[tr, daily_cols], daily.loc[tr, "encounters"])

    encf = encounter_model_frame(cl)
    etr, _ = time_split(encf, "date", 0.75)
    admit = Pipeline(
        [
            ("prep", admit_transformer()),
            ("model", RandomForestClassifier(n_estimators=160, min_samples_leaf=4, class_weight="balanced", random_state=0, n_jobs=-1)),
        ]
    )
    admit.fit(encf.loc[etr, admit_cols], encf.loc[etr, "admit_int"])

    pat = patient_feature_frame(cl)
    c_cols = ["age", "log_visits", "admits", "recency_days", "comorbid", "avg_esi"]
    scaler = StandardScaler()
    Xz = scaler.fit_transform(pat[c_cols].fillna(0))
    km = KMeans(n_clusters=4, n_init=10, random_state=0)
    pat = pat.copy()
    pat["segment"] = km.fit_predict(Xz)
    return {
        "demand": demand,
        "admit": admit,
        "daily": daily,
        "te": te,
        "pat": pat,
        "encf": encf,
        "c_cols": c_cols,
        "scaler": scaler,
        "km": km,
    }


models = _models()
daily = models["daily"]
te = models["te"]
last_day = daily.loc[te, "date"].max()
st.caption(f"Service day in the lab (held-out): **{last_day.date()}** — census model never trained on this day.")

night = daily[daily["date"] == last_day].copy()
season_override = st.selectbox("What if season were…", ["as observed", "typical", "flu_wave", "heat"])
sim = night.copy()
if season_override != "as observed":
    sim["season"] = season_override
    sim["flu_wave"] = int(season_override == "flu_wave")
    sim["is_heat"] = int(season_override == "heat")
sim["pred"] = models["demand"].predict(sim[daily_cols])

tab_d, tab_c, tab_a, tab_g, tab_ag = st.tabs(
    ["Census board", "Reagents (NumPy)", "Admit desk", "Phenotypes", "Run agent"]
)

with tab_d:
    show = sim[["site", "season", "encounters", "pred", "enc_lag7"]].rename(
        columns={"encounters": "actual", "pred": "forecast", "enc_lag7": "naive_lag7"}
    )
    show["forecast"] = show["forecast"].round(1)
    st.dataframe(show, hide_index=True)
    st.bar_chart(show.set_index("site")[["actual", "forecast", "naive_lag7"]])
    st.metric("MAE that day", f"{np.mean(np.abs(show['actual'] - show['forecast'])):.1f} encounters / site")
    ok("Modules 05–06: time split, lag features, beat naive, then act.")

with tab_c:
    st.markdown("### Assay pull = condition counts @ protocol matrix")
    st.code("pull = counts @ protocol     # (8,) @ (8, 14) → (14,)", language="python")
    site = st.selectbox("Site to restock", clinic.sites)
    row = sim[sim.site.astype(str) == site].iloc[0]
    tickets = float(row["pred"])
    mix = clinic.atlas.set_index("condition")["popularity"].to_numpy()
    mix = mix / mix.sum()
    counts = tickets * mix
    pull = counts @ clinic.protocol
    on_hand = clinic.inventory[clinic.site_index(site)]
    cover = np.divide(on_hand, np.clip(pull, 1e-6, None))
    board = pd.DataFrame(
        {
            "assay": clinic.biomarker_names,
            "forecast_units": np.round(pull, 1),
            "on_hand": np.round(on_hand, 1),
            "nights_of_cover": np.round(cover, 2),
        }
    ).sort_values("nights_of_cover")
    st.dataframe(board, hide_index=True, width="stretch")
    short = board[board.nights_of_cover < 1.0]
    if len(short):
        st.error("Short tonight: " + ", ".join(short.assay))
    else:
        st.success("No assay is forecast to go negative tonight at this mix.")
    fig = px.bar(board, x="assay", y=["forecast_units", "on_hand"], barmode="group", title="Pull vs on-hand")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), xaxis_tickangle=-35)
    st.plotly_chart(fig, width="stretch")

with tab_a:
    st.markdown("### Score an encounter before it leaves triage")
    c1, c2, c3 = st.columns(3)
    with c1:
        t_site = st.selectbox("Site", clinic.sites, key="ad_site")
        t_arr = st.selectbox("Arrival", ["walk_in", "ambulance", "referral"])
        t_esi = st.select_slider("ESI", options=[1, 2, 3, 4, 5], value=3)
    with c2:
        t_age = st.slider("Age", 1, 95, 64)
        t_spo2 = st.slider("SpO2", 80, 100, 94)
        t_hr = st.slider("HR", 50, 160, 98)
    with c3:
        t_temp = st.slider("Temp °C", 35.5, 40.5, 38.0, 0.1)
        t_wbc = st.slider("WBC", 3.0, 22.0, 11.0, 0.1)
        t_season = st.selectbox("Season", ["typical", "flu_wave", "heat"], key="ad_season")

    ticket = pd.DataFrame(
        [
            {
                "age": t_age,
                "esi_n": int(t_esi),
                "hour": 19,
                "spo2": t_spo2,
                "hr": t_hr,
                "temp_c": t_temp,
                "sbp": 118,
                "wbc": t_wbc,
                "lactate_f": 1.4,
                "troponin_f": 0.02,
                "rush": 1,
                "site": t_site,
                "arrival": t_arr,
                "season": t_season,
            }
        ]
    )
    p = float(models["admit"].predict_proba(ticket[admit_cols])[0, 1])
    st.metric("P(admit)", f"{p:.1%}")
    if p >= 0.45:
        st.warning("High band — start bed search / senior review. Not an automatic admit.")
    else:
        st.success("Lower band — still correlate with vitals. Clinician confirms.")

with tab_g:
    prof = (
        models["pat"]
        .groupby("segment")
        .agg(n=("patient_id", "count"), age=("age", "mean"), visits=("visits", "mean"), admits=("admits", "mean"), comorbid=("comorbid", "mean"))
        .round(1)
    )
    st.dataframe(prof)
    stories = []
    for seg, r in prof.iterrows():
        tag = []
        tag.append("older" if r["age"] > prof["age"].median() else "younger")
        tag.append("high-util" if r["visits"] > prof["visits"].median() else "low-util")
        tag.append("multi-morbid" if r["comorbid"] > prof["comorbid"].median() else "few-comorbid")
        stories.append({"segment": int(seg), "n": int(r["n"]), "story": ", ".join(tag)})
    st.dataframe(pd.DataFrame(stories), hide_index=True)

with tab_ag:
    st.markdown("### Same agent as Module 09, on tonight's board")
    tonight_e = clinic.encounters[clinic.encounters["date"] == last_day]
    eid = st.selectbox("Tonight encounter", tonight_e["encounter_id"].tolist(), key="cap_eid")
    if st.button("Run agent on this case", type="primary"):
        state, trace, ev = run_agent(clinic, models["admit"], int(eid))
        st.dataframe(
            pd.DataFrame([{"step": r.step, "thought": r.thought, "tool": r.tool, "observation": r.observation} for r in trace]),
            hide_index=True,
            width="stretch",
        )
        st.success(state.recommendation)
        st.dataframe(pd.DataFrame(ev), hide_index=True)

st.markdown("---")
st.markdown("### You now have the ML + agent working set")
st.markdown(
    """
1. **Python** for records and streaming.  
2. **NumPy** for panels, ranges, and any dense math.  
3. **Pandas** for the chart, joins, missingness, time.  
4. **EDA** that earns features.  
5. **Leakage-safe** lags and time splits.  
6. **sklearn Pipelines** for census, admit, phenotypes.  
7. **An agent** — tools + planner + stop — with the model as one tool.  
8. **A decision** — beds, assays, escalate — not a screenshot of AUC.
"""
)
