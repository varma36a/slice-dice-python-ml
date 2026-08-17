import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from clinic.ml import daily_model_frame, time_split
from clinic.quiz import ask
from clinic.ui import header, inject, load_clinic, ok, pitfall, warn, why

inject()
clinic = load_clinic()
feat = daily_model_frame(clinic)

header(
    "Module 05 · Feature engineering",
    "Split first. Then encode. Never let tomorrow's census into train.",
    "The difference between a demo R² and a floor that trusts you is leakage control. Agents inherit whatever you leaked.",
)
warn("Synthetic census. Same-day admit_rate as a 'feature' is the illegal demo.")
why("Random row splits on time-series census leak tomorrow into today. Target encoding without CV leaks the label.")

st.markdown("### What we already built (site-day)")
st.dataframe(feat.head(8), hide_index=True, width="stretch")
st.markdown(
    """
| Feature | Rule |
|---|---|
| `dow`, `is_weekend` | calendar, known at dawn |
| `flu_wave`, `is_heat` | season — treat as known only if you have a forecast |
| `enc_lag7` | same weekday last week, per site |
| `enc_roll7` | mean of the **past** 7 days (`shift(1).rolling(7)`) |
"""
)

st.markdown("### Leakage demo — same model, two splits")
X_num = feat[["dow", "is_weekend", "flu_wave", "is_heat", "enc_lag7", "enc_roll7"]].to_numpy()
enc_oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_cat = enc_oh.fit_transform(feat[["site"]])
X = np.hstack([X_num, X_cat])
y = feat["encounters"].to_numpy()

Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X, y, test_size=0.25, random_state=0)
r2_rand = r2_score(yte_r, LinearRegression().fit(Xtr_r, ytr_r).predict(Xte_r))

tr_idx, te_idx = time_split(feat, "date", 0.75)
r2_time = r2_score(y[te_idx], LinearRegression().fit(X[tr_idx], y[tr_idx]).predict(X[te_idx]))

X_leak = np.hstack([X, feat[["admit_rate"]].to_numpy()])
r2_leak = r2_score(y[te_idx], LinearRegression().fit(X_leak[tr_idx], y[tr_idx]).predict(X_leak[te_idx]))

c1, c2, c3 = st.columns(3)
c1.metric("Random 75/25 R²", f"{r2_rand:.3f}")
c2.metric("Time split R²", f"{r2_time:.3f}")
c3.metric("Time split + same-day admit_rate (illegal)", f"{r2_leak:.3f}")
pitfall("Same-day admit_rate is an outcome cousin of volume and mix. It looks brilliant at 9am and cannot be used at 9am.")

st.code(
    """# BAD  — encoder sees test frequencies
enc.fit(full["site"])
# GOOD — ColumnTransformer inside a Pipeline, fit on train only
pipe.fit(X_train, y_train)
""",
    language="python",
)

full_mean = clinic.encounters.groupby("condition", observed=True)["admit"].transform("mean")
st.markdown("### Target encoding leak")
st.caption("If `admit` is the target, mean-admit-by-condition on the full table is contaminated.")
st.dataframe(
    clinic.encounters.assign(leaky_cond_admit=full_mean)[["condition", "admit", "leaky_cond_admit"]].head(6),
    hide_index=True,
)

fig = px.scatter(feat, x="enc_lag7", y="encounters", color="site", title="Lag-7 vs today")
fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, width="stretch")
ok("Features are functions of past and known-at-prediction-time columns. Time-split by default for census.")

ask(
    "q5_feat",
    "A colleague one-hot encodes `patient_id` (~1.6k levels) to predict admit. Why is this a bad feature?",
    [
        "sklearn cannot handle that many columns",
        "It overfits identity, won't generalize to new patients, and usually leaks prior admit frequency",
        "One-hot is only for integers",
    ],
    "It overfits identity, won't generalize to new patients, and usually leaks prior admit frequency",
    "High-cardinality IDs are hashes of history. Use train-only aggregated priors with smoothing, not the raw id. Agents that pass MRN into a prompt have the same bug.",
)
