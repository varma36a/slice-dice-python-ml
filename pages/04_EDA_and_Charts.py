import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

from clinic.cards import module_explainers
from clinic.quiz import ask
from clinic.ui import header, inject, load_clinic, ok, pitfall, warn, why

inject()
clinic = load_clinic()
enc = clinic.encounters
daily = clinic.daily

header(
    "Module 04 · EDA & charts",
    "Plot until the next modeling (or agent tool) choice is obvious",
    "EDA is how you discover flu-wave hypoxia and ESI-2 waits — the features the agent should encode, not bury in a net.",
)
warn("Synthetic. Do not treat these curves as epidemiology.")
why("A residual plot or an admit heatmap will change your feature list faster than another estimator.")
module_explainers("EDA")

st.markdown("### Census is not flat")
fig = px.line(daily, x="date", y="encounters", color="site", title="Encounters per site-day")
fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), legend_title="")
st.plotly_chart(fig, width="stretch")

st.markdown("### Arrival mix by hour")
hourly = enc.groupby(["hour", "arrival"], observed=True).size().rename("n").reset_index()
fig_h = px.bar(hourly, x="hour", y="n", color="arrival", barmode="stack", title="Encounters by hour × arrival")
fig_h.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_h, width="stretch")

st.markdown("### Wait vs ESI — the triage SLA")
fig_b = px.box(enc, x="esi", y="wait_min", color="season", title="Wait minutes by ESI and season", points=False)
fig_b.add_hline(y=20, line_dash="dash", annotation_text="ESI≤2 target 20 min")
fig_b.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_b, width="stretch")

st.markdown("### Admit rate heatmap")
adm = enc.groupby(["site", "hour"], observed=True)["admit"].mean().astype(float).unstack("hour")
fig_hm = px.imshow(adm, aspect="auto", color_continuous_scale="Teal", title="Admit rate · site × hour", labels=dict(color="admit"))
fig_hm.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_hm, width="stretch")

st.markdown("### Condition mix vs admit baseline")
mix = enc.groupby("condition", observed=True).agg(n=("encounter_id", "count"), admit_rate=("admit", "mean")).reset_index()
mix = mix.merge(clinic.atlas[["condition", "admit_base", "respiratory"]], on="condition")
fig_s = px.scatter(
    mix, x="n", y="admit_rate", size="n", color="respiratory", hover_name="condition",
    title="Volume vs empirical admit rate",
)
fig_s.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_s, width="stretch")

st.markdown("### Correlation — numeric site-day slice")
num = daily[["encounters", "admit_rate", "avg_wait", "late_triage_rate", "avg_los", "ambulance_share"]].corr()
fig_c = px.imshow(num, color_continuous_scale="RdBu", color_continuous_midpoint=0, title="Site-day correlations")
fig_c.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_c, width="stretch")
pitfall("Pearson on a pooled site-day table can be Simpson. Facet by site before you believe it.")

sample = enc.sample(min(2500, len(enc)), random_state=1)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
sns.scatterplot(data=sample, x="spo2", y="wait_min", hue="admit", alpha=0.35, ax=ax, s=16)
ax.axvline(92, ls="--", color="#E9C46A")
ax.set_title("Wait vs SpO2 (sample)")
st.pyplot(fig, width="stretch")
plt.close(fig)

ok("Every chart should threaten a feature: hour, season, ESI, SpO2, lag-7 census. If it doesn't change the spec, it's a slide.")
ask(
    "q4_eda",
    "Admit heatmap lights up Harbor at 19:00 in a flu wave. First move?",
    [
        "Grab a bigger net and let it find the interaction",
        "Add site + hour + flu_wave (or an explicit interaction) and stratify metrics",
        "Drop Harbor because it's an outlier",
    ],
    "Add site + hour + flu_wave (or an explicit interaction) and stratify metrics",
    "You saw the interaction. Encode it. Report admit rate by site — a global AUC can hide Harbor.",
)
