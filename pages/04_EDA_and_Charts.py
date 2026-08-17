import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

from pizza.quiz import ask
from pizza.ui import header, inject, load_shop, ok, pitfall, why

inject()
shop = load_shop()
orders = shop.orders
daily = shop.daily

header(
    "Module 04 · EDA & charts",
    "Plot until the next modeling choice is obvious",
    "EDA is not decoration. It is how you discover rush-hour, rain, and Campus-weekend collapse — the actual features.",
)

why(
    "A residual plot or a late-rate heatmap will change your feature list faster than another estimator. "
    "If you cannot see the SLA problem, you will not encode it."
)

st.markdown("### Story 1 — demand is not flat")
fig = px.line(
    daily,
    x="date",
    y="tickets",
    color="store",
    title="Tickets per store-day",
)
fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), legend_title="")
st.plotly_chart(fig, width="stretch")

st.markdown("### Story 2 — lunch and dinner are different products")
hourly = (
    orders.groupby(["hour", "channel"], observed=True)
    .size()
    .rename("tickets")
    .reset_index()
)
fig_h = px.bar(hourly, x="hour", y="tickets", color="channel", barmode="stack", title="Tickets by hour × channel")
fig_h.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_h, width="stretch")

st.markdown("### Story 3 — rain stretches delivery")
deliv = orders[orders.channel.eq("delivery")].dropna(subset=["delivery_min"])
fig_b = px.box(
    deliv,
    x="store",
    y="delivery_min",
    color="weather",
    title="Delivery minutes by store and weather",
    points=False,
)
fig_b.add_hline(y=40, line_dash="dash", annotation_text="SLA 40 min")
fig_b.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_b, width="stretch")

st.markdown("### Story 4 — heatmap of late rate")
late = (
    deliv.groupby(["store", "hour"], observed=True)["late"]
    .mean()
    .astype(float)
    .unstack("hour")
)
fig_hm = px.imshow(
    late,
    aspect="auto",
    color_continuous_scale="YlOrRd",
    title="Late rate (delivery) · store × hour",
    labels=dict(color="late rate"),
)
fig_hm.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_hm, width="stretch")

st.markdown("### Story 5 — menu mix vs margin")
mix = orders.groupby("pizza", observed=True).agg(pies=("qty", "sum"), revenue=("revenue", "sum")).reset_index()
mix = mix.merge(shop.menu[["pizza", "margin_medium", "veg"]], on="pizza")
fig_s = px.scatter(
    mix,
    x="pies",
    y="margin_medium",
    size="revenue",
    color="veg",
    hover_name="pizza",
    title="Volume vs unit margin (bubble = revenue)",
)
fig_s.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_s, width="stretch")

st.markdown("### Correlation — only the numeric slice")
num = daily[["tickets", "revenue", "avg_ticket", "delivery_share", "late_rate", "avg_delivery", "avg_rating"]].corr()
fig_c = px.imshow(num, color_continuous_scale="RdBu", color_continuous_midpoint=0, title="Store-day correlations")
fig_c.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_c, width="stretch")
pitfall(
    "Pearson on a store-day table **pools** Downtown and Suburb. A correlation can be Simpson. Facet by store before you believe it."
)

st.markdown("### Seaborn residual-style: rating vs wait")
sample = deliv.sample(min(2000, len(deliv)), random_state=1)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.2, 3.6))
sns.scatterplot(data=sample, x="delivery_min", y="rating", hue="store", alpha=0.35, ax=ax, s=18)
ax.axvline(40, ls="--", color="#E85D04")
ax.set_title("Rating vs delivery minutes (sample)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
st.pyplot(fig, width="stretch")
plt.close(fig)

ok(
    "Every chart should threaten a feature: hour, weather, store, channel, lag-7, rush flag. "
    "If a plot doesn't change the model spec, it belongs in a slide, not the notebook."
)

ask(
    "q4_eda",
    "The late-rate heatmap lights up Suburb at 19:00 in rain. What do you do first?",
    [
        "Grab XGBoost and let it 'find' the interaction",
        "Add store × hour × rain as an explicit feature (or at least store + hour + rain) and stratify metrics",
        "Drop Suburb because it's an outlier",
    ],
    "Add store × hour × rain as an explicit feature (or at least store + hour + rain) and stratify metrics",
    "You saw the interaction. Encode it. Also report late-rate by store — a global F1 can hide Suburb.",
)
