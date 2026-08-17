import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from pizza.ml import daily_model_frame, time_split
from pizza.quiz import ask
from pizza.ui import header, inject, load_shop, ok, pitfall, why

inject()
shop = load_shop()
feat = daily_model_frame(shop)

header(
    "Module 05 · Feature engineering",
    "Split first. Then encode. Never let the future into train.",
    "The difference between a demo R² and a shop that trusts you is leakage control.",
)

why(
    "Random row splits on time-series demand leak tomorrow into today. Target encoding without CV leaks the label. "
    "Imputing with the full-frame mean leaks test. Features are a data-leakage discipline."
)

st.markdown("### What we already built (store-day)")
st.dataframe(feat.head(8), hide_index=True)
st.markdown(
    """
| Feature | Rule |
|---|---|
| `dow`, `is_weekend` | calendar, known at dawn |
| `is_rain`, `is_heat` | weather — treat as **known** only if you have a forecast; here it's observed |
| `tickets_lag7` | same weekday last week — **shift(7)** per store |
| `tickets_roll7` | mean of the **past** 7 days (`shift(1).rolling(7)`) |
"""
)

st.markdown("### Leakage demo — same model, two splits")
st.markdown("Predict `tickets` from store + weather + lag features. Compare a **random** split vs a **time** split.")

X_num = feat[["dow", "is_weekend", "is_rain", "is_heat", "tickets_lag7", "tickets_roll7"]].to_numpy()
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_cat = enc.fit_transform(feat[["store"]])
X = np.hstack([X_num, X_cat])
y = feat["tickets"].to_numpy()

Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X, y, test_size=0.25, random_state=0)
m_rand = LinearRegression().fit(Xtr_r, ytr_r)
r2_rand = r2_score(yte_r, m_rand.predict(Xte_r))

tr_idx, te_idx = time_split(feat, "date", 0.75)
m_time = LinearRegression().fit(X[tr_idx], y[tr_idx])
r2_time = r2_score(y[te_idx], m_time.predict(X[te_idx]))

# Illegal: include same-day revenue as a "feature"
X_leak = np.hstack([X, feat[["revenue"]].to_numpy()])
m_leak = LinearRegression().fit(X_leak[tr_idx], y[tr_idx])
r2_leak = r2_score(y[te_idx], m_leak.predict(X_leak[te_idx]))

c1, c2, c3 = st.columns(3)
c1.metric("Random 75/25 R²", f"{r2_rand:.3f}")
c2.metric("Time split R²", f"{r2_time:.3f}")
c3.metric("Time split + same-day revenue (illegal)", f"{r2_leak:.3f}")
pitfall(
    "Same-day `revenue` is almost a deterministic function of tickets × mix. The illegal model looks brilliant and cannot be used at 9am."
)

st.markdown("### Encoding without leaking")
st.code(
    """# BAD  — encoder sees test categories AND test frequencies
enc.fit(full["store"])

# GOOD — ColumnTransformer inside a Pipeline, fit on train only
pipe.fit(X_train, y_train)
pipe.transform(X_test)
""",
    language="python",
)

st.markdown("### Target encoding (the tempting leak)")
st.markdown(
    "Mean tickets by pizza on the **full** order table includes the row's own target. "
    "If you then predict late/not-late using that mean, you smuggled the label."
)
full_mean = shop.orders.groupby("pizza", observed=True)["rating"].transform("mean")
st.caption("If `rating` is the target, this column is contaminated. Use out-of-fold means or a prior from train only.")

show = shop.orders.assign(leaky_pizza_rating=full_mean)[["pizza", "rating", "leaky_pizza_rating"]].head(6)
st.dataframe(show, hide_index=True)

st.markdown("### Interaction you actually saw in EDA")
feat2 = feat.copy()
feat2["suburb_rain"] = ((feat2.store.astype(str) == "Suburb") & (feat2.is_rain == 1)).astype(int)
st.line_chart(
    feat2.groupby("date")[["tickets"]].sum(),
    color="#E85D04",
)
ok(
    "Write features as functions of **past and known-at-prediction-time** columns. "
    "Unit-test one store's lag-7 against a hand calculation. Time-split by default for demand."
)

fig = px.scatter(
    feat,
    x="tickets_lag7",
    y="tickets",
    color="store",
    title="Lag-7 vs today (should be a useful but imperfect diagonal)",
)
fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, width="stretch")

ask(
    "q5_feat",
    "A colleague one-hot encodes `customer_id` (1.4k levels) to predict late delivery. Why is this a bad feature?",
    [
        "sklearn cannot handle that many columns",
        "It overfits identity, won't generalize to new guests, and usually leaks frequency of being late",
        "One-hot is only for integers",
    ],
    "It overfits identity, won't generalize to new guests, and usually leaks frequency of being late",
    "High-cardinality IDs are hashes of history. Use aggregated train-only features (prior late rate with smoothing), not the raw id.",
)
