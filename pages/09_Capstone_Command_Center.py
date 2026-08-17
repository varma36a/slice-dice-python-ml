"""Friday-night command center — every module shows up on one screen."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pizza.ml import (
    customer_feature_frame,
    daily_model_frame,
    daily_transformer,
    delivery_model_frame,
    delivery_transformer,
    time_split,
)
from pizza.ui import header, inject, load_shop, ok, why

inject()
shop = load_shop()

header(
    "Module 09 · Capstone",
    "Friday night command center",
    "Demand · commissary pull · SLA risk · guest segments. This is the exam: can the pieces talk to each other?",
)

why(
    "A real ML system is not a model. It is a decision: how much dough to proof, which store needs a driver, "
    "who gets the gold-tier SMS. The math is NumPy + Pandas + a fitted Pipeline."
)


@st.cache_resource(show_spinner="Training shop models (cached)…")
def _models():
    shop_ = load_shop()
    daily = daily_model_frame(shop_)
    tr, te = time_split(daily, "date", 0.75)
    x_cols = ["store", "weather", "dow", "is_weekend", "is_rain", "is_heat", "tickets_lag7", "tickets_roll7"]
    demand = Pipeline(
        [
            ("prep", daily_transformer()),
            ("model", RandomForestRegressor(n_estimators=220, min_samples_leaf=3, random_state=0, n_jobs=-1)),
        ]
    )
    demand.fit(daily.loc[tr, x_cols], daily.loc[tr, "tickets"])

    deliv = delivery_model_frame(shop_)
    dtr, dte = time_split(deliv, "date", 0.75)
    d_cols = ["store", "pizza", "size", "weather", "hour", "qty", "is_weekend", "rush"]
    late = Pipeline(
        [
            ("prep", delivery_transformer()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=180, min_samples_leaf=4, class_weight="balanced", random_state=0, n_jobs=-1
                ),
            ),
        ]
    )
    late.fit(deliv.loc[dtr, d_cols], deliv.loc[dtr, "late_int"])

    cust = customer_feature_frame(shop_)
    c_cols = ["log_spend", "log_orders", "recency_days", "avg_rating", "veg_share", "delivery_share"]
    scaler = StandardScaler()
    Xz = scaler.fit_transform(cust[c_cols].fillna(0))
    km = KMeans(n_clusters=4, n_init=10, random_state=0)
    labels = km.fit_predict(Xz)
    cust = cust.copy()
    cust["segment"] = labels
    return {
        "demand": demand,
        "late": late,
        "kmeans": km,
        "scaler": scaler,
        "c_cols": c_cols,
        "x_cols": x_cols,
        "d_cols": d_cols,
        "daily": daily,
        "te": te,
        "cust": cust,
        "segment_profile": cust.groupby("segment").agg(
            n=("customer_id", "count"),
            spend=("spend", "mean"),
            orders=("orders", "mean"),
            recency=("recency_days", "mean"),
            veg=("veg_share", "mean"),
        ),
    }


models = _models()
daily = models["daily"]
te = models["te"]
last_day = daily.loc[te, "date"].max()
st.caption(f"Service night in the lab (held-out): **{last_day.date()}** — models never trained on this day.")

night = daily[daily["date"] == last_day].copy()
weather_override = st.selectbox("What if weather were…", ["as observed", "clear", "rain", "heat"])
sim = night.copy()
if weather_override != "as observed":
    sim["weather"] = weather_override
    sim["is_rain"] = int(weather_override == "rain")
    sim["is_heat"] = int(weather_override == "heat")
sim["pred"] = models["demand"].predict(sim[models["x_cols"]])

tab_d, tab_c, tab_l, tab_g = st.tabs(["Demand board", "Commissary (NumPy)", "SLA desk", "Guest segments"])

with tab_d:
    st.markdown("### Predicted tickets vs actual (held-out night)")
    show = sim[["store", "weather", "tickets", "pred", "tickets_lag7"]].rename(
        columns={"tickets": "actual", "pred": "forecast", "tickets_lag7": "naive_lag7"}
    )
    show["forecast"] = show["forecast"].round(1)
    st.dataframe(show, hide_index=True)
    chart = show.set_index("store")[["actual", "forecast", "naive_lag7"]]
    st.bar_chart(chart)
    mae = np.mean(np.abs(show["actual"] - show["forecast"]))
    st.metric("MAE that night", f"{mae:.1f} tickets / store")
    ok("This is the Module 05–06 contract: time split, lag features, beat naive, then **act**.")

with tab_c:
    st.markdown("### Ingredient pull = pizza counts @ recipe matrix")
    st.code("pull = counts @ recipes     # (8,) @ (8, 14) → (14,)", language="python")
    store = st.selectbox("Store to restock", shop.stores)
    row = sim[sim.store.astype(str) == store].iloc[0]
    tickets = float(row["pred"])
    mix = shop.menu.set_index("pizza")["popularity"].to_numpy()
    mix = mix / mix.sum()
    # Average ~1.2 pies per ticket from the ledger
    pies_per_ticket = float(shop.orders.qty.mean())
    counts = tickets * pies_per_ticket * mix  # (8,)
    pull = counts @ shop.recipes  # (14,)
    on_hand = shop.inventory[shop.store_index(store)]
    cover = np.divide(on_hand, np.clip(pull, 1e-6, None))
    board = pd.DataFrame(
        {
            "ingredient": shop.ingredient_names,
            "forecast_units": np.round(pull, 1),
            "on_hand": np.round(on_hand, 1),
            "nights_of_cover": np.round(cover, 2),
        }
    ).sort_values("nights_of_cover")
    st.dataframe(board, hide_index=True)
    short = board[board.nights_of_cover < 1.0]
    if len(short):
        st.error("Short tonight: " + ", ".join(short.ingredient))
    else:
        st.success("No ingredient is forecast to go negative tonight at this mix.")
    fig = px.bar(board, x="ingredient", y=["forecast_units", "on_hand"], barmode="group", title="Pull vs on-hand")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), xaxis_tickangle=-35)
    st.plotly_chart(fig, width="stretch")

with tab_l:
    st.markdown("### Score a ticket before it leaves the make line")
    c1, c2, c3 = st.columns(3)
    with c1:
        t_store = st.selectbox("Store", shop.stores, key="late_store")
        t_pizza = st.selectbox("Pizza", shop.pizza_names, key="late_pizza")
        t_size = st.selectbox("Size", ["S", "M", "L", "XL"], index=2)
    with c2:
        t_weather = st.selectbox("Weather", ["clear", "rain", "heat"], key="late_w")
        t_hour = st.slider("Hour", 11, 23, 19)
        t_qty = st.slider("Qty", 1, 3, 1)
    with c3:
        t_weekend = st.checkbox("Weekend", value=True)
        t_rush = int(t_hour in (12, 13, 18, 19, 20))
        st.write("Rush hour flag:", bool(t_rush))

    ticket = pd.DataFrame(
        [
            {
                "store": t_store,
                "pizza": t_pizza,
                "size": t_size,
                "weather": t_weather,
                "hour": t_hour,
                "qty": t_qty,
                "is_weekend": int(t_weekend),
                "rush": t_rush,
            }
        ]
    )
    p_late = float(models["late"].predict_proba(ticket[models["d_cols"]])[0, 1])
    st.metric("P(late > 40 min)", f"{p_late:.1%}")
    if p_late >= 0.45:
        st.warning("Dispatch an extra runner or quote 50–55 min. Don't surprise Harbor in the rain.")
    else:
        st.success("SLA looks holdable. Keep the standard quote.")
    st.caption("Classifier from Module 07, trained on deliveries before this night. Threshold is an ops choice.")

with tab_g:
    st.markdown("### Four guest segments (k-means on scaled RFM-ish features)")
    prof = models["segment_profile"].round(1)
    # Stable names from seed=0 k=4 — describe from the table, not hardcoded fantasy if empty
    st.dataframe(prof)
    stories = []
    for seg, r in prof.iterrows():
        tag = []
        if r["recency"] > prof["recency"].median():
            tag.append("lapsing")
        else:
            tag.append("active")
        if r["spend"] > prof["spend"].median():
            tag.append("high spend")
        else:
            tag.append("light spend")
        if r["veg"] > 0.45:
            tag.append("veg-leaning")
        stories.append({"segment": int(seg), "n": int(r["n"]), "story": ", ".join(tag)})
    st.dataframe(pd.DataFrame(stories), hide_index=True)

    cid = st.selectbox("Look up a guest", shop.customers.customer_id.head(40))
    row = models["cust"].loc[models["cust"].customer_id == cid]
    if len(row):
        r = row.iloc[0]
        st.write(
            {
                "segment": int(r["segment"]),
                "orders": int(r["orders"]),
                "spend": round(float(r["spend"]), 1),
                "recency_days": int(r["recency_days"]),
                "loyalty": str(shop.customers.set_index("customer_id").loc[cid, "loyalty_tier"]),
            }
        )
    else:
        st.info("This guest never ordered in the window — cold start. Don't one-hot their id; use a prior.")

st.markdown("---")
st.markdown("### You now have the ML working set")
st.markdown(
    """
1. **Python** for records and streaming.  
2. **NumPy** for recipes, inventory, and any dense math.  
3. **Pandas** for the ledger, joins, missingness, time.  
4. **EDA** that earns features.  
5. **Leakage-safe** lags and time splits.  
6. **sklearn Pipelines** for regression, classification, clustering.  
7. **A decision** — stock, SLA, segment — not a screenshot of R².
"""
)
