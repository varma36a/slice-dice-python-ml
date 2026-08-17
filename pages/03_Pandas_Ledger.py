import numpy as np
import pandas as pd
import streamlit as st

from pizza.quiz import ask
from pizza.ui import header, inject, load_shop, ok, pitfall, why

inject()
shop = load_shop()
orders = shop.orders
customers = shop.customers

header(
    "Module 03 · Pandas ledger",
    "The shop's source of truth is a DataFrame",
    "Indexing, groupby, merge, missingness, time, reshape. This is 70% of applied ML time.",
)

why(
    "Models fail from silent row loss, duplicated keys, and target leakage in joins — not from picking the wrong kernel. "
    "Pandas is where those bugs live."
)

st.markdown("### Anatomy")
c1, c2, c3, c4 = st.columns(4)
c1.metric("rows", f"{len(orders):,}")
c2.metric("columns", orders.shape[1])
c3.metric("memory", f"{orders.memory_usage(deep=True).sum() / 1e6:.2f} MB")
c4.metric("delivery NA", int(orders["delivery_min"].isna().sum()))
st.dataframe(orders.head(8), hide_index=True)

with st.expander("dtypes — categoricals are free accuracy"):
    st.dataframe(pd.DataFrame({"dtype": orders.dtypes.astype(str), "nulls": orders.isna().sum()}))
    st.caption("store / size / channel / weather are Categorical. That's faster groupby and a hint to OneHotEncoder later.")

st.markdown("### loc / iloc / boolean — three different jobs")
st.code(
    """orders.iloc[:5, :4]                              # position
orders.loc[orders.store.eq("Campus"), "revenue"]  # labels + mask
rush = orders[orders.hour.isin([12, 13, 18, 19])]
""",
    language="python",
)
store = st.multiselect("Stores", shop.stores, default=["Downtown", "Campus"])
channels = st.multiselect("Channels", ["dine_in", "takeout", "delivery"], default=["delivery"])
hour_min, hour_max = st.slider("Hour window", 11, 23, (17, 21))
sub = orders[
    orders["store"].isin(store)
    & orders["channel"].isin(channels)
    & orders["hour"].between(hour_min, hour_max)
]
k1, k2, k3 = st.columns(3)
k1.metric("tickets", f"{len(sub):,}")
k2.metric("revenue", f"${sub.revenue.sum():,.0f}")
k3.metric("late rate", f"{sub.late.mean():.1%}" if sub.late.notna().any() else "n/a")

st.markdown("### groupby · agg · transform")
st.markdown("`agg` reduces. `transform` **keeps the index** so you can add a column like `revenue / store_mean`.")
g = (
    orders.groupby(["store", "pizza"], observed=True)
    .agg(tickets=("order_id", "count"), revenue=("revenue", "sum"), avg_rating=("rating", "mean"))
    .reset_index()
    .sort_values("revenue", ascending=False)
)
st.dataframe(g.head(12), hide_index=True)

orders_t = orders.copy()
orders_t["store_avg"] = orders_t.groupby("store", observed=True)["revenue"].transform("mean")
orders_t["vs_store"] = orders_t["revenue"] / orders_t["store_avg"]
st.caption("Share of tickets that beat their store's mean ticket:")
st.bar_chart(orders_t.groupby("store", observed=True)["vs_store"].apply(lambda s: (s > 1).mean()), color="#E85D04")

st.markdown("### merge — tickets ⨝ loyalty")
merged = orders.merge(customers[["customer_id", "loyalty_tier", "neighborhood"]], on="customer_id", how="left")
rev_tier = merged.groupby("loyalty_tier", observed=True)["revenue"].mean().rename("avg_ticket")
st.bar_chart(rev_tier, color="#FAA307")
st.code(
    """tickets.merge(customers, on="customer_id", how="left")   # keep all tickets
# how='inner' silently drops guests. Count rows before and after every join.
""",
    language="python",
)

st.markdown("### Missing delivery times")
miss = orders["delivery_min"].isna() & (orders["channel"] == "delivery")
st.write(
    f"{int(miss.sum()):,} delivery tickets have a missing time "
    f"({miss.sum() / (orders.channel.eq('delivery').sum()):.1%} of deliveries)."
)
method = st.radio("Impute with", ["drop", "store median", "store + hour median"], horizontal=True)
d = orders.loc[orders.channel.eq("delivery"), ["store", "hour", "delivery_min"]].copy()
if method == "drop":
    filled = d.dropna()
elif method == "store median":
    filled = d.copy()
    filled["delivery_min"] = filled["delivery_min"].fillna(filled.groupby("store", observed=True)["delivery_min"].transform("median"))
else:
    filled = d.copy()
    filled["delivery_min"] = filled["delivery_min"].fillna(
        filled.groupby(["store", "hour"], observed=True)["delivery_min"].transform("median")
    )
m1, m2 = st.columns(2)
m1.metric("rows kept", f"{len(filled):,}")
m2.metric("mean delivery min", f"{filled.delivery_min.mean():.1f}")
pitfall(
    "Imputing the target with the global mean **before** a train/test split leaks the test distribution. "
    "Fit imputers on train only (see Pipelines)."
)

st.markdown("### Time: dt accessor, shift, rolling")
daily_dt = (
    orders.groupby(["date", "store"], observed=True)["revenue"].sum().reset_index()
)
pick = st.selectbox("Store trend", shop.stores, key="pd_store")
s = daily_dt[daily_dt.store.eq(pick)].set_index("date").sort_index()["revenue"]
s = s.to_frame("revenue")
s["lag7"] = s["revenue"].shift(7)
s["roll7"] = s["revenue"].rolling(7, min_periods=3).mean()
st.line_chart(s)

st.markdown("### pivot / melt")
pt = pd.crosstab(orders["store"], orders["pizza"], values=orders["qty"], aggfunc="sum").fillna(0)
st.dataframe(pt.astype(int))
st.caption("crosstab = pivot_table for counts. `melt` is the inverse when sklearn or Altair wants long form.")

ok(
    "After every filter/join: assert `len`. Use `observed=True` on categorical groupby. "
    "Vectorize (`dt`, `.str`, arithmetic) before `.apply`."
)

ask(
    "q3_pandas",
    "You want each ticket's revenue as a % of that store's same-day revenue. Which is correct?",
    [
        "groupby(['store','date'])['revenue'].sum()  then merge — but that's agg; you still need a join back",
        "groupby(['store','date'])['revenue'].transform(lambda s: s / s.sum())",
        "revenue / revenue.mean()  on the whole frame",
    ],
    "groupby(['store','date'])['revenue'].transform(lambda s: s / s.sum())",
    "transform returns a Series aligned to the original index. A global mean mixes Downtown Friday with Campus Tuesday.",
)
