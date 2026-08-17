import time

import numpy as np
import pandas as pd
import streamlit as st

from pizza.quiz import ask
from pizza.ui import header, inject, load_shop, ok, pitfall, why

inject()
shop = load_shop()

header(
    "Module 02 · NumPy kitchen",
    "Recipes are matrices. Inventory is a matrix. Cost is a dot product.",
    "If you remember one thing: stop writing Python loops over pies. Broadcast, mask, multiply.",
)

why(
    "Every estimator in sklearn is linear algebra plus a loop over samples in C. "
    "If you can't shape `(n_pizzas, n_ingredients)` and `(n_ingredients,)`, you will fight the API for the rest of the course."
)

st.markdown("### Shapes in this shop")
c1, c2, c3 = st.columns(3)
c1.metric("recipes", f"{shop.recipes.shape[0]} × {shop.recipes.shape[1]}")
c2.metric("inventory", f"{shop.inventory.shape[0]} × {shop.inventory.shape[1]}")
c3.metric("ingredient cost", f"{shop.ingredient_cost.shape}")

st.code(
    """recipes     # (8 pizzas, 14 ingredients)  float64
inventory   # (5 stores, 14 ingredients)
cost        # (14,)
cogs = recipes @ cost                         # (8,)  COGS per medium pie
need = orders_vec @ recipes                   # (14,) ingredients for a night
""",
    language="python",
)

st.markdown("### Lab A — COGS and margin (matmul)")
cogs = shop.recipes @ shop.ingredient_cost
margin = shop.menu["base_price"].to_numpy() - cogs
show = shop.menu[["pizza", "base_price"]].copy()
show["cogs"] = np.round(cogs, 2)
show["margin"] = np.round(margin, 2)
show["margin_pct"] = np.round(margin / shop.menu["base_price"].to_numpy() * 100, 1)
st.dataframe(show, hide_index=True)

pick = st.selectbox("Inspect recipe vector", shop.pizza_names)
vec = shop.recipes[shop.pizza_index(pick)]
st.bar_chart(pd.Series(vec, index=shop.ingredient_names), color="#FAA307")

st.markdown("### Lab B — broadcasting size multipliers")
st.markdown(
    "A medium recipe times a **column** of size multipliers gives a full price-card of ingredient use. "
    "No nested `for size in sizes`."
)
mult = np.array([0.75, 1.00, 1.30, 1.60])[:, None]  # (4,1)
sized = shop.recipes[shop.pizza_index(pick)] * mult  # (4,14) via broadcast
st.dataframe(
    pd.DataFrame(sized, index=["S", "M", "L", "XL"], columns=shop.ingredient_names).round(2)
)
st.code(
    """mult = np.array([0.75, 1.00, 1.30, 1.60])[:, None]   # (4, 1)
sized = recipe * mult                                  # (4, 14)
# rule: trailing dims 1 or equal. (14,) vs (4,1) → (4,14)
""",
    language="python",
)

st.markdown("### Lab C — boolean masks (reorder list)")
store = st.selectbox("Store", shop.stores)
par = st.slider("Reorder when stock < this % of Downtown's mean ingredient", 20, 90, 45)
inv = shop.inventory[shop.store_index(store)]
threshold = (par / 100.0) * shop.inventory[0].mean()
mask = inv < threshold
low = pd.DataFrame(
    {"ingredient": np.array(shop.ingredient_names)[mask], "on_hand": np.round(inv[mask], 1)}
)
st.dataframe(low if len(low) else pd.DataFrame({"ingredient": ["— all healthy —"], "on_hand": [np.nan]}), hide_index=True)
st.caption(f"{mask.sum()} of {mask.size} ingredients flagged. Masks are the NumPy equivalent of SQL WHERE.")

st.markdown("### Lab D — vectorized vs Python loop")
rng = np.random.default_rng(0)
fake_orders = rng.integers(0, 8, size=20_000)
t0 = time.perf_counter()
acc = np.zeros(shop.recipes.shape[1])
for idx in fake_orders:
    acc += shop.recipes[idx]
loop_ms = (time.perf_counter() - t0) * 1000
t1 = time.perf_counter()
binc = np.bincount(fake_orders, minlength=8).astype(np.float64)
acc2 = binc @ shop.recipes
vec_ms = (time.perf_counter() - t1) * 1000
k1, k2, k3 = st.columns(3)
k1.metric("Python loop", f"{loop_ms:.1f} ms")
k2.metric("bincount + matmul", f"{vec_ms:.2f} ms")
k3.metric("max |diff|", f"{np.max(np.abs(acc - acc2)):.1e}")

st.markdown("### View vs copy (this one bites)")
st.code(
    """row = recipes[0]          # view (usually)
row *= 0                  # mutates recipes[0]  ← silent inventory bug
safe = recipes[0].copy()
""",
    language="python",
)
pitfall(
    "`recipes[0, :]` is typically a **view**. Fancy indexing (`recipes[[0, 2]]`) copies. "
    "If a feature function mutates a view, your train set changes under you."
)
ok(
    "After any slice you will write into, `.copy()`. Treat `@`, `*`, `np.where`, and masks as the default control flow."
)

st.markdown("### Axis reductions")
st.write("Mean stock by ingredient across stores (`axis=0`) vs total stock by store (`axis=1`).")
a0, a1 = st.columns(2)
a0.bar_chart(pd.Series(shop.inventory.mean(axis=0), index=shop.ingredient_names), color="#E85D04")
a1.bar_chart(pd.Series(shop.inventory.sum(axis=1), index=shop.stores), color="#FAA307")

ask(
    "q2_numpy",
    "You have recipes (8, 14) and a night's pizza counts (8,). Ingredient pull is:",
    [
        "recipes * counts",
        "counts @ recipes",
        "recipes @ counts",
    ],
    "counts @ recipes",
    "(8,) @ (8,14) → (14,). `recipes @ counts` is a shape error. Elementwise `*` would also fail without broadcasting a column.",
)
