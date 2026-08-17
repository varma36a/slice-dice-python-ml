import streamlit as st

from pizza.ui import card, header, inject, load_shop

inject()
shop = load_shop()
meta = shop.meta

header(
    "Course home · 11-year engineer track",
    "Slice & Dice: Python for ML",
    "One pizzeria. Every tool you actually need before (and beside) the models: NumPy, Pandas, EDA, features, sklearn.",
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Orders in the lab", f"{meta['n_orders']:,}")
c2.metric("Stores", len(shop.stores))
c3.metric("Menu items", len(shop.pizza_names))
c4.metric("Window", f"{meta['days']} days")

st.markdown("### How this is taught")
left, right = st.columns((1.25, 1))
with left:
    st.markdown(
        """
You already write production software. This course does **not** reteach `if`/`for`.
It rebuilds the **ML working set** around one shop:

1. **Python** — types, collections, functions, OOP, errors/files, datetime, stdlib, generators — all on the shop tables.
2. **NumPy** — recipes, inventory, broadcasting, views vs copies, vectorized COGS.
3. **Pandas** — the order ledger: groupby, merge, missing data, time, pivots.
4. **EDA** — the plots that change a modeling decision.
5. **Features** — lags, leakage, time splits (the stuff that silently ruins models).
6. **sklearn** — regression, classification, clustering, pipelines, metrics.
7. **Capstone** — Friday-night command center: demand, stock, late-risk, segments.
        """
    )
with right:
    card(
        "The scenario",
        """<b>Slice & Dice</b> is a five-store pizzeria (Downtown, Airport, Campus, Harbor, Suburb).
        Eight pies. Fourteen ingredients. Delivery SLAs, loyalty tiers, weather.
        Every array and DataFrame in the app is generated from one seeded universe — so a NumPy inventory
        lesson and a RandomForest late-delivery model are talking about the <i>same</i> shop.""",
    )

st.markdown("### Course map")

modules = [
    ("01 · Python for ML", "pages/01_Python_for_ML.py", "Types → collections → functions → OOP → files → datetime → stdlib, each with shop tables."),
    ("02 · NumPy kitchen", "pages/02_NumPy_Kitchen.py", "Recipe matrix @ costs, stock masks, broadcasting sizes."),
    ("03 · Pandas ledger", "pages/03_Pandas_Ledger.py", "Groupby, merge, missing delivery times, rolling demand."),
    ("04 · EDA & charts", "pages/04_EDA_and_Charts.py", "What the shop looks like before you fit anything."),
    ("05 · Feature engineering", "pages/05_Feature_Engineering.py", "Lags, rush flags, leakage vs time split."),
    ("06 · Regression", "pages/06_Regression_Demand.py", "Predict store-day tickets. Residuals. Importance."),
    ("07 · Classification", "pages/07_Classification_Late.py", "Will this delivery blow the 40-minute SLA?"),
    ("08 · Clusters & pipelines", "pages/08_Clusters_and_Pipelines.py", "RFM segments + ColumnTransformer."),
    ("09 · Capstone", "pages/09_Capstone_Command_Center.py", "Run the shop: forecast, restock, late risk."),
]

for title, _path, blurb in modules:
    st.markdown(f"**{title}** — {blurb}")

st.markdown("---")
st.markdown("### Mental model (keep this)")
st.code(
    """orders:     pandas  — messy, labeled, joins, time   (what happened)
recipes:    numpy   — dense, same dtype, matmul      (physics of a pie)
features:   pandas→numpy — leakage-safe, split first (what the model may see)
model:      sklearn Pipeline — preprocess + estimator (what you ship)
decision:   stock / labor / SLA / promo              (why the shop cares)
""",
    language="text",
)

st.markdown("### How to use it")
st.markdown(
    """
- Work top to bottom once. Then use **Capstone** as the integration test.
- Every page has a **pizza lab** (widgets) and a **check** (one sharp question).
- Copy the patterns, not the story. The same recipe matrix becomes a user-item matrix on Monday.
"""
)

st.caption(f"Lab window {meta['start']} → {meta['end']} · seed {meta['seed']}")
