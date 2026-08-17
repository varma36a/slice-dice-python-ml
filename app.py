import streamlit as st

from clinic.topics import CATALOG_COLS, catalog_frame
from clinic.ui import card, header, inject, load_clinic, warn

inject()
clinic = load_clinic()
meta = clinic.meta

header(
    "Course home · 11-year engineer track",
    "Agentic Doctor: Python for ML",
    "One urgent-care network. NumPy, Pandas, EDA, leakage-safe features, sklearn — then a diagnostic agent that calls those models as tools.",
)

warn(
    "Every number in this app is synthetic. This is a software/ML lab, not a diagnostic product and not medical advice."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Encounters", f"{meta['n_encounters']:,}")
c2.metric("Sites", len(clinic.sites))
c3.metric("Condition atlas", len(clinic.condition_names))
c4.metric("Window", f"{meta['days']} days")

st.markdown("### How this is taught")
left, right = st.columns((1.25, 1))
with left:
    st.markdown(
        """
You already write production software. This course does **not** reteach `if`/`for`.
It rebuilds the **ML + agent working set** around one clinic:

1. **Python** — types, collections, functions, OOP, files, datetime, stdlib — on encounter tables.
2. **NumPy** — lab-mean matrix, reference-range masks, protocol @ assay cost.
3. **Pandas** — the chart: groupby, merge patients, missing lactate/troponin, rolling census.
4. **EDA** — flu waves, hypoxia, ESI — features you can see.
5. **Features** — lags, leakage, time splits (the stuff that silently ruins models).
6. **sklearn** — wait regression, admit classification, phenotype clustering, pipelines.
7. **Agent workflow** — tools, a planner, a trace: chart → labs → model → protocol → stop.
8. **Capstone** — triage board: census forecast, reagent pull, admit risk, run the agent.
        """
    )
with right:
    card(
        "The scenario",
        """<b>Northshore Urgent Care</b> is five sites (Downtown, Airport, Campus, Harbor, Suburb).
        Eight conditions. Fourteen biomarkers. ESI, arrival mode, flu waves.
        Every array and DataFrame is generated from one seeded universe — so a NumPy z-score
        lesson and a RandomForest admit model and the agent are talking about the <i>same</i> patients.""",
    )

st.markdown("### Course map")
modules = [
    ("01 · Python for ML", "Types → collections → functions → OOP → files → datetime → stdlib, each with encounter tables."),
    ("02 · NumPy labs", "Lab-mean matrix, broadcasting vs reference ranges, protocol cost, reagent stock."),
    ("03 · Pandas chart", "Groupby, merge comorbidities, missing labs, rolling census."),
    ("04 · EDA & charts", "What the floor looks like before you fit anything."),
    ("05 · Feature engineering", "Time split vs random split vs illegal same-day admit rate."),
    ("06 · Regression", "Site-day census. Beat lag-7 naive."),
    ("07 · Classification", "Will this encounter admit? Threshold as a bed budget."),
    ("08 · Clusters & pipelines", "Patient phenotypes + ColumnTransformer + TimeSeriesSplit."),
    ("09 · Agent workflow", "Tools, planner, trace — ML is a tool, not the product."),
    ("10 · Capstone", "Triage command center: forecast, reagents, admit risk, run the agent."),
]
for title, blurb in modules:
    st.markdown(f"**{title}** — {blurb}")

st.markdown("---")
st.markdown("### Mental model (keep this)")
st.code(
    """encounters: pandas  — messy, labeled, joins, time     (what happened)
labs:       numpy   — dense, reference ranges, matmul (physics of a panel)
features:   pandas→numpy — leakage-safe, split first  (what the model may see)
model:      sklearn Pipeline — preprocess + estimator (a tool)
agent:      plan → tool → observe → stop              (the workflow)
decision:   triage / next test / escalate             (why the clinic cares)
""",
    language="text",
)

st.markdown("### How to use it")
st.markdown(
    """
- Work top to bottom once. **Agent** is the point of the rewrite; **Capstone** is the integration test.
- Every page has a **lab** (widgets) and a **check**.
- Copy the patterns: protocol matrix → user-item matrix; admit model → any binary risk; agent tools → any workflow.
"""
)
st.caption(f"Lab window {meta['start']} → {meta['end']} · seed {meta['seed']}")

st.markdown("---")
st.markdown("### All topics covered on this site")
st.markdown(
    "Index below. **Full example + result for every topic:** sidebar **All Topics → By module**. "
    "Interview drill: **Interview questions**."
)
all_df = catalog_frame()
st.dataframe(all_df[CATALOG_COLS], hide_index=True, width="stretch", height=420)
