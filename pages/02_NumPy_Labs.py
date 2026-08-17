import time

import numpy as np
import pandas as pd
import streamlit as st

from clinic.quiz import ask
from clinic.ui import header, inject, load_clinic, ok, pitfall, warn, why

inject()
clinic = load_clinic()

header(
    "Module 02 · NumPy labs",
    "Panels are matrices. Reference ranges broadcast. Cost is a dot product.",
    "If you remember one thing: stop looping over biomarkers. Mask, z-score, multiply.",
)
warn("Synthetic lab means and costs. Not a real assay menu.")
why("Every clinical score (qSOFA, NEWS2-style) is linear algebra plus thresholds. Agents that 'flag labs' are NumPy masks behind a tool name.")

c1, c2, c3 = st.columns(3)
c1.metric("lab_means", f"{clinic.lab_means.shape[0]} × {clinic.lab_means.shape[1]}")
c2.metric("protocol", str(clinic.protocol.shape))
c3.metric("inventory", str(clinic.inventory.shape))

st.code(
    """lab_means   # (8 conditions, 14 biomarkers)
protocol    # (8, 14)  which assays are indicated
ref_low/high# (14,)
cost        # (14,)
panel_cost = protocol @ cost              # (8,)
z = (x - ref_low) / (ref_high - ref_low)  # broadcast (n, 14)
""",
    language="python",
)

st.markdown("### Lab A — protocol cost (matmul)")
show = clinic.atlas[["condition", "panel_cost", "admit_base"]].copy()
st.dataframe(show, hide_index=True)
pick = st.selectbox("Inspect typical lab vector", clinic.condition_names)
vec = clinic.lab_means[clinic.condition_index(pick)]
st.bar_chart(pd.Series(vec, index=clinic.biomarker_names), color="#2A9D8F")

st.markdown("### Lab B — broadcasting vs reference ranges")
z = (vec - clinic.ref_low) / np.clip(clinic.ref_high - clinic.ref_low, 1e-6, None)
flag = (vec < clinic.ref_low) | (vec > clinic.ref_high)
st.dataframe(
    pd.DataFrame(
        {
            "biomarker": clinic.biomarker_names,
            "typical": np.round(vec, 2),
            "ref_low": clinic.ref_low,
            "ref_high": clinic.ref_high,
            "z_vs_band": np.round(z, 2),
            "abnormal": flag,
        }
    ),
    hide_index=True,
    width="stretch",
)
st.code(
    """abnormal = (x < ref_low) | (x > ref_high)   # (14,) mask
# patient matrix X (n, 14) vs ref (14,) broadcasts row-wise
""",
    language="python",
)

st.markdown("### Lab C — reagent reorder mask")
site = st.selectbox("Site", clinic.sites)
par = st.slider("Flag reagent if stock < this % of Downtown mean", 20, 90, 45)
inv = clinic.inventory[clinic.site_index(site)]
thr = (par / 100.0) * clinic.inventory[0].mean()
mask = inv < thr
low = pd.DataFrame({"assay": np.array(clinic.biomarker_names)[mask], "on_hand": np.round(inv[mask], 1)})
st.dataframe(low if len(low) else pd.DataFrame({"assay": ["— all healthy —"], "on_hand": [np.nan]}), hide_index=True)

st.markdown("### Lab D — vectorized vs Python loop")
rng = np.random.default_rng(0)
fake = rng.integers(0, 8, size=20_000)
t0 = time.perf_counter()
acc = np.zeros(clinic.lab_means.shape[1])
for idx in fake:
    acc += clinic.lab_means[idx]
loop_ms = (time.perf_counter() - t0) * 1000
t1 = time.perf_counter()
binc = np.bincount(fake, minlength=8).astype(np.float64)
acc2 = binc @ clinic.lab_means
vec_ms = (time.perf_counter() - t1) * 1000
k1, k2, k3 = st.columns(3)
k1.metric("Python loop", f"{loop_ms:.1f} ms")
k2.metric("bincount + matmul", f"{vec_ms:.2f} ms")
k3.metric("max |diff|", f"{np.max(np.abs(acc - acc2)):.1e}")

pitfall("`lab_means[0, :]` is typically a **view**. Mutating it poisons the atlas the agent retrieves.")
ok("After any slice you will write into, `.copy()`. Treat masks and `@` as default control flow.")

st.markdown("### Axis reductions")
a0, a1 = st.columns(2)
a0.bar_chart(pd.Series(clinic.inventory.mean(axis=0), index=clinic.biomarker_names), color="#2A9D8F")
a1.bar_chart(pd.Series(clinic.inventory.sum(axis=1), index=clinic.sites), color="#7DDECF")

ask(
    "q2_numpy",
    "You have protocol (8, 14) and tonight's condition counts (8,). Assay pull is:",
    ["protocol * counts", "counts @ protocol", "protocol @ counts"],
    "counts @ protocol",
    "(8,) @ (8,14) → (14,). `protocol @ counts` is a shape error.",
)
