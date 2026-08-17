import pandas as pd
import streamlit as st

from clinic.quiz import ask
from clinic.ui import header, inject, load_clinic, ok, pitfall, warn, why

inject()
clinic = load_clinic()
enc = clinic.encounters
patients = clinic.patients

header(
    "Module 03 · Pandas chart",
    "The source of truth is a DataFrame of encounters",
    "Indexing, groupby, merge, missing labs, time, reshape. This is still 70% of applied ML time.",
)
warn("Synthetic charts. Missing lactate/troponin are planted so you practice imputation — not real lab ops.")
why("Models fail from silent row loss and leakage in joins. Agents that `get_chart` are doing a filtered loc.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("rows", f"{len(enc):,}")
c2.metric("columns", enc.shape[1])
c3.metric("memory", f"{enc.memory_usage(deep=True).sum() / 1e6:.2f} MB")
c4.metric("lactate NA", int(enc["lactate"].isna().sum()))
st.dataframe(enc.head(8), hide_index=True, width="stretch")

with st.expander("dtypes"):
    st.dataframe(pd.DataFrame({"dtype": enc.dtypes.astype(str), "nulls": enc.isna().sum()}))

st.markdown("### loc / boolean")
st.code(
    """enc.loc[enc.site.eq("Campus") & enc.esi.isin([1, 2]), "spo2"]
hypoxia = enc[enc.spo2 < 92]
""",
    language="python",
)
sites = st.multiselect("Sites", clinic.sites, default=["Downtown", "Campus"])
arrivals = st.multiselect("Arrival", ["walk_in", "ambulance", "referral"], default=["ambulance"])
esi_keep = st.multiselect("ESI", [1, 2, 3, 4, 5], default=[1, 2, 3])
sub = enc[enc["site"].isin(sites) & enc["arrival"].isin(arrivals) & enc["esi"].isin(esi_keep)]
k1, k2, k3 = st.columns(3)
k1.metric("encounters", f"{len(sub):,}")
k2.metric("admit rate", f"{sub.admit.mean():.1%}" if len(sub) else "n/a")
k3.metric("mean wait", f"{sub.wait_min.mean():.1f} min" if len(sub) else "n/a")

st.markdown("### groupby · agg · transform")
g = (
    enc.groupby(["site", "condition"], observed=True)
    .agg(n=("encounter_id", "count"), admit_rate=("admit", "mean"), avg_wait=("wait_min", "mean"))
    .reset_index()
    .sort_values("n", ascending=False)
)
st.dataframe(g.head(12), hide_index=True, width="stretch")
enc_t = enc.copy()
enc_t["site_wait"] = enc_t.groupby("site", observed=True)["wait_min"].transform("mean")
enc_t["vs_site"] = enc_t["wait_min"] / enc_t["site_wait"]
st.caption("Share of encounters slower than their site mean wait")
st.bar_chart(enc_t.groupby("site", observed=True)["vs_site"].apply(lambda s: (s > 1).mean()), color="#2A9D8F")

st.markdown("### merge — encounters ⨝ comorbidities")
merged = enc.merge(patients[["patient_id", "payer", "copd", "cad", "dm"]], on="patient_id", how="left")
st.bar_chart(merged.groupby("payer", observed=True)["admit"].mean(), color="#7DDECF")
st.code(
    """enc.merge(patients, on="patient_id", how="left")
# inner silently drops first-time guests / unmatched MRNs. Count rows before and after.
""",
    language="python",
)

st.markdown("### Missing lactate / troponin")
st.write(
    f"lactate NA {int(enc.lactate.isna().sum()):,} · troponin NA {int(enc.troponin.isna().sum()):,}"
)
method = st.radio("Impute lactate with", ["drop", "site median", "site + esi median"], horizontal=True)
d = enc[["site", "esi", "lactate"]].copy()
if method == "drop":
    filled = d.dropna()
elif method == "site median":
    filled = d.copy()
    filled["lactate"] = filled["lactate"].fillna(filled.groupby("site", observed=True)["lactate"].transform("median"))
else:
    filled = d.copy()
    filled["lactate"] = filled["lactate"].fillna(
        filled.groupby(["site", "esi"], observed=True)["lactate"].transform("median")
    )
m1, m2 = st.columns(2)
m1.metric("rows kept", f"{len(filled):,}")
m2.metric("mean lactate", f"{filled.lactate.mean():.2f}")
pitfall("Imputing the target (or a leaky cousin) on the full frame before a split leaks the test distribution. Fit imputers on train only.")

st.markdown("### Time: shift / rolling census")
pick = st.selectbox("Site trend", clinic.sites)
s = clinic.daily[clinic.daily.site.eq(pick)].set_index("date").sort_index()[["encounters"]]
s["lag7"] = s["encounters"].shift(7)
s["roll7"] = s["encounters"].rolling(7, min_periods=3).mean()
st.line_chart(s)

st.markdown("### crosstab site × condition")
pt = pd.crosstab(enc["site"], enc["condition"])
st.dataframe(pt)
ok("After every filter/join: assert `len`. `observed=True` on categorical groupby. Vectorize before `.apply`.")

ask(
    "q3_pandas",
    "Each encounter's wait as a % of that site's same-day wait total. Correct tool?",
    [
        "groupby(['site','date'])['wait_min'].sum() only",
        "groupby(['site','date'])['wait_min'].transform(lambda s: s / s.sum())",
        "wait_min / wait_min.mean() on the whole frame",
    ],
    "groupby(['site','date'])['wait_min'].transform(lambda s: s / s.sum())",
    "transform stays aligned to the original index. A global mean mixes Downtown Monday with Campus Saturday.",
)
