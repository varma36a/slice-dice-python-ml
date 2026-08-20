import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, silhouette_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from clinic.ml import daily_model_frame, patient_feature_frame, time_split
from clinic.cards import module_explainers
from clinic.quiz import ask
from clinic.ui import header, inject, load_clinic, ok, pitfall, warn, why

inject()
clinic = load_clinic()

header(
    "Module 08 · Clusters & pipelines",
    "Phenotypes for panel design. Pipelines so preprocess can't leak.",
    "KMeans on patient features, then the only sklearn pattern you should ship: Pipeline + ColumnTransformer.",
)
warn("Clusters are unsupervised stories, not diagnoses.")
why("A Pipeline is the difference between a notebook and `joblib.dump`. The agent loads one object, not five pickle files that disagree.")
module_explainers("Clusters & pipelines")

st.markdown("### Patient features")
pat = patient_feature_frame(clinic)
st.dataframe(pat[["patient_id", "age", "payer", "visits", "admits", "comorbid", "recency_days"]].head(8), hide_index=True)

cols = ["age", "log_visits", "admits", "recency_days", "comorbid", "avg_esi"]
X = pat[cols].fillna(0).to_numpy()
Xz = StandardScaler().fit_transform(X)
k = st.slider("k (clusters)", 2, 8, 4)
labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xz)
pat = pat.copy()
pat["segment"] = labels
st.metric("Silhouette", f"{silhouette_score(Xz, labels):.3f}")

xy = PCA(n_components=2, random_state=0).fit_transform(Xz)
plot = pd.DataFrame({"pc1": xy[:, 0], "pc2": xy[:, 1], "segment": labels.astype(str), "age": pat["age"]})
fig = px.scatter(plot, x="pc1", y="pc2", color="segment", size="age", title="Patients in PCA space (size = age)")
fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, width="stretch")

profile = (
    pat.groupby("segment")
    .agg(n=("patient_id", "count"), age=("age", "mean"), visits=("visits", "mean"), admits=("admits", "mean"), comorbid=("comorbid", "mean"), recency=("recency_days", "mean"))
    .round(2)
)
st.markdown("### Segment profile — name these in English")
st.dataframe(profile)
st.caption("Silhouette is a hint. If care management cannot name a cluster, k is wrong.")

st.markdown("---")
st.markdown("### Pipeline + TimeSeriesSplit")
st.code(
    """pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])),
    ("model", Ridge()),
])
pipe.fit(X_train, y_train)
""",
    language="python",
)

feat = daily_model_frame(clinic)
tr, te = time_split(feat, "date", 0.75)
x_cols = ["site", "season", "dow", "is_weekend", "flu_wave", "is_heat", "enc_lag7", "enc_roll7"]
X_train, X_test = feat.loc[tr, x_cols], feat.loc[te, x_cols]
y_train, y_test = feat.loc[tr, "encounters"], feat.loc[te, "encounters"]
prep = ColumnTransformer(
    [
        ("num", StandardScaler(), ["dow", "is_weekend", "flu_wave", "is_heat", "enc_lag7", "enc_roll7"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["site", "season"]),
    ]
)
pipe = Pipeline([("prep", prep), ("model", Ridge())])
if st.checkbox("Run a small α grid (TimeSeriesSplit)", value=True):
    grid = GridSearchCV(pipe, {"model__alpha": [0.3, 1.0, 3.0, 10.0]}, cv=TimeSeriesSplit(n_splits=3), scoring="neg_mean_absolute_error")
    grid.fit(X_train, y_train)
    st.write("best α", grid.best_params_["model__alpha"])
    st.metric("Test MAE", f"{mean_absolute_error(y_test, grid.predict(X_test)):.2f}")
    cv = pd.DataFrame(grid.cv_results_)[["param_model__alpha", "mean_test_score"]]
    cv["MAE"] = -cv["mean_test_score"]
    st.dataframe(cv[["param_model__alpha", "MAE"]].round(3), hide_index=True)

pitfall("Fitting StandardScaler on train+test then splitting is the classic leak. If it isn't inside the Pipeline, you will do this at 1am. The agent will then score leaked features in production.")
ok("One object: `pipe.fit` / `pipe.predict`. Same object the agent imports as `score_admit`.")

ask(
    "q8_pipe",
    "KMeans on raw `age` (1–95) plus `comorbid` (0–4) without scaling. What happens?",
    [
        "k-means is scale-invariant",
        "Euclidean distance is dominated by age; comorbidity is ignored",
        "The algorithm refuses to fit",
    ],
    "Euclidean distance is dominated by age; comorbidity is ignored",
    "Scale for k-means / PCA / anything Euclidean. Trees don't need it; k-means does.",
)
