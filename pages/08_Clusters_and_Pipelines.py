import numpy as np
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

from pizza.ml import customer_feature_frame, daily_model_frame, time_split
from pizza.quiz import ask
from pizza.ui import header, inject, load_shop, ok, pitfall, why

inject()
shop = load_shop()

header(
    "Module 08 · Clusters & pipelines",
    "Segments for marketing. Pipelines so preprocess can't leak.",
    "KMeans on RFM-style guest features, then the only sklearn pattern you should ship: Pipeline + ColumnTransformer.",
)

why(
    "A Pipeline is the difference between a notebook and a thing you can `joblib.dump`. "
    "Clustering is unsupervised — you validate with a **business story**, not with accuracy."
)

st.markdown("### Customer features (RFM-ish)")
cust = customer_feature_frame(shop)
st.dataframe(cust[["customer_id", "loyalty_tier", "orders", "spend", "recency_days", "veg_share"]].head(8), hide_index=True)

cols = ["log_spend", "log_orders", "recency_days", "avg_rating", "veg_share", "delivery_share"]
X = cust[cols].fillna(0).to_numpy()
Xz = StandardScaler().fit_transform(X)

k = st.slider("k (clusters)", 2, 8, 4)
km = KMeans(n_clusters=k, n_init=10, random_state=0)
labels = km.fit_predict(Xz)
cust["segment"] = labels
sil = silhouette_score(Xz, labels) if len(set(labels)) > 1 else 0.0
st.metric("Silhouette", f"{sil:.3f}")

pca = PCA(n_components=2, random_state=0)
xy = pca.fit_transform(Xz)
plot = pd.DataFrame({"pc1": xy[:, 0], "pc2": xy[:, 1], "segment": labels.astype(str), "spend": cust["spend"]})
fig = px.scatter(plot, x="pc1", y="pc2", color="segment", size="spend", title="Guests in PCA space (size = spend)")
fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, width="stretch")

profile = cust.groupby("segment").agg(
    n=("customer_id", "count"),
    avg_orders=("orders", "mean"),
    avg_spend=("spend", "mean"),
    recency=("recency_days", "mean"),
    veg=("veg_share", "mean"),
    delivery=("delivery_share", "mean"),
    rating=("avg_rating", "mean"),
).round(2)
st.markdown("### Segment profile — name these in English")
st.dataframe(profile)

names_guess = {
    0: "Write the story: high recency + low spend → lapsed. High veg_share → campus greens. High delivery + spend → suburb Friday.",
}
st.caption("Silhouette is a hint, not a KPI. If marketing cannot name a cluster, k is wrong.")

st.markdown("---")
st.markdown("### Pipeline + ColumnTransformer (the shippable unit)")
st.code(
    """pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])),
    ("model", Ridge()),
])
pipe.fit(X_train, y_train)          # scaler stats from TRAIN only
joblib.dump(pipe, "demand.joblib")
""",
    language="python",
)

feat = daily_model_frame(shop)
tr, te = time_split(feat, "date", 0.75)
x_cols = ["store", "weather", "dow", "is_weekend", "is_rain", "is_heat", "tickets_lag7", "tickets_roll7"]
X_train, X_test = feat.loc[tr, x_cols], feat.loc[te, x_cols]
y_train, y_test = feat.loc[tr, "tickets"], feat.loc[te, "tickets"]

prep = ColumnTransformer(
    [
        ("num", StandardScaler(), ["dow", "is_weekend", "is_rain", "is_heat", "tickets_lag7", "tickets_roll7"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["store", "weather"]),
    ]
)
pipe = Pipeline([("prep", prep), ("model", Ridge())])

st.markdown("### GridSearch with a **TimeSeriesSplit**")
st.markdown("Random K-fold on demand shuffles Fridays into train and test. Use `TimeSeriesSplit`.")
if st.checkbox("Run a small α grid (fits 12 models)", value=True):
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(pipe, {"model__alpha": [0.3, 1.0, 3.0, 10.0]}, cv=tscv, scoring="neg_mean_absolute_error")
    grid.fit(X_train, y_train)
    pred = grid.predict(X_test)
    st.write("best α", grid.best_params_["model__alpha"])
    st.metric("Test MAE", f"{mean_absolute_error(y_test, pred):.2f}")
    cv = pd.DataFrame(grid.cv_results_)[["param_model__alpha", "mean_test_score", "std_test_score"]]
    cv["MAE"] = -cv["mean_test_score"]
    st.dataframe(cv[["param_model__alpha", "MAE", "std_test_score"]].round(3), hide_index=True)

pitfall(
    "Fitting `StandardScaler` on X_train **and** X_test, then splitting, is the classic leak. "
    "If it isn't inside the Pipeline, you will do this at 1am."
)
ok("One object: `pipe.fit` / `pipe.predict`. Same code in the capstone and in production.")

ask(
    "q8_pipe",
    "KMeans on raw `spend` (tens of dollars to thousands) plus `veg_share` (0–1) without scaling. What happens?",
    [
        "k-means is scale-invariant, so nothing",
        "Euclidean distance is dominated by spend; veg_share is ignored",
        "The algorithm refuses to fit",
    ],
    "Euclidean distance is dominated by spend; veg_share is ignored",
    "Always scale for k-means / PCA / anything with Euclidean geometry. Trees don't need it; k-means does.",
)
