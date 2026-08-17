import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from pizza.ml import delivery_model_frame, delivery_transformer, time_split
from pizza.quiz import ask
from pizza.ui import header, inject, load_shop, ok, pitfall, why

inject()
shop = load_shop()
df = delivery_model_frame(shop)

header(
    "Module 07 · Classification",
    "Will this delivery miss the 40-minute SLA?",
    "Precision, recall, thresholds — not 'accuracy'. The shop pays for late pies, not for pretty accuracy.",
)

why(
    "Late deliveries are the minority class. A dummy 'always on time' model looks accurate and is useless. "
    "You pick a threshold from the cost of a false alarm (extra driver) vs a miss (cold pizza, lost rating)."
)

cols = ["store", "pizza", "size", "weather", "hour", "qty", "is_weekend", "rush"]
tr, te = time_split(df, "date", 0.75)
X_train, X_test = df.loc[tr, cols], df.loc[te, cols]
y_train, y_test = df.loc[tr, "late_int"], df.loc[te, "late_int"]

c1, c2, c3 = st.columns(3)
c1.metric("Train late rate", f"{y_train.mean():.1%}")
c2.metric("Test late rate", f"{y_test.mean():.1%}")
c3.metric("Majority-class accuracy", f"{1 - y_test.mean():.1%}")

clf_name = st.selectbox("Estimator", ["LogisticRegression", "RandomForest"])
cw_label = st.selectbox("class_weight", ["none", "balanced"])
cw = None if cw_label == "none" else "balanced"
if clf_name == "LogisticRegression":
    est = LogisticRegression(max_iter=400, class_weight=cw)
else:
    est = RandomForestClassifier(
        n_estimators=180,
        min_samples_leaf=4,
        class_weight=cw,
        random_state=0,
        n_jobs=-1,
    )

pipe = Pipeline([("prep", delivery_transformer()), ("model", est)])
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]

thr = st.slider("Decision threshold (P(late))", 0.05, 0.90, 0.50, 0.01)
pred = (proba >= thr).astype(int)

m1, m2, m3 = st.columns(3)
m1.metric("ROC AUC", f"{roc_auc_score(y_test, proba):.3f}")
m2.metric(f"F1 @ {thr:.2f}", f"{f1_score(y_test, pred):.3f}")
m3.metric("Predicted late share", f"{pred.mean():.1%}")

st.markdown("### Confusion matrix at this threshold")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4.2, 3.6))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    pred,
    display_labels=["on time", "late"],
    ax=ax,
    colorbar=False,
    cmap="Oranges",
)
ax.set_title("Test set")
st.pyplot(fig, width="content")
plt.close(fig)

raw = classification_report(
    y_test,
    pred,
    target_names=["on time", "late"],
    output_dict=True,
    zero_division=0,
)
rows = []
for label in ["on time", "late", "macro avg", "weighted avg"]:
    r = raw[label]
    rows.append(
        {
            "class": label,
            "precision": round(r["precision"], 3),
            "recall": round(r["recall"], 3),
            "f1": round(r["f1-score"], 3),
            "n": int(r["support"]),
        }
    )
rows.append(
    {
        "class": "accuracy",
        "precision": None,
        "recall": None,
        "f1": round(raw["accuracy"], 3),
        "n": int(raw["macro avg"]["support"]),
    }
)
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.markdown("### ROC and PR — threshold is a product decision")
fpr, tpr, _ = roc_curve(y_test, proba)
prec, rec, pthr = precision_recall_curve(y_test, proba)
left, right = st.columns(2)
with left:
    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    fig_r = px.line(roc_df, x="FPR", y="TPR", title="ROC")
    fig_r.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#666"))
    fig_r.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_r, width="stretch")
with right:
    pr_df = pd.DataFrame({"recall": rec, "precision": prec})
    fig_p = px.line(pr_df, x="recall", y="precision", title="Precision–recall")
    fig_p.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_p, width="stretch")

pitfall(
    "Accuracy at 0.5 with `class_weight=None` will look like the majority baseline. "
    "If you only remember one number, remember **recall on late** at the threshold ops will actually use."
)
ok(
    "Tune the threshold on a time-based validation slice. Put the cost story in the notebook: "
    "`cost = c_fp * FP + c_fn * FN`."
)

ask(
    "q7_clf",
    "Ops can spare 8 extra runs a night (false positives). You should:",
    [
        "Maximize accuracy",
        "Raise the threshold until predicted-late volume fits the budget, then read recall",
        "Always use 0.5 because it's the default",
    ],
    "Raise the threshold until predicted-late volume fits the budget, then read recall",
    "The threshold is a capacity constraint. Sweep it. Don't pretend 0.5 is principled.",
)
