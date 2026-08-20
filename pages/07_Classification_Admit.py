import matplotlib.pyplot as plt
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

from clinic.ml import admit_transformer, encounter_model_frame, time_split
from clinic.cards import module_explainers
from clinic.quiz import ask
from clinic.ui import header, inject, load_clinic, ok, pitfall, warn, why

inject()
clinic = load_clinic()
df = encounter_model_frame(clinic)

header(
    "Module 07 · Classification",
    "Will this encounter admit?",
    "Precision, recall, thresholds — not accuracy. Beds are the budget. The agent will call this model as `score_admit`.",
)
warn("Synthetic labels. Not a real admit predictor. Not for clinical use.")
why("Admit is the minority class. A dummy 'always discharge' model looks accurate and is useless. Threshold is a bed constraint.")
module_explainers("Classification")

cols = ["age", "esi_n", "hour", "spo2", "hr", "temp_c", "sbp", "wbc", "lactate_f", "troponin_f", "rush", "site", "arrival", "season"]
tr, te = time_split(df, "date", 0.75)
X_train, X_test = df.loc[tr, cols], df.loc[te, cols]
y_train, y_test = df.loc[tr, "admit_int"], df.loc[te, "admit_int"]

c1, c2, c3 = st.columns(3)
c1.metric("Train admit rate", f"{y_train.mean():.1%}")
c2.metric("Test admit rate", f"{y_test.mean():.1%}")
c3.metric("Majority-class accuracy", f"{1 - y_test.mean():.1%}")

clf_name = st.selectbox("Estimator", ["LogisticRegression", "RandomForest"])
cw_label = st.selectbox("class_weight", ["none", "balanced"])
cw = None if cw_label == "none" else "balanced"
if clf_name == "LogisticRegression":
    est = LogisticRegression(max_iter=500, class_weight=cw)
else:
    est = RandomForestClassifier(n_estimators=180, min_samples_leaf=4, class_weight=cw, random_state=0, n_jobs=-1)

pipe = Pipeline([("prep", admit_transformer()), ("model", est)])
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
thr = st.slider("Decision threshold (P(admit))", 0.05, 0.90, 0.35, 0.01)
pred = (proba >= thr).astype(int)

m1, m2, m3 = st.columns(3)
m1.metric("ROC AUC", f"{roc_auc_score(y_test, proba):.3f}")
m2.metric(f"F1 @ {thr:.2f}", f"{f1_score(y_test, pred):.3f}")
m3.metric("Predicted admit share", f"{pred.mean():.1%}")

st.markdown("### Confusion matrix at this threshold")
fig, ax = plt.subplots(figsize=(4.2, 3.6))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["discharge", "admit"], ax=ax, colorbar=False, cmap="Greens")
ax.set_title("Test set")
st.pyplot(fig, width="content")
plt.close(fig)

raw = classification_report(y_test, pred, target_names=["discharge", "admit"], output_dict=True, zero_division=0)
rows = []
for label in ["discharge", "admit", "macro avg", "weighted avg"]:
    r = raw[label]
    rows.append({"class": label, "precision": round(r["precision"], 3), "recall": round(r["recall"], 3), "f1": round(r["f1-score"], 3), "n": int(r["support"])})
rows.append({"class": "accuracy", "precision": None, "recall": None, "f1": round(raw["accuracy"], 3), "n": int(raw["macro avg"]["support"])})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.markdown("### ROC and PR — threshold is a bed decision")
fpr, tpr, _ = roc_curve(y_test, proba)
prec, rec, _ = precision_recall_curve(y_test, proba)
left, right = st.columns(2)
with left:
    fig_r = px.line(pd.DataFrame({"FPR": fpr, "TPR": tpr}), x="FPR", y="TPR", title="ROC")
    fig_r.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#666"))
    fig_r.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_r, width="stretch")
with right:
    fig_p = px.line(pd.DataFrame({"recall": rec, "precision": prec}), x="recall", y="precision", title="Precision–recall")
    fig_p.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_p, width="stretch")

pitfall("Accuracy at 0.5 with no class_weight will look like the majority baseline. Remember recall on admit at the threshold ops will actually use.")
ok("Tune the threshold on a time-based slice. `cost = c_fp * FP + c_fn * FN` with c_fn = missed admit.")

ask(
    "q7_clf",
    "The floor can spare 6 extra obs-beds (false positives). You should:",
    [
        "Maximize accuracy",
        "Raise the threshold until predicted-admit volume fits the budget, then read recall",
        "Always use 0.5 because it's the default",
    ],
    "Raise the threshold until predicted-admit volume fits the budget, then read recall",
    "The threshold is a capacity constraint. Sweep it. Don't pretend 0.5 is principled. The agent should expose the same threshold.",
)
