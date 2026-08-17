"""High-yield interview bank + live clinic demos (analog of GET /api/interview/*)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from clinic.data import Clinic
from clinic.topics import TOPICS

# Extra questions interviewers actually ask — not every micro-topic, the 45-minute set.
INTERVIEW_BANK: list[dict[str, object]] = [
    {
        "area": "Python",
        "topic": "GIL vs NumPy",
        "interview_q": "Python has a GIL. Why is NumPy still fast?",
        "interview_a": "The GIL serializes Python bytecode. NumPy loops in C/Fortran and releases the GIL for large ufuncs, so cores can run vectorized work. Your Python `for` over 20k encounters does not. That is why flag_labs is a mask, not a nested loop.",
        "example": "abnormal = (X < low) | (X > high)   # C loop, GIL released\n# vs for i in range(n): for j in range(14): ...",
        "choices": [
            "Ufuncs run in C and can release the GIL; Python for-loops do not",
            "The GIL was removed in Python 3.12 for all code",
            "NumPy starts a process per biomarker",
        ],
        "answer": "Ufuncs run in C and can release the GIL; Python for-loops do not",
        "where": "NumPy labs · Lab D",
    },
    {
        "area": "Python",
        "topic": "is vs ==",
        "interview_q": "When is `x is None` required instead of `== None`?",
        "interview_a": "`is` tests identity. `==` can be overloaded (NumPy arrays return an array, pandas can surprise you). For None/singletons use `is`. For value equality of labs use `==` or `np.isclose`.",
        "example": "if pending_troponin is None:\n    ...\n# never: if vec == None  # array comparison",
        "choices": [
            "is is identity; == can be overloaded (arrays) — None checks use is",
            "is is faster so always use it for numbers",
            "== None is a SyntaxError",
        ],
        "answer": "is is identity; == can be overloaded (arrays) — None checks use is",
        "where": "Python → Types",
    },
    {
        "area": "Pandas",
        "topic": "SettingWithCopy",
        "interview_q": "What is SettingWithCopyWarning really warning you about?",
        "interview_a": "You may be writing into a copy, so the original chart does not change. `enc[enc.esi==1]['wait_min'] = 0` is chained indexing. Use `enc.loc[mask, 'wait_min'] = 0` on a `.copy()` you intend to own.",
        "example": "# BAD\nenc[enc.esi == 1][\"wait_min\"] = 0\n# GOOD\nenc.loc[enc.esi == 1, \"wait_min\"] = 0",
        "choices": [
            "Chained indexing may write to a copy; use loc on a frame you own",
            "Pandas forbids assignment",
            "It means you leaked the test set",
        ],
        "answer": "Chained indexing may write to a copy; use loc on a frame you own",
        "where": "Pandas chart",
    },
    {
        "area": "Features",
        "topic": "Train/serving skew",
        "interview_q": "The notebook AUC is 0.91. Production scores garbage. Name three causes.",
        "interview_a": "1) Different columns (forgot rush, renamed lactate). 2) Encoder/imputer fit on full notebook frame, production uses train-only Pipeline — or the reverse. 3) Time leakage in the notebook split; live data is future. Also category drift (new arrival code) and missingness pattern change.",
        "example": "pipe.feature_names_in_  # must match ticket.columns\nOneHotEncoder(handle_unknown=\"ignore\")  # seatbelt, not a feature",
        "choices": [
            "Column mismatch, preprocess not in a Pipeline, notebook leakage vs live future data",
            "Streamlit cannot call sklearn",
            "AUC cannot be computed in production",
        ],
        "answer": "Column mismatch, preprocess not in a Pipeline, notebook leakage vs live future data",
        "where": "Capstone · Admit desk",
    },
    {
        "area": "Classification",
        "topic": "Calibration",
        "interview_q": "ROC-AUC is high but P(admit)=0.8 does not mean 80% admit. Why?",
        "interview_a": "AUC is ranking, not calibration. Trees especially are miscalibrated. If the agent shows '80%' to a clinician, you need reliability curves / Platt or isotonic on a held-out time window. Until then, call it a score or a band, not a probability.",
        "example": "return {\"p_admit\": round(proba, 3), \"band\": \"high\" if proba >= 0.45 else \"low\"}\n# band is honest when calibration is unproven",
        "choices": [
            "AUC ranks; it does not make 0.8 a frequency — calibrate or show a band",
            "LogReg is always calibrated",
            "0.8 always means 80% in sklearn",
        ],
        "answer": "AUC ranks; it does not make 0.8 a frequency — calibrate or show a band",
        "where": "Classification · Agent",
    },
    {
        "area": "Regression",
        "topic": "Bias–variance",
        "interview_q": "Ridge vs RandomForest for site-day census — bias-variance in one minute.",
        "interview_a": "Ridge: high bias, low variance, stable under collinear lags. RF: lower bias, higher variance, can memorize a flu week if unconstrained. TimeSeriesSplit + min_samples_leaf is how you keep RF from fitting last Friday's noise. Prefer the model that beats lag-7 on a future window, not the one with train R²=0.99.",
        "example": "RandomForestRegressor(min_samples_leaf=4)  # raise leaf to cut variance\nRidge(alpha=1.0)  # raise alpha to cut variance",
        "choices": [
            "Ridge: more bias, less variance; RF can memorize a flu week — judge on a future window vs lag-7",
            "Variance means NaNs in y",
            "Bias is always worse than variance",
        ],
        "answer": "Ridge: more bias, less variance; RF can memorize a flu week — judge on a future window vs lag-7",
        "where": "Regression",
    },
    {
        "area": "Features",
        "topic": "L1 vs L2",
        "interview_q": "Lasso vs Ridge for this feature set?",
        "interview_a": "L2 (Ridge) shrinks but keeps correlated lags/dummies. L1 (Lasso) zeros some coefficients — useful if you want a short list of assays/features. Elastic net is the mix. Do not use L1 as automated science on 8 dummy sites.",
        "example": "Ridge(alpha=1.0)    # L2, keep collinear lags\nLasso(alpha=0.1)    # L1, sparse — careful with dummies",
        "choices": [
            "L2 shrinks and keeps correlated features; L1 zeros some — Ridge is the default here",
            "L1 and L2 are the same on one-hots",
            "Lasso is a clustering method",
        ],
        "answer": "L2 shrinks and keeps correlated features; L1 zeros some — Ridge is the default here",
        "where": "Regression · Clusters",
    },
    {
        "area": "Agent",
        "topic": "Eval vs RAG leakage",
        "interview_q": "How do you eval an agent without teaching it the answer?",
        "interview_a": "Tools return observations without gold labels. After stop, compare guess vs gold_label_condition and p_admit vs gold_admit. If retrieval includes the answer sheet (RAG over labeled charts), you have leakage. Same rule as a held-out test set.",
        "example": "eval = {\"match\": protocol[\"condition\"] == gold[\"gold_label_condition\"]}\n# gold never passed into plan() or get_chart()",
        "choices": [
            "Hide gold from tools/planner; score after stop — retrieval must not include the answer sheet",
            "Put gold in the system prompt for accuracy",
            "Eval is only ROC-AUC of the planner",
        ],
        "answer": "Hide gold from tools/planner; score after stop — retrieval must not include the answer sheet",
        "where": "Agent workflow",
    },
    {
        "area": "Classification",
        "topic": "Dummy baseline",
        "interview_q": "What dummy classifiers belong in every admit interview answer?",
        "interview_a": "most_frequent (always discharge) for accuracy. stratified random for a chance PR-AUC. You must beat both at the operating threshold, not only ROC-AUC vs a coin flip that looks fine on TNs.",
        "example": "DummyClassifier(strategy=\"most_frequent\")\nDummyClassifier(strategy=\"stratified\")",
        "choices": [
            "most_frequent (always discharge) and stratified — beat them at the real threshold",
            "DummyClassifier is deprecated",
            "Only neural nets need baselines",
        ],
        "answer": "most_frequent (always discharge) and stratified — beat them at the real threshold",
        "where": "Classification",
    },
]


def all_questions() -> list[dict[str, object]]:
    """Per-topic drills plus the high-yield bank."""
    rows = []
    for t in TOPICS:
        rows.append(
            {
                "area": t["area"],
                "topic": t["topic"],
                "interview_q": t["interview_q"],
                "interview_a": t["interview_a"],
                "example": t["example"],
                "choices": t["choices"],
                "answer": t["answer"],
                "where": t["where"],
                "source": "topic",
            }
        )
    for q in INTERVIEW_BANK:
        rows.append({**q, "source": "bank"})
    return rows


def live_interview(clinic: Clinic) -> dict[str, pd.DataFrame]:
    """Runnable examples — same idea as GET /api/interview/di-lifetimes and /linq."""
    enc = clinic.encounters
    name_to_i = {n: i for i, n in enumerate(clinic.condition_names)}
    idx = enc["condition"].astype(str).map(name_to_i).to_numpy()
    counts = np.bincount(idx.astype(int), minlength=len(clinic.condition_names))
    pull = pd.DataFrame(
        {"assay": clinic.biomarker_names, "pull": np.round(counts @ clinic.protocol, 1)}
    ).sort_values("pull", ascending=False)
    atlas = clinic.atlas.set_index("condition").loc[list(clinic.condition_names)]
    panel = pd.DataFrame(
        {
            "condition": clinic.condition_names,
            "panel_cost": np.round(clinic.protocol @ clinic.assay_cost, 2),
            "admit_base": np.round(atlas["admit_base"].to_numpy(), 3),
        }
    )
    proto = "Pneumonia" if "Pneumonia" in clinic.condition_names else clinic.condition_names[0]
    vec = clinic.lab_means[clinic.condition_index(proto)]
    flag = (vec < clinic.ref_low) | (vec > clinic.ref_high)
    flags = pd.DataFrame(
        {
            "biomarker": clinic.biomarker_names,
            f"typical_{proto}": np.round(vec, 2),
            "ref_low": clinic.ref_low,
            "ref_high": clinic.ref_high,
            "abnormal": flag,
        }
    )
    grp = (
        enc.groupby(["site", "condition"], observed=True)
        .agg(n=("encounter_id", "count"), admit_rate=("admit", "mean"))
        .reset_index()
        .sort_values("n", ascending=False)
        .head(8)
    )
    grp["admit_rate"] = grp["admit_rate"].round(3)
    merged = enc.merge(clinic.patients[["patient_id"]], on="patient_id", how="left")
    join_stats = pd.DataFrame(
        [
            {"join": "encounters", "rows": len(enc)},
            {"join": "after left merge patients", "rows": len(merged)},
            {"join": "inner would keep", "rows": int(enc["patient_id"].isin(clinic.patients["patient_id"]).sum())},
        ]
    )
    return {
        "panel_cost (protocol @ cost)": panel,
        "broadcast flags (Pneumonia vs ref)": flags,
        "groupby site × condition": grp,
        "merge row counts": join_stats,
        "reagent pull (counts @ protocol)": pull.head(8),
    }
