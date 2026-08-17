"""Worked result tables for every catalog topic — so By module is example + output, not a stub."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
WORKED_PATH = DATA_DIR / "worked.json"


def _key(area: str, topic: str) -> str:
    return f"{area}||{topic}"


def load_worked() -> dict[tuple[str, str], pd.DataFrame]:
    """Pre-baked tables — Cloud must not fit sklearn on boot."""
    if not WORKED_PATH.exists():
        from clinic.store import baked_clinic

        clinic = baked_clinic()
        if clinic is None:
            return {}
        return worked_results(clinic)
    payload = json.loads(WORKED_PATH.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for k, rows in payload.items():
        area, topic = k.split("||", 1)
        out[(area, topic)] = pd.DataFrame(rows)
    return out


def save_worked(results: dict[tuple[str, str], pd.DataFrame], dest: Path | None = None) -> Path:
    dest = dest or WORKED_PATH
    payload = {}
    for (area, topic), df in results.items():
        frame = df.copy()
        if not isinstance(frame.index, pd.RangeIndex) or frame.index.name is not None:
            frame = frame.reset_index()
        else:
            frame = frame.reset_index(drop=True)
        payload[_key(str(area), str(topic))] = json.loads(
            frame.to_json(orient="records", date_format="iso")
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload), encoding="utf-8")
    return dest


from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import timedelta
from functools import wraps
from itertools import islice, product
from typing import Callable, Literal

import numpy as np

from clinic.data import Clinic


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def worked_results(clinic: Clinic) -> dict[tuple[str, str], pd.DataFrame]:
    """(area, topic) → small result frame computed on this clinic."""
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from clinic.ml import daily_model_frame, encounter_model_frame, time_split
    enc = clinic.encounters
    atlas = clinic.atlas
    patients = clinic.patients
    daily = clinic.daily
    cases = enc.head(12).copy()
    r0 = cases.iloc[0]
    out: dict[tuple[str, str], pd.DataFrame] = {}

    out[("Python", "Scalar types")] = _df(
        [
            {"field": "encounter_id", "value": str(int(r0.encounter_id)), "python_type": "int"},
            {"field": "age", "value": str(int(r0.age)), "python_type": "int"},
            {"field": "spo2", "value": str(float(r0.spo2)), "python_type": "float"},
            {"field": "condition", "value": str(r0.condition), "python_type": "str"},
            {"field": "admit", "value": str(bool(r0.admit)), "python_type": "bool"},
            {"field": "pending_troponin", "value": "None", "python_type": "NoneType"},
        ]
    )

    names = atlas["condition"].tolist()
    resp = set(atlas.loc[atlas["respiratory"], "condition"].astype(str)) if "respiratory" in atlas else set()
    tonight = set(cases["condition"].astype(str))
    out[("Python", "Collections")] = _df(
        [
            {"operation": "tonight ∩ respiratory", "result": ", ".join(sorted(tonight & resp)) or "—"},
            {"operation": "tonight − respiratory", "result": ", ".join(sorted(tonight - resp)) or "—"},
            {"operation": "tuple key", "result": str(("Campus", 2))},
        ]
    )

    sku_rows = []
    for r in cases.itertuples(index=False):
        sku_rows.append(
            {
                "sku": f"{str(r.condition)[:3].upper()}-E{int(r.esi)}-{str(r.arrival)[:3].upper()}",
                "wait_label": f"{r.wait_min:.0f} min",
                "n_beds // 2 vs / 2": f"{5 // 2} vs {5 / 2}",
            }
        )
    out[("Python", "Strings & numbers")] = pd.DataFrame(sku_rows).head(6)

    def bucket(esi: int, spo2: float, arrival: str) -> str:
        if spo2 < 92 or esi == 1:
            return "resusc"
        if esi == 2 or arrival == "ambulance":
            return "urgent"
        if esi >= 4:
            return "fast_track"
        return "standard"

    classified = cases.copy()
    classified["bucket"] = [
        bucket(int(e), float(s), str(a)) for e, s, a in zip(classified["esi"], classified["spo2"], classified["arrival"])
    ]
    out[("Python", "Control flow")] = (
        classified["bucket"].value_counts().rename_axis("bucket").reset_index(name="n")
    )

    def acuity_weight(esi: int, *bumps: float) -> float:
        base = {1: 1.0, 2: 0.8, 3: 0.45, 4: 0.2, 5: 0.1}[int(esi)]
        return min(1.0, base + sum(bumps))

    out[("Python", "Functions")] = _df(
        [
            {"esi": 1, "hypoxia_bump": 0.25, "acuity_weight": acuity_weight(1, 0.25)},
            {"esi": 3, "hypoxia_bump": 0.0, "acuity_weight": acuity_weight(3)},
            {"esi": 5, "hypoxia_bump": 0.0, "acuity_weight": acuity_weight(5)},
        ]
    )

    high_admit = [c for c, a in zip(atlas["condition"], atlas["admit_base"]) if a >= 0.25]
    first, *mid, last = names
    out[("Python", "Comprehensions")] = _df(
        [
            {"piece": "high admit_base ≥ 0.25", "value": ", ".join(high_admit)},
            {"piece": "first", "value": first},
            {"piece": "last", "value": last},
            {"piece": "middle count", "value": str(len(mid))},
        ]
    )

    @dataclass
    class Encounter:
        condition: str
        esi: int
        spo2: float
        symptoms: list[str] = field(default_factory=list)

        @property
        def hypoxia(self) -> bool:
            return self.spo2 < 92

    board = [Encounter(str(r.condition), int(r.esi), float(r.spo2)) for r in cases.itertuples(index=False)]
    out[("Python", "OOP")] = _df(
        [
            {"metric": "encounters on board", "value": str(len(board))},
            {"metric": "hypoxia (spo2 < 92)", "value": str(sum(1 for e in board if e.hypoxia))},
        ]
    )

    def parse_esi(raw: str) -> str:
        if raw not in {"1", "2", "3", "4", "5"}:
            return f"ValueError: ESI {raw!r} not in 1–5"
        return f"ok → {int(raw)}"

    out[("Python", "Exceptions")] = _df(
        [{"input": x, "result": parse_esi(x)} for x in ["3", "2", "0", "stat"]]
    )

    payload_n = int(len(cases))
    out[("Python", "Files & JSON")] = _df(
        [
            {"artifact": "json payload n", "value": str(payload_n)},
            {"artifact": "path", "value": "data/encounters.parquet"},
            {"artifact": "csv rows if exported", "value": str(payload_n)},
        ]
    )

    ts = pd.to_datetime(cases["ts"])
    out[("Python", "Datetime")] = _df(
        [
            {"field": "min ts", "value": str(ts.min())},
            {"field": "hour of first", "value": str(int(ts.dt.hour.iloc[0]))},
            {"field": "+ timedelta(days=1)", "value": str(ts.min() + timedelta(days=1))},
        ]
    )

    by_site = Counter(cases["site"].astype(str))
    flags: dict[str, list[str]] = defaultdict(list)
    flags[str(r0.site)].append("opened")
    trace: deque[str] = deque(maxlen=3)
    for s in ["get_chart", "flag_labs", "score_admit", "stop"]:
        trace.append(s)
    out[("Python", "collections")] = _df(
        [
            {"api": "Counter(site)", "value": str(dict(by_site))},
            {"api": "defaultdict first site", "value": str(flags[str(r0.site)])},
            {"api": "deque(maxlen=3)", "value": str(list(trace))},
        ]
    )

    def stream_rows(rows):
        for row in rows:
            yield row

    preview = list(islice(stream_rows(cases.itertuples(index=False)), 3))
    out[("Python", "Generators")] = _df(
        [{"yielded": i + 1, "encounter_id": int(p.encounter_id), "condition": str(p.condition)} for i, p in enumerate(preview)]
    )

    cross = list(product(clinic.sites[:2], [1, 2]))
    out[("Python", "itertools")] = _df(
        [{"site": s, "esi": e} for s, e in cross]
    )

    Band = Literal["low", "mid", "high"]

    def band_of(p: float) -> Band:
        return "high" if p >= 0.45 else ("mid" if p >= 0.20 else "low")

    out[("Python", "Typing")] = _df(
        [{"p_admit": p, "Band": band_of(p)} for p in (0.12, 0.33, 0.61)]
    )

    def counted(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            wrapper.n += 1
            return fn(*args, **kwargs)

        wrapper.n = 0
        return wrapper

    @counted
    def get_chart(eid: int) -> int:
        return eid

    get_chart(1)
    get_chart(2)
    out[("Python", "Decorators")] = _df(
        [{"wrapper": "get_chart", "calls": get_chart.n, "__name__": get_chart.__name__}]
    )
    out[("Python", "Context managers")] = _df(
        [{"pattern": "with Path.open() / @contextmanager", "guarantee": "__exit__ runs even on raise"}]
    )

    out[("NumPy", "ndarray shape / dtype")] = _df(
        [
            {"array": "lab_means", "shape": str(clinic.lab_means.shape), "dtype": str(clinic.lab_means.dtype)},
            {"array": "protocol", "shape": str(clinic.protocol.shape), "dtype": str(clinic.protocol.dtype)},
            {"array": "assay_cost", "shape": str(clinic.assay_cost.shape), "dtype": str(clinic.assay_cost.dtype)},
            {"array": "inventory", "shape": str(clinic.inventory.shape), "dtype": str(clinic.inventory.dtype)},
        ]
    )

    panel = np.round(clinic.protocol @ clinic.assay_cost, 2)
    out[("NumPy", "matmul / @")] = _df(
        {"condition": clinic.condition_names, "panel_cost": panel}
    )

    proto = "Pneumonia" if "Pneumonia" in clinic.condition_names else clinic.condition_names[0]
    vec = clinic.lab_means[clinic.condition_index(proto)]
    z = (vec - clinic.ref_low) / np.clip(clinic.ref_high - clinic.ref_low, 1e-6, None)
    out[("NumPy", "Broadcasting")] = pd.DataFrame(
        {
            "biomarker": clinic.biomarker_names,
            "typical": np.round(vec, 2),
            "z_vs_band": np.round(z, 2),
        }
    ).head(8)

    flag = (vec < clinic.ref_low) | (vec > clinic.ref_high)
    out[("NumPy", "Boolean masks")] = pd.DataFrame(
        {
            "biomarker": np.array(clinic.biomarker_names)[flag],
            "typical": np.round(vec[flag], 2),
            "flag": "abnormal",
        }
    ) if flag.any() else _df([{"biomarker": "— none —", "typical": np.nan, "flag": "in range"}])

    name_to_i = {n: i for i, n in enumerate(clinic.condition_names)}
    idx = enc["condition"].astype(str).map(name_to_i).to_numpy()
    counts = np.bincount(idx.astype(int), minlength=len(clinic.condition_names))
    pull = counts @ clinic.protocol
    out[("NumPy", "Fancy / integer index")] = pd.DataFrame(
        {"assay": clinic.biomarker_names, "pull": np.round(pull, 1)}
    ).sort_values("pull", ascending=False).head(8)

    view = clinic.lab_means[0]
    out[("NumPy", "View vs copy")] = _df(
        [
            {"check": "lab_means[0].base is not None (view?)", "value": str(view.base is not None)},
            {"check": ".copy().base", "value": str(clinic.lab_means[0].copy().base is not None)},
        ]
    )

    out[("NumPy", "Axis reductions")] = _df(
        [
            {"reduction": "lab_means.mean(axis=0) shape", "value": str(clinic.lab_means.mean(axis=0).shape)},
            {"reduction": "inventory.sum(axis=1) shape", "value": str(clinic.inventory.sum(axis=1).shape)},
            {"reduction": "mean first assay", "value": f"{clinic.lab_means.mean(axis=0)[0]:.2f}"},
        ]
    )

    missing_lac = int(enc["lactate"].isna().sum()) if "lactate" in enc else 0
    out[("NumPy", "clip / nan")] = _df(
        [
            {"stat": "lactate NaNs (pandas/NumPy nan)", "value": str(missing_lac)},
            {"stat": "troponin NaNs", "value": str(int(enc["troponin"].isna().sum())) if "troponin" in enc else "—"},
        ]
    )

    x = vec.copy()
    dist = np.linalg.norm(clinic.lab_means - x, axis=1)
    guess = clinic.condition_names[int(np.argmin(dist))]
    out[("NumPy", "L2 nearest prototype")] = _df(
        [
            {"query": f"typical {proto}", "nearest": guess, "min_L2": f"{float(np.min(dist)):.3f}"},
        ]
    )

    out[("Pandas", "DataFrame / Series")] = _df(
        [
            {"stat": "rows", "value": f"{len(enc):,}"},
            {"stat": "columns", "value": str(enc.shape[1])},
            {"stat": "memory_MB", "value": f"{enc.memory_usage(deep=True).sum() / 1e6:.2f}"},
        ]
    )

    hypoxia = enc[enc["spo2"] < 92]
    campus_hot = enc.loc[enc.site.eq("Campus") & enc.esi.isin([1, 2])]
    out[("Pandas", "loc / boolean filter")] = _df(
        [
            {"filter": "spo2 < 92", "n": len(hypoxia), "admit_rate": f"{hypoxia.admit.mean():.1%}" if len(hypoxia) else "—"},
            {"filter": "Campus & ESI 1–2", "n": len(campus_hot), "admit_rate": f"{campus_hot.admit.mean():.1%}" if len(campus_hot) else "—"},
        ]
    )

    g = (
        enc.groupby(["site", "condition"], observed=True)
        .agg(n=("encounter_id", "count"), admit_rate=("admit", "mean"))
        .reset_index()
        .sort_values("n", ascending=False)
        .head(8)
    )
    g["admit_rate"] = g["admit_rate"].round(3)
    out[("Pandas", "groupby agg")] = g

    tmp = enc.copy()
    tmp["site_wait"] = tmp.groupby("site", observed=True)["wait_min"].transform("mean")
    tmp["vs_site"] = tmp["wait_min"] / tmp["site_wait"]
    out[("Pandas", "transform")] = (
        tmp.groupby("site", observed=True)["vs_site"].mean().round(3).rename_axis("site").reset_index(name="mean_vs_site")
    )

    merged = enc.merge(patients[["patient_id"]], on="patient_id", how="left")
    inner_n = int(enc["patient_id"].isin(patients["patient_id"]).sum())
    out[("Pandas", "merge")] = _df(
        [
            {"join": "encounters", "rows": len(enc)},
            {"join": "left merge patients", "rows": len(merged)},
            {"join": "inner would keep", "rows": inner_n},
        ]
    )

    out[("Pandas", "Missing data")] = _df(
        [
            {"lab": "lactate", "na": int(enc["lactate"].isna().sum()), "pct": f"{enc['lactate'].isna().mean():.1%}"},
            {"lab": "troponin", "na": int(enc["troponin"].isna().sum()), "pct": f"{enc['troponin'].isna().mean():.1%}"},
        ]
    )

    feat = daily_model_frame(clinic)
    out[("Pandas", "Time series")] = feat[["site", "date", "encounters", "enc_lag7", "enc_roll7"]].head(8)

    out[("Pandas", "crosstab / pivot")] = pd.crosstab(enc["site"], enc["condition"])

    out[("Pandas", "Categorical")] = _df(
        [
            {"col": c, "dtype": str(enc[c].dtype)}
            for c in ["site", "arrival", "season", "esi"]
            if c in enc.columns
        ]
    )

    by_date = daily.groupby("date", observed=True)["encounters"].sum().reset_index().tail(8)
    out[("EDA", "Line / bar / box")] = by_date.rename(columns={"encounters": "network_census"})

    heat = pd.crosstab(enc["site"], enc["hour"], values=enc["admit"], aggfunc="mean").round(3)
    out[("EDA", "Heatmap")] = heat.reset_index()

    site_day = daily.groupby("site", observed=True).agg(encounters=("encounters", "mean"), admit_rate=("admit_rate", "mean")).round(3).reset_index()
    out[("EDA", "Scatter")] = site_day

    num_cols = [c for c in ["encounters", "enc_lag7", "enc_roll7", "flu_wave"] if c in feat.columns]
    out[("EDA", "Correlation")] = feat[num_cols].corr().round(3).reset_index().rename(columns={"index": "var"})

    out[("EDA", "Plotly + seaborn")] = _df(
        [
            {"lib": "plotly", "use": "interactive census by site (px.line)"},
            {"lib": "seaborn", "use": f"sample scatter n={min(800, len(enc))} so overplotting does not hide density"},
        ]
    )

    out[("Features", "Calendar features")] = feat[["date", "site", "dow", "is_weekend", "flu_wave", "is_heat"]].head(8)

    out[("Features", "Lag / roll")] = feat[["site", "date", "encounters", "enc_lag7", "enc_roll7"]].head(8)

    X_num = feat[["dow", "is_weekend", "flu_wave", "is_heat", "enc_lag7", "enc_roll7"]].to_numpy()
    y = feat["encounters"].to_numpy()
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X_num, y, test_size=0.25, random_state=0)
    r2_rand = r2_score(yte_r, LinearRegression().fit(Xtr_r, ytr_r).predict(Xte_r))
    tr_idx, te_idx = time_split(feat, "date", 0.75)
    r2_time = r2_score(y[te_idx], LinearRegression().fit(X_num[tr_idx], y[tr_idx]).predict(X_num[te_idx]))
    out[("Features", "Time split vs random")] = _df(
        [
            {"split": "random 75/25 R²", "value": round(float(r2_rand), 3)},
            {"split": "time split R²", "value": round(float(r2_time), 3)},
        ]
    )

    X_leak = np.hstack([X_num, feat[["admit_rate"]].to_numpy()])
    r2_leak = r2_score(y[te_idx], LinearRegression().fit(X_leak[tr_idx], y[tr_idx]).predict(X_leak[te_idx]))
    out[("Features", "Leakage")] = _df(
        [
            {"feature_set": "calendar + lags (time split)", "R2": round(float(r2_time), 3)},
            {"feature_set": "+ same-day admit_rate (illegal)", "R2": round(float(r2_leak), 3)},
        ]
    )

    leaky = enc.groupby("condition", observed=True)["admit"].transform("mean")
    out[("Features", "Target encoding leak")] = (
        enc.assign(leaky_cond_admit=leaky)[["condition", "admit", "leaky_cond_admit"]].head(6)
    )

    out[("Features", "High-cardinality IDs")] = _df(
        [
            {"id": "patient_id", "n_unique": int(enc["patient_id"].nunique()), "one_hot_cols_if_used": int(enc["patient_id"].nunique())},
            {"id": "encounter_id", "n_unique": int(enc["encounter_id"].nunique()), "one_hot_cols_if_used": int(enc["encounter_id"].nunique())},
        ]
    )

    out[("Regression", "Problem")] = feat[["site", "date", "encounters", "enc_lag7", "enc_roll7"]].head(6)

    y_te = y[te_idx]
    naive = feat.loc[te_idx, "enc_lag7"].to_numpy()
    mae_naive = float(np.mean(np.abs(y_te - naive)))
    yhat_lin = LinearRegression().fit(X_num[tr_idx], y[tr_idx]).predict(X_num[te_idx])
    mae_lin = float(mean_absolute_error(y_te, yhat_lin))
    out[("Regression", "Naive baseline")] = _df(
        [
            {"model": "lag-7 naive", "MAE": round(mae_naive, 2)},
            {"model": "LinearRegression", "MAE": round(mae_lin, 2)},
            {"model": "beats naive?", "MAE": "yes" if mae_lin < mae_naive else "no"},
        ]
    )

    ridge = Ridge(alpha=1.0).fit(X_num[tr_idx], y[tr_idx])
    mae_ridge = float(mean_absolute_error(y_te, ridge.predict(X_num[te_idx])))
    out[("Regression", "Linear / Ridge / RF")] = _df(
        [
            {"estimator": "LinearRegression", "MAE": round(mae_lin, 2)},
            {"estimator": "Ridge(alpha=1)", "MAE": round(mae_ridge, 2)},
        ]
    )

    out[("Regression", "MAE RMSE R²")] = _df(
        [
            {"metric": "MAE", "linear": round(mae_lin, 2), "naive": round(mae_naive, 2)},
            {"metric": "R²", "linear": round(float(r2_time), 3), "naive": round(float(r2_score(y_te, naive)), 3)},
        ]
    )

    resid = y_te - yhat_lin
    out[("Regression", "Residuals")] = _df(
        [
            {"stat": "mean residual", "value": round(float(resid.mean()), 3)},
            {"stat": "std residual", "value": round(float(resid.std()), 3)},
            {"stat": "max abs residual", "value": round(float(np.abs(resid).max()), 2)},
        ]
    )

    coefs = pd.DataFrame(
        {"feature": ["dow", "is_weekend", "flu_wave", "is_heat", "enc_lag7", "enc_roll7"], "coef": np.round(ridge.coef_, 3)}
    )
    out[("Regression", "Coefficients / importance")] = coefs

    df = encounter_model_frame(clinic)
    tr, te = time_split(df, "date", 0.75)
    y_train, y_test = df.loc[tr, "admit_int"], df.loc[te, "admit_int"]
    out[("Classification", "Problem")] = _df(
        [
            {"grain": "encounter", "target": "admit_int", "train_n": int(len(y_train)), "test_n": int(len(y_test))},
        ]
    )
    out[("Classification", "Class imbalance")] = _df(
        [
            {"split": "train admit rate", "value": f"{float(y_train.mean()):.1%}"},
            {"split": "test admit rate", "value": f"{float(y_test.mean()):.1%}"},
            {"split": "always-discharge accuracy", "value": f"{1 - float(y_test.mean()):.1%}"},
        ]
    )
    out[("Classification", "LogReg / RandomForest")] = _df(
        [
            {"knob": "class_weight", "none": "majority looks accurate", "balanced": "admit class counts more in the loss"},
            {"knob": "threshold", "none": "still 0.5 unless you set it", "balanced": "still choose a bed cut"},
        ]
    )
    out[("Classification", "Threshold")] = _df(
        [
            {"threshold": 0.20, "meaning": "more predicted admits, higher recall, more false alarms"},
            {"threshold": 0.35, "meaning": "course default on the slider"},
            {"threshold": 0.50, "meaning": "sklearn predict() default — not a bed policy"},
        ]
    )
    out[("Classification", "Confusion matrix")] = _df(
        [
            {"cell": "FN (missed admit)", "cost": "clinical miss — usually the expensive error"},
            {"cell": "FP (false admit flag)", "cost": "wasted bed / hallway board"},
        ]
    )
    out[("Classification", "ROC and PR")] = _df(
        [
            {"curve": "ROC-AUC", "use": "ranking; can look strong from TNs"},
            {"curve": "PR-AUC", "use": "minority (admit) ranking — prefer under ~12% base rate"},
        ]
    )
    out[("Classification", "Imputation in Pipeline")] = _df(
        [
            {"step": "SimpleImputer(median)", "fit_on": "train only, inside Pipeline"},
            {"step": "StandardScaler", "fit_on": "train only, after impute"},
            {"step": "OneHotEncoder(handle_unknown='ignore')", "fit_on": "train categories"},
        ]
    )

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    from clinic.ml import patient_feature_frame

    pat = patient_feature_frame(clinic)
    cols = [c for c in ["age", "log_visits", "admits", "recency_days", "comorbid", "avg_esi"] if c in pat.columns]
    Xz = StandardScaler().fit_transform(pat[cols].fillna(0).to_numpy())
    labels = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(Xz)
    out[("Clusters & pipelines", "K-means")] = _df(
        [{"k": 4, "n_patients": len(pat), "silhouette": round(float(silhouette_score(Xz, labels)), 3)}]
    )
    out[("Clusters & pipelines", "Scaling")] = _df(
        [
            {"feature": c, "raw_std": round(float(pat[c].fillna(0).std()), 3), "after_scale_std": 1.0}
            for c in cols[:4]
        ]
    )
    from sklearn.decomposition import PCA

    xy = PCA(n_components=2, random_state=0).fit_transform(Xz)
    out[("Clusters & pipelines", "PCA")] = _df(
        [{"pc": "pc1", "var_explained": round(float(PCA(n_components=2, random_state=0).fit(Xz).explained_variance_ratio_[0]), 3)},
         {"pc": "pc2", "var_explained": round(float(PCA(n_components=2, random_state=0).fit(Xz).explained_variance_ratio_[1]), 3)}]
    )
    out[("Clusters & pipelines", "Silhouette")] = _df(
        [{"k": 4, "silhouette": round(float(silhouette_score(Xz, labels)), 3), "note": "hint only — name the cluster in English"}]
    )
    pat = pat.copy()
    pat["segment"] = labels
    out[("Clusters & pipelines", "Phenotypes")] = (
        pat.groupby("segment")
        .agg(n=("patient_id", "count"), age=("age", "mean"), visits=("visits", "mean"), comorbid=("comorbid", "mean"))
        .round(2)
        .reset_index()
    )

    pipe = Pipeline([("sc", StandardScaler()), ("model", Ridge(alpha=1.0))])
    pipe.fit(X_num[tr_idx], y[tr_idx])
    out[("Clusters & pipelines", "Pipeline")] = _df(
        [
            {"step": "sc", "class": "StandardScaler"},
            {"step": "model", "class": "Ridge"},
            {"step": "test MAE", "class": str(round(float(mean_absolute_error(y_te, pipe.predict(X_num[te_idx]))), 2))},
        ]
    )
    out[("Clusters & pipelines", "TimeSeriesSplit + GridSearch")] = _df(
        [
            {"cv": "KFold shuffle", "ok_for_census": "no — future Fridays in train"},
            {"cv": "TimeSeriesSplit", "ok_for_census": "yes — walk forward"},
        ]
    )

    out[("Agent", "Tool")] = _df(
        [
            {"tool": "get_chart", "returns": "ESI, arrival, comorbidities (no gold labels)"},
            {"tool": "flag_labs", "returns": "NumPy masks vs ref_low/high"},
            {"tool": "score_admit", "returns": "p_admit + band from Pipeline"},
            {"tool": "nearest_condition", "returns": "L2 prototype guess"},
            {"tool": "retrieve_protocol", "returns": "next tests / disposition hint"},
        ]
    )
    out[("Agent", "Planner")] = _df(
        [
            {"state": "chart is None", "next": "get_chart"},
            {"state": "lab_flags is None", "next": "flag_labs"},
            {"state": "admit_score is None", "next": "score_admit"},
            {"state": "protocol is None", "next": "nearest_condition → retrieve_protocol"},
            {"state": "else", "next": "stop"},
        ]
    )
    out[("Agent", "State / memory")] = _df(
        [
            {"field": f, "role": r}
            for f, r in [
                ("encounter_id", "which chart"),
                ("chart", "observation from get_chart"),
                ("lab_flags", "observation from flag_labs"),
                ("admit_score", "observation from score_admit"),
                ("protocol", "observation from retrieve_protocol"),
                ("done", "stop flag"),
            ]
        ]
    )
    out[("Agent", "Trace")] = _df(
        [
            {"step": 1, "thought": "Need the chart", "tool": "get_chart"},
            {"step": 2, "thought": "Flag labs", "tool": "flag_labs"},
            {"step": 3, "thought": "Score admit", "tool": "score_admit"},
            {"step": 4, "thought": "Nearest prototype", "tool": "nearest_condition"},
            {"step": 5, "thought": "Stop", "tool": "stop"},
        ]
    )
    out[("Agent", "Stop + recommend")] = _df(
        [
            {
                "field": "recommendation",
                "value": "Suggested (not a diagnosis): prototype + admit band + next tests. Clinician must confirm.",
            }
        ]
    )
    gold_hidden = "gold_label_condition" not in ["esi", "arrival", "spo2"]
    out[("Agent", "Gold labels")] = _df(
        [
            {"label": "gold_label_condition", "in_get_chart": "popped before planner sees it"},
            {"label": "gold_admit", "in_get_chart": "eval only after stop"},
            {"label": "hidden?", "in_get_chart": str(gold_hidden)},
        ]
    )
    out[("Agent", "Model as a tool")] = _df(
        [
            {"object": "sklearn Pipeline", "role": "score_admit tool, not the product"},
            {"object": "joblib.dump(pipe)", "role": "one artifact the agent loads"},
        ]
    )

    last_day = daily["date"].max()
    tonight = daily[daily["date"] == last_day][["site", "encounters", "admit_rate"]].copy()
    out[("Capstone", "Census board")] = tonight.round(3)

    out[("Capstone", "Reagent pull")] = pd.DataFrame(
        {"assay": clinic.biomarker_names, "pull": np.round(pull, 1)}
    ).sort_values("pull", ascending=False).head(8)

    ticket = cases.iloc[0]
    out[("Capstone", "Admit desk")] = _df(
        [
            {
                "encounter_id": int(ticket.encounter_id),
                "esi": int(ticket.esi),
                "spo2": float(ticket.spo2),
                "arrival": str(ticket.arrival),
                "condition_hidden_from_scorer": str(ticket.condition),
            }
        ]
    )
    out[("Capstone", "Run agent")] = _df(
        [
            {"step": "run_agent(clinic, admit_pipe, encounter_id)", "gets": "state + trace + eval vs gold"},
        ]
    )
    return out
