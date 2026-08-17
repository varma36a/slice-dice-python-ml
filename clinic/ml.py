"""Shared feature tables so wait, admit, clustering, and the agent stay consistent."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from clinic.data import BIOMARKERS, Clinic


def daily_model_frame(clinic: Clinic) -> pd.DataFrame:
    """Site-day census with calendar + season + lag features (no future leak)."""
    d = clinic.daily.sort_values(["site", "date"]).copy()
    d["enc_lag7"] = d.groupby("site", observed=True)["encounters"].shift(7)
    d["enc_roll7"] = d.groupby("site", observed=True)["encounters"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).mean()
    )
    d["is_heat"] = (d["season"] == "heat").astype(int)
    return d.dropna(subset=["enc_lag7", "enc_roll7"]).reset_index(drop=True)


def encounter_model_frame(clinic: Clinic) -> pd.DataFrame:
    """Encounter-level table for admit / late-triage models."""
    e = clinic.encounters.copy()
    e["site"] = e["site"].astype(str)
    e["condition"] = e["condition"].astype(str)
    e["arrival"] = e["arrival"].astype(str)
    e["season"] = e["season"].astype(str)
    e["esi_n"] = e["esi"].astype(int)
    e["rush"] = e["hour"].isin([10, 11, 17, 18, 19]).astype(int)
    e["admit_int"] = e["admit"].astype(int)
    e["late_int"] = e["late_triage"].astype(int)
    e["lactate_f"] = e["lactate"]
    e["troponin_f"] = e["troponin"]
    return e.reset_index(drop=True)


def patient_feature_frame(clinic: Clinic) -> pd.DataFrame:
    p = clinic.patients.copy()
    p = p[p["visits"] > 0].copy()
    p["log_visits"] = np.log1p(p["visits"])
    p["comorbid"] = p[["htn", "dm", "copd", "cad"]].sum(axis=1)
    return p.reset_index(drop=True)


def time_split(df: pd.DataFrame, date_col: str = "date", frac: float = 0.75) -> tuple[pd.Index, pd.Index]:
    cutoff = df[date_col].quantile(frac)
    train = df.index[df[date_col] <= cutoff]
    test = df.index[df[date_col] > cutoff]
    return train, test


def daily_transformer() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                StandardScaler(),
                ["dow", "is_weekend", "flu_wave", "is_heat", "enc_lag7", "enc_roll7"],
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["site", "season"]),
        ]
    )


def admit_transformer() -> ColumnTransformer:
    num = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            (
                "num",
                num,
                ["age", "esi_n", "hour", "spo2", "hr", "temp_c", "sbp", "wbc", "lactate_f", "troponin_f", "rush"],
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["site", "arrival", "season"]),
        ]
    )


def wait_transformer() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), ["hour", "esi_n", "is_weekend", "rush"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["site", "arrival", "season"]),
        ]
    )


def lab_vector(row: pd.Series, names: list[str] | None = None) -> np.ndarray:
    names = names or list(BIOMARKERS)
    vals = []
    for n in names:
        v = row.get(n)
        vals.append(np.nan if v is None else float(v) if pd.notna(v) else np.nan)
    return np.array(vals, dtype=np.float64)
