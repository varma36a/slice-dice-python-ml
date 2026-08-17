"""Synthetic Northshore Urgent Care universe.

Educational data only — not medical advice, not a real diagnostic system.
One seeded generator so NumPy → sklearn → the diagnostic agent share the same clinic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

SEED = 42

SITES = ["Downtown", "Airport", "Campus", "Harbor", "Suburb"]
ARRIVALS = ["walk_in", "ambulance", "referral"]
SEASONS = ["typical", "flu_wave", "heat"]
ESI = [1, 2, 3, 4, 5]
PAYERS = ["self_pay", "medicaid", "commercial", "medicare"]

BIOMARKERS = [
    "wbc",
    "crp",
    "lactate",
    "spo2",
    "hr",
    "temp_c",
    "sbp",
    "glucose",
    "creatinine",
    "sodium",
    "potassium",
    "troponin",
    "ua_nitrite",
    "cxr_infiltrate",
]

# Lab reagent / assay unit cost — NumPy COGS analogue.
ASSAY_COST = np.array(
    [4.0, 6.5, 8.0, 0.4, 0.2, 0.3, 0.2, 1.5, 3.0, 1.2, 1.2, 18.0, 2.5, 22.0],
    dtype=np.float64,
)

# Adult reference: (low, high). Binary flags use (0, 0.5).
REF_LOW = np.array(
    [4.0, 0.0, 0.4, 95.0, 55.0, 36.1, 95.0, 70.0, 0.6, 135.0, 3.5, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
REF_HIGH = np.array(
    [11.0, 10.0, 1.8, 100.0, 100.0, 37.5, 140.0, 140.0, 1.2, 145.0, 5.1, 0.04, 0.5, 0.5],
    dtype=np.float64,
)

# rows = conditions, cols = biomarkers. Typical means used to sample labs.
CONDITION_SPEC: list[dict[str, Any]] = [
    {
        "condition": "URI",
        "popularity": 0.20,
        "admit_base": 0.03,
        "los_h": 1.4,
        "respiratory": True,
        "esi_p": [0.00, 0.04, 0.22, 0.48, 0.26],
        "symptoms": ["cough", "sore_throat", "rhinorrhea"],
        "labs": {
            "wbc": 7.2,
            "crp": 6.0,
            "lactate": 1.0,
            "spo2": 98.0,
            "hr": 78,
            "temp_c": 37.2,
            "sbp": 124,
            "glucose": 98,
            "creatinine": 0.9,
            "sodium": 139,
            "potassium": 4.1,
            "troponin": 0.01,
            "ua_nitrite": 0.05,
            "cxr_infiltrate": 0.05,
        },
        "protocol": {
            "wbc": 0.4,
            "crp": 0.3,
            "spo2": 1.0,
            "hr": 1.0,
            "temp_c": 1.0,
            "sbp": 1.0,
        },
    },
    {
        "condition": "Influenza",
        "popularity": 0.14,
        "admit_base": 0.08,
        "los_h": 2.2,
        "respiratory": True,
        "esi_p": [0.00, 0.10, 0.40, 0.38, 0.12],
        "symptoms": ["fever", "myalgia", "cough", "fatigue"],
        "labs": {
            "wbc": 5.8,
            "crp": 22.0,
            "lactate": 1.2,
            "spo2": 95.0,
            "hr": 96,
            "temp_c": 38.6,
            "sbp": 118,
            "glucose": 104,
            "creatinine": 0.95,
            "sodium": 137,
            "potassium": 4.0,
            "troponin": 0.01,
            "ua_nitrite": 0.05,
            "cxr_infiltrate": 0.12,
        },
        "protocol": {
            "wbc": 0.7,
            "crp": 0.6,
            "spo2": 1.0,
            "hr": 1.0,
            "temp_c": 1.0,
            "sbp": 1.0,
            "cxr_infiltrate": 0.4,
        },
    },
    {
        "condition": "Pneumonia",
        "popularity": 0.11,
        "admit_base": 0.42,
        "los_h": 18.0,
        "respiratory": True,
        "esi_p": [0.02, 0.28, 0.48, 0.18, 0.04],
        "symptoms": ["dyspnea", "fever", "cough", "chest_pain"],
        "labs": {
            "wbc": 14.5,
            "crp": 82.0,
            "lactate": 2.3,
            "spo2": 90.5,
            "hr": 112,
            "temp_c": 38.9,
            "sbp": 108,
            "glucose": 128,
            "creatinine": 1.15,
            "sodium": 134,
            "potassium": 3.9,
            "troponin": 0.02,
            "ua_nitrite": 0.05,
            "cxr_infiltrate": 0.82,
        },
        "protocol": {
            "wbc": 1.0,
            "crp": 0.9,
            "lactate": 1.0,
            "spo2": 1.0,
            "hr": 1.0,
            "temp_c": 1.0,
            "sbp": 1.0,
            "cxr_infiltrate": 1.0,
        },
    },
    {
        "condition": "UTI",
        "popularity": 0.13,
        "admit_base": 0.10,
        "los_h": 3.5,
        "respiratory": False,
        "esi_p": [0.00, 0.08, 0.42, 0.40, 0.10],
        "symptoms": ["dysuria", "frequency", "fever"],
        "labs": {
            "wbc": 11.2,
            "crp": 28.0,
            "lactate": 1.3,
            "spo2": 97.5,
            "hr": 92,
            "temp_c": 38.1,
            "sbp": 122,
            "glucose": 110,
            "creatinine": 1.05,
            "sodium": 138,
            "potassium": 4.2,
            "troponin": 0.01,
            "ua_nitrite": 0.78,
            "cxr_infiltrate": 0.04,
        },
        "protocol": {
            "wbc": 0.8,
            "crp": 0.5,
            "temp_c": 1.0,
            "hr": 1.0,
            "sbp": 1.0,
            "ua_nitrite": 1.0,
            "creatinine": 0.7,
        },
    },
    {
        "condition": "Gastroenteritis",
        "popularity": 0.12,
        "admit_base": 0.09,
        "los_h": 4.0,
        "respiratory": False,
        "esi_p": [0.00, 0.08, 0.38, 0.42, 0.12],
        "symptoms": ["vomiting", "diarrhea", "abdominal_pain"],
        "labs": {
            "wbc": 9.4,
            "crp": 12.0,
            "lactate": 1.9,
            "spo2": 98.0,
            "hr": 102,
            "temp_c": 37.6,
            "sbp": 106,
            "glucose": 92,
            "creatinine": 1.2,
            "sodium": 133,
            "potassium": 3.4,
            "troponin": 0.01,
            "ua_nitrite": 0.05,
            "cxr_infiltrate": 0.03,
        },
        "protocol": {
            "wbc": 0.5,
            "lactate": 0.8,
            "hr": 1.0,
            "sbp": 1.0,
            "creatinine": 0.8,
            "sodium": 0.8,
            "potassium": 0.8,
        },
    },
    {
        "condition": "Migraine",
        "popularity": 0.10,
        "admit_base": 0.02,
        "los_h": 2.0,
        "respiratory": False,
        "esi_p": [0.00, 0.05, 0.30, 0.45, 0.20],
        "symptoms": ["headache", "photophobia", "nausea"],
        "labs": {
            "wbc": 7.0,
            "crp": 3.0,
            "lactate": 0.9,
            "spo2": 99.0,
            "hr": 74,
            "temp_c": 36.8,
            "sbp": 128,
            "glucose": 96,
            "creatinine": 0.85,
            "sodium": 140,
            "potassium": 4.0,
            "troponin": 0.01,
            "ua_nitrite": 0.04,
            "cxr_infiltrate": 0.02,
        },
        "protocol": {"hr": 1.0, "sbp": 1.0, "temp_c": 0.6, "spo2": 0.6},
    },
    {
        "condition": "Cellulitis",
        "popularity": 0.09,
        "admit_base": 0.18,
        "los_h": 10.0,
        "respiratory": False,
        "esi_p": [0.00, 0.12, 0.48, 0.32, 0.08],
        "symptoms": ["skin_redness", "warmth", "fever"],
        "labs": {
            "wbc": 12.4,
            "crp": 46.0,
            "lactate": 1.5,
            "spo2": 97.0,
            "hr": 94,
            "temp_c": 37.9,
            "sbp": 126,
            "glucose": 118,
            "creatinine": 1.0,
            "sodium": 138,
            "potassium": 4.1,
            "troponin": 0.01,
            "ua_nitrite": 0.05,
            "cxr_infiltrate": 0.03,
        },
        "protocol": {"wbc": 1.0, "crp": 0.8, "temp_c": 1.0, "hr": 1.0, "lactate": 0.5},
    },
    {
        "condition": "Cardiac_rule_out",
        "popularity": 0.11,
        "admit_base": 0.55,
        "los_h": 16.0,
        "respiratory": False,
        "esi_p": [0.04, 0.42, 0.40, 0.12, 0.02],
        "symptoms": ["chest_pain", "diaphoresis", "dyspnea"],
        "labs": {
            "wbc": 8.4,
            "crp": 8.0,
            "lactate": 1.4,
            "spo2": 96.0,
            "hr": 92,
            "temp_c": 36.9,
            "sbp": 152,
            "glucose": 122,
            "creatinine": 1.05,
            "sodium": 139,
            "potassium": 4.3,
            "troponin": 0.09,
            "ua_nitrite": 0.04,
            "cxr_infiltrate": 0.08,
        },
        "protocol": {
            "troponin": 1.0,
            "hr": 1.0,
            "sbp": 1.0,
            "spo2": 1.0,
            "glucose": 0.5,
            "cxr_infiltrate": 0.5,
        },
    },
]


def _matrices() -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = [row["condition"] for row in CONDITION_SPEC]
    idx = {name: i for i, name in enumerate(BIOMARKERS)}
    means = np.zeros((len(CONDITION_SPEC), len(BIOMARKERS)), dtype=np.float64)
    protocol = np.zeros_like(means)
    for r, spec in enumerate(CONDITION_SPEC):
        for k, v in spec["labs"].items():
            means[r, idx[k]] = float(v)
        for k, v in spec["protocol"].items():
            protocol[r, idx[k]] = float(v)
    return means, protocol, names


@dataclass
class Clinic:
    """The whole clinic: lab matrices for NumPy, frames for Pandas, labels for ML/agents."""

    lab_means: np.ndarray
    protocol: np.ndarray
    assay_cost: np.ndarray
    ref_low: np.ndarray
    ref_high: np.ndarray
    inventory: np.ndarray
    condition_names: list[str]
    biomarker_names: list[str]
    sites: list[str]
    atlas: pd.DataFrame
    patients: pd.DataFrame
    encounters: pd.DataFrame
    daily: pd.DataFrame
    protocols: dict[str, dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)

    def condition_index(self, name: str) -> int:
        return self.condition_names.index(name)

    def site_index(self, name: str) -> int:
        return self.sites.index(name)

    @property
    def panel_cost(self) -> np.ndarray:
        return self.protocol @ self.assay_cost


def _atlas_frame(means: np.ndarray, protocol: np.ndarray) -> pd.DataFrame:
    cost = protocol @ ASSAY_COST
    rows = []
    for i, spec in enumerate(CONDITION_SPEC):
        rows.append(
            {
                "condition": spec["condition"],
                "popularity": spec["popularity"],
                "admit_base": spec["admit_base"],
                "typical_los_h": spec["los_h"],
                "respiratory": spec["respiratory"],
                "panel_cost": round(float(cost[i]), 2),
                "symptoms": ", ".join(spec["symptoms"]),
            }
        )
    return pd.DataFrame(rows)


def _patients(rng: np.random.Generator, n: int = 1600) -> pd.DataFrame:
    sex = rng.choice(["F", "M"], size=n, p=[0.52, 0.48])
    age = np.clip(rng.normal(42, 22, size=n), 1, 95).round(0).astype(int)
    home = rng.choice(SITES, size=n, p=[0.28, 0.12, 0.22, 0.16, 0.22])
    payer = rng.choice(PAYERS, size=n, p=[0.12, 0.22, 0.48, 0.18])
    # Older patients more often medicare
    payer = np.where(age >= 65, rng.choice(PAYERS, size=n, p=[0.04, 0.08, 0.18, 0.70]), payer)
    return pd.DataFrame(
        {
            "patient_id": [f"P{i:04d}" for i in range(n)],
            "age": age,
            "sex": pd.Categorical(sex),
            "home_site": pd.Categorical(home, categories=SITES),
            "payer": pd.Categorical(payer, categories=PAYERS),
            "htn": rng.random(n) < (0.12 + 0.01 * np.maximum(age - 40, 0)),
            "dm": rng.random(n) < (0.06 + 0.008 * np.maximum(age - 40, 0)),
            "copd": rng.random(n) < (0.03 + 0.004 * np.maximum(age - 50, 0)),
            "cad": rng.random(n) < (0.02 + 0.006 * np.maximum(age - 50, 0)),
        }
    )


def _hourly_lambda(hour: int, site: str, is_weekend: bool, flu: bool) -> float:
    morning = np.exp(-0.5 * ((hour - 10.0) / 2.2) ** 2)
    evening = np.exp(-0.5 * ((hour - 18.5) / 2.4) ** 2)
    base = 1.1 * morning + 1.4 * evening + 0.25
    site_k = {"Downtown": 1.30, "Airport": 1.10, "Campus": 1.20, "Harbor": 0.92, "Suburb": 0.88}[site]
    if site == "Campus" and is_weekend:
        site_k *= 0.55
    if is_weekend and site != "Campus":
        site_k *= 1.12
    if flu:
        site_k *= 1.28
    if hour < 8 or hour > 22:
        return 0.12 * site_k
    return float(base * site_k * 2.6)


def build_clinic(days: int = 56, seed: int = SEED) -> Clinic:
    rng = np.random.default_rng(seed)
    lab_means, protocol, condition_names = _matrices()
    atlas = _atlas_frame(lab_means, protocol)
    patients = _patients(rng)
    spec_of = {s["condition"]: s for s in CONDITION_SPEC}
    idx = {name: i for i, name in enumerate(BIOMARKERS)}

    turn_k = np.array([0.92, 1.05, 1.00, 1.10, 1.15])[:, None]
    weekly = protocol.T @ (atlas["popularity"].to_numpy() * 380.0)
    inventory = np.clip(weekly * turn_k * rng.uniform(0.85, 1.25, size=(len(SITES), 1)), 30, None)

    start = pd.Timestamp("2026-05-04")
    dates = pd.date_range(start, periods=days, freq="D")
    cond_p = atlas["popularity"].to_numpy()
    cond_p = cond_p / cond_p.sum()

    site_arr_p = {
        "Downtown": [0.70, 0.18, 0.12],
        "Airport": [0.62, 0.28, 0.10],
        "Campus": [0.82, 0.06, 0.12],
        "Harbor": [0.68, 0.20, 0.12],
        "Suburb": [0.78, 0.10, 0.12],
    }
    site_wait = {"Downtown": 18.0, "Airport": 22.0, "Campus": 14.0, "Harbor": 16.0, "Suburb": 20.0}

    pids = patients["patient_id"].to_numpy()
    home_groups = {s: patients.loc[patients.home_site.eq(s), "patient_id"].to_numpy() for s in SITES}
    age_of = patients.set_index("patient_id")["age"].to_dict()
    copd_of = patients.set_index("patient_id")["copd"].to_dict()
    cad_of = patients.set_index("patient_id")["cad"].to_dict()
    dm_of = patients.set_index("patient_id")["dm"].to_dict()

    records: list[dict[str, Any]] = []
    eid = 20000
    season_by_day: dict[pd.Timestamp, str] = {}

    for day in dates:
        is_weekend = day.dayofweek >= 5
        season = rng.choice(SEASONS, p=[0.62, 0.26, 0.12])
        season_by_day[day] = season
        flu = season == "flu_wave"
        heat = season == "heat"
        for site in SITES:
            for hour in range(8, 23):
                lam = _hourly_lambda(hour, site, is_weekend, flu)
                n = int(rng.poisson(lam))
                if n == 0:
                    continue
                local = home_groups[site]
                pick_local = rng.random(n) < 0.80
                chosen = [rng.choice(local) if loc and len(local) else rng.choice(pids) for loc in pick_local]
                conds = rng.choice(condition_names, size=n, p=cond_p)
                if flu:
                    # tilt toward respiratory
                    resp = [c for c in condition_names if spec_of[c]["respiratory"]]
                    flip = rng.random(n) < 0.35
                    conds = np.where(flip, rng.choice(resp, size=n), conds)
                arr_p = np.array(site_arr_p[site], dtype=float)
                if heat:
                    arr_p = arr_p * np.array([0.9, 1.25, 1.0])
                    arr_p = arr_p / arr_p.sum()
                arrivals = rng.choice(ARRIVALS, size=n, p=arr_p)
                minute = rng.integers(0, 60, size=n)
                ts = pd.to_datetime(day) + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")

                for i in range(n):
                    cond = str(conds[i])
                    spec = spec_of[cond]
                    pid = str(chosen[i])
                    age = int(age_of[pid])
                    arrival = str(arrivals[i])
                    esi = int(rng.choice(ESI, p=spec["esi_p"]))
                    if arrival == "ambulance" and esi > 2:
                        esi = int(rng.choice([1, 2, 3], p=[0.1, 0.55, 0.35]))

                    rush = 8.0 if hour in (10, 11, 17, 18, 19) else 0.0
                    wait = float(np.clip(rng.normal(site_wait[site] + rush + (6 if flu else 0) - (10 if arrival == "ambulance" else 0) + 4 * (esi >= 4), 6.5), 2, 120))
                    if esi <= 2:
                        wait = min(wait, float(np.clip(rng.normal(12, 5), 1, 45)))

                    late_triage = bool(esi <= 2 and wait > 20)

                    # Labs from condition means + noise
                    mu = lab_means[condition_names.index(cond)].copy()
                    noise = rng.normal(0.0, 0.12, size=mu.shape) * np.maximum(np.abs(mu), 0.2)
                    labs = np.clip(mu + noise, 0, None)
                    if copd_of[pid]:
                        labs[idx["spo2"]] -= rng.uniform(1.5, 4.0)
                    if cad_of[pid] and cond == "Cardiac_rule_out":
                        labs[idx["troponin"]] += rng.uniform(0.02, 0.12)
                    if dm_of[pid]:
                        labs[idx["glucose"]] += rng.uniform(20, 80)
                    if age >= 75:
                        labs[idx["creatinine"]] += rng.uniform(0.1, 0.5)

                    # ~5% missing lactate / troponin (device / not ordered)
                    miss_lactate = rng.random() < 0.06
                    miss_trop = rng.random() < 0.08
                    lactate = None if miss_lactate else round(float(labs[idx["lactate"]]), 2)
                    trop = None if miss_trop else round(float(labs[idx["troponin"]]), 3)

                    spo2 = float(np.clip(labs[idx["spo2"]], 78, 100))
                    hr = float(np.clip(labs[idx["hr"]], 45, 180))
                    temp = float(np.clip(labs[idx["temp_c"]], 34.5, 41.2))
                    sbp = float(np.clip(labs[idx["sbp"]], 70, 220))

                    logit = (
                        spec["admit_base"] * 4
                        + 0.9 * (esi <= 2)
                        + 0.8 * (age >= 70)
                        + 1.1 * (spo2 < 92)
                        + 0.9 * ((lactate or 1.0) > 2.0)
                        + 0.7 * ((trop or 0.01) > 0.04)
                        + 0.4 * (arrival == "ambulance")
                        + 0.35 * bool(copd_of[pid] and spec["respiratory"])
                    )
                    p_admit = float(1 / (1 + np.exp(-logit + 2.2)))
                    admit = bool(rng.random() < p_admit)

                    los = spec["los_h"] * (1.4 if admit else 0.35)
                    los = float(np.clip(rng.normal(los, los * 0.25), 0.4, 72))

                    symptoms = list(spec["symptoms"])
                    if rng.random() < 0.15:
                        extra = rng.choice(["fatigue", "nausea", "dizziness"])
                        if extra not in symptoms:
                            symptoms.append(str(extra))

                    records.append(
                        {
                            "encounter_id": eid,
                            "ts": ts[i],
                            "date": day.normalize(),
                            "hour": hour,
                            "dow": int(day.dayofweek),
                            "is_weekend": is_weekend,
                            "site": site,
                            "patient_id": pid,
                            "age": age,
                            "condition": cond,
                            "esi": esi,
                            "arrival": arrival,
                            "symptoms": "|".join(symptoms),
                            "wait_min": round(wait, 1),
                            "late_triage": late_triage,
                            "admit": admit,
                            "los_h": round(los, 2),
                            "season": season,
                            "respiratory": bool(spec["respiratory"]),
                            "wbc": round(float(labs[idx["wbc"]]), 1),
                            "crp": round(float(labs[idx["crp"]]), 1),
                            "lactate": lactate,
                            "spo2": round(spo2, 1),
                            "hr": round(hr, 0),
                            "temp_c": round(temp, 1),
                            "sbp": round(sbp, 0),
                            "glucose": round(float(labs[idx["glucose"]]), 0),
                            "creatinine": round(float(labs[idx["creatinine"]]), 2),
                            "sodium": round(float(labs[idx["sodium"]]), 0),
                            "potassium": round(float(labs[idx["potassium"]]), 1),
                            "troponin": trop,
                            "ua_nitrite": round(float(np.clip(labs[idx["ua_nitrite"]], 0, 1)), 2),
                            "cxr_infiltrate": round(float(np.clip(labs[idx["cxr_infiltrate"]], 0, 1)), 2),
                        }
                    )
                    eid += 1

    enc = pd.DataFrame.from_records(records)
    enc["ts"] = pd.to_datetime(enc["ts"])
    enc["date"] = pd.to_datetime(enc["date"])
    enc["site"] = pd.Categorical(enc["site"], categories=SITES)
    enc["condition"] = pd.Categorical(enc["condition"], categories=condition_names)
    enc["arrival"] = pd.Categorical(enc["arrival"], categories=ARRIVALS)
    enc["season"] = pd.Categorical(enc["season"], categories=SEASONS)
    enc["esi"] = pd.Categorical(enc["esi"], categories=ESI, ordered=True)

    daily = (
        enc.groupby(["date", "site"], observed=True)
        .agg(
            encounters=("encounter_id", "count"),
            admits=("admit", "sum"),
            admit_rate=("admit", "mean"),
            avg_wait=("wait_min", "mean"),
            late_triage_rate=("late_triage", "mean"),
            avg_los=("los_h", "mean"),
            ambulance_share=("arrival", lambda s: float((s == "ambulance").mean())),
            avg_esi=("esi", lambda s: float(pd.Series(s).astype(float).mean())),
        )
        .reset_index()
    )
    season_df = pd.DataFrame({"date": list(season_by_day), "season": list(season_by_day.values())})
    daily = daily.merge(season_df, on="date", how="left")
    daily["dow"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = daily["dow"] >= 5
    daily["flu_wave"] = (daily["season"] == "flu_wave").astype(int)

    roll = (
        enc.groupby("patient_id")
        .agg(
            visits=("encounter_id", "count"),
            admits=("admit", "sum"),
            last_visit=("ts", "max"),
            avg_esi=("esi", lambda s: float(pd.Series(s).astype(float).mean())),
        )
        .reset_index()
    )
    patients = patients.merge(roll, on="patient_id", how="left")
    patients["visits"] = patients["visits"].fillna(0).astype(int)
    patients["admits"] = patients["admits"].fillna(0).astype(int)
    last_day = enc["ts"].max()
    patients["recency_days"] = (last_day - patients["last_visit"]).dt.days
    patients["recency_days"] = patients["recency_days"].fillna(999).astype(int)

    protocols = {
        spec["condition"]: {
            "symptoms": spec["symptoms"],
            "next_tests": [k for k, v in spec["protocol"].items() if v >= 0.7],
            "red_flags": _red_flags(spec["condition"]),
            "disposition_hint": "consider_admit" if spec["admit_base"] >= 0.25 else "likely_discharge",
        }
        for spec in CONDITION_SPEC
    }

    return Clinic(
        lab_means=lab_means,
        protocol=protocol,
        assay_cost=ASSAY_COST.copy(),
        ref_low=REF_LOW.copy(),
        ref_high=REF_HIGH.copy(),
        inventory=inventory.astype(np.float64),
        condition_names=condition_names,
        biomarker_names=list(BIOMARKERS),
        sites=list(SITES),
        atlas=atlas,
        patients=patients,
        encounters=enc,
        daily=daily,
        protocols=protocols,
        meta={
            "seed": seed,
            "days": days,
            "start": str(dates[0].date()),
            "end": str(dates[-1].date()),
            "n_encounters": int(len(enc)),
            "disclaimer": "Synthetic educational data. Not medical advice. Not for clinical use.",
        },
    )


def _red_flags(condition: str) -> list[str]:
    common = ["spo2 < 92", "sbp < 90", "lactate > 2.0"]
    extra = {
        "Pneumonia": ["cxr_infiltrate high + hypoxia"],
        "Cardiac_rule_out": ["troponin above URL", "ongoing chest pain"],
        "UTI": ["age >= 75 + fever"],
        "Influenza": ["copd + hypoxia"],
    }
    return common + extra.get(condition, [])
