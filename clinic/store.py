"""Load a pre-baked clinic from data/ so Cloud does not generate 10k rows on boot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from clinic.data import BIOMARKERS, SITES, Clinic

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _cloud_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Write dtypes pandas 2.2 on Streamlit Cloud can read."""
    out = df.copy()
    for col in out.columns:
        name = str(out[col].dtype)
        if name == "category":
            if col == "esi":
                out[col] = out[col].astype("int64")
            else:
                out[col] = out[col].astype("string")
        elif name in {"str", "string"}:
            out[col] = out[col].astype("string")
        elif name == "boolean":
            out[col] = out[col].astype("bool")
        elif name.startswith("datetime64"):
            out[col] = pd.to_datetime(out[col]).astype("datetime64[ns]")
    return out


def save_clinic(clinic: Clinic, dest: Path | None = None) -> Path:
    dest = dest or DATA_DIR
    dest.mkdir(parents=True, exist_ok=True)
    _cloud_frame(clinic.atlas).to_parquet(dest / "atlas.parquet", index=False)
    _cloud_frame(clinic.patients).to_parquet(dest / "patients.parquet", index=False)
    _cloud_frame(clinic.encounters).to_parquet(dest / "encounters.parquet", index=False)
    _cloud_frame(clinic.daily).to_parquet(dest / "daily.parquet", index=False)
    np.savez(
        dest / "matrices.npz",
        lab_means=clinic.lab_means,
        protocol=clinic.protocol,
        assay_cost=clinic.assay_cost,
        ref_low=clinic.ref_low,
        ref_high=clinic.ref_high,
        inventory=clinic.inventory,
    )
    (dest / "meta.json").write_text(
        json.dumps(
            {
                "meta": clinic.meta,
                "condition_names": clinic.condition_names,
                "biomarker_names": clinic.biomarker_names,
                "sites": clinic.sites,
                "protocols": clinic.protocols,
            },
            default=str,
        ),
        encoding="utf-8",
    )
    return dest


def baked_clinic(src: Path | None = None) -> Clinic | None:
    src = src or DATA_DIR
    if not (src / "encounters.parquet").exists():
        return None
    blob = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    mats = np.load(src / "matrices.npz")
    return Clinic(
        lab_means=mats["lab_means"],
        protocol=mats["protocol"],
        assay_cost=mats["assay_cost"],
        ref_low=mats["ref_low"],
        ref_high=mats["ref_high"],
        inventory=mats["inventory"],
        condition_names=list(blob["condition_names"]),
        biomarker_names=list(blob.get("biomarker_names") or BIOMARKERS),
        sites=list(blob.get("sites") or SITES),
        atlas=pd.read_parquet(src / "atlas.parquet"),
        patients=pd.read_parquet(src / "patients.parquet"),
        encounters=pd.read_parquet(src / "encounters.parquet"),
        daily=pd.read_parquet(src / "daily.parquet"),
        protocols=blob["protocols"],
        meta=blob["meta"],
    )
