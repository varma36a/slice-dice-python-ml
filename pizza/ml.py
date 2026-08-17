"""Shared feature tables so regression, classification, and the capstone stay consistent."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pizza.data import Shop


def daily_model_frame(shop: Shop) -> pd.DataFrame:
    """Store-day demand table with calendar + weather + lag features (no future leak)."""
    d = shop.daily.sort_values(["store", "date"]).copy()
    d["tickets_lag7"] = d.groupby("store", observed=True)["tickets"].shift(7)
    d["rev_lag7"] = d.groupby("store", observed=True)["revenue"].shift(7)
    d["tickets_roll7"] = d.groupby("store", observed=True)["tickets"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).mean()
    )
    d["month"] = d["date"].dt.month
    d["is_rain"] = (d["weather"] == "rain").astype(int)
    d["is_heat"] = (d["weather"] == "heat").astype(int)
    return d.dropna(subset=["tickets_lag7", "tickets_roll7"]).reset_index(drop=True)


def delivery_model_frame(shop: Shop) -> pd.DataFrame:
    """Ticket-level delivery table. Drop unknown late labels (missing times)."""
    o = shop.orders.copy()
    o = o[o["channel"] == "delivery"].copy()
    o = o.dropna(subset=["delivery_min"])
    o["late_int"] = (o["delivery_min"] > 40).astype(int)
    o["size"] = o["size"].astype(str)
    o["store"] = o["store"].astype(str)
    o["pizza"] = o["pizza"].astype(str)
    o["weather"] = o["weather"].astype(str)
    o["rush"] = o["hour"].isin([12, 13, 18, 19, 20]).astype(int)
    return o.reset_index(drop=True)


def customer_feature_frame(shop: Shop) -> pd.DataFrame:
    c = shop.customers.copy()
    c = c[c["orders"] > 0].copy()
    c["avg_ticket"] = np.where(c["orders"] > 0, c["spend"] / c["orders"], 0.0)
    c["log_spend"] = np.log1p(c["spend"])
    c["log_orders"] = np.log1p(c["orders"])
    return c.reset_index(drop=True)


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
                ["dow", "is_weekend", "is_rain", "is_heat", "tickets_lag7", "tickets_roll7"],
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["store", "weather"]),
        ]
    )


def delivery_transformer() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), ["hour", "qty", "is_weekend", "rush"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["store", "pizza", "size", "weather"],
            ),
        ]
    )
