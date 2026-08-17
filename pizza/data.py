"""Synthetic Slice & Dice Pizzeria universe.

One seeded generator so every lesson (NumPy → sklearn) shares the same shop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

SEED = 42

STORES = ["Downtown", "Airport", "Campus", "Harbor", "Suburb"]
CHANNELS = ["dine_in", "takeout", "delivery"]
WEATHER = ["clear", "rain", "heat"]
SIZES = ["S", "M", "L", "XL"]
TIERS = ["slice", "regular", "gold", "founders"]

INGREDIENTS = [
    "dough",
    "tomato_sauce",
    "mozzarella",
    "pepperoni",
    "mushrooms",
    "onions",
    "peppers",
    "olives",
    "chicken",
    "ham",
    "pineapple",
    "bbq_sauce",
    "sausage",
    "truffle_oil",
]

# Unit cost in $ per recipe unit (ballpark COGS, not a real commissary).
INGREDIENT_COST = np.array(
    [0.55, 0.40, 1.10, 0.95, 0.50, 0.20, 0.25, 0.35, 1.20, 0.85, 0.30, 0.35, 0.90, 1.80],
    dtype=np.float64,
)

SIZE_MULT = {"S": 0.75, "M": 1.00, "L": 1.30, "XL": 1.60}

# rows = pizzas, cols = ingredients. Values are recipe units for a medium pie.
MENU_SPEC: list[dict[str, Any]] = [
    {
        "pizza": "Margherita",
        "price": 11.0,
        "veg": True,
        "prep_min": 11,
        "popularity": 0.16,
        "recipe": {"dough": 1, "tomato_sauce": 1, "mozzarella": 1.0},
    },
    {
        "pizza": "Pepperoni",
        "price": 13.5,
        "veg": False,
        "prep_min": 12,
        "popularity": 0.18,
        "recipe": {"dough": 1, "tomato_sauce": 1, "mozzarella": 1.0, "pepperoni": 1.1},
    },
    {
        "pizza": "Quattro Formaggi",
        "price": 15.0,
        "veg": True,
        "prep_min": 13,
        "popularity": 0.10,
        "recipe": {"dough": 1, "tomato_sauce": 0.3, "mozzarella": 2.2},
    },
    {
        "pizza": "Veggie",
        "price": 13.0,
        "veg": True,
        "prep_min": 13,
        "popularity": 0.12,
        "recipe": {
            "dough": 1,
            "tomato_sauce": 1,
            "mozzarella": 0.9,
            "mushrooms": 0.8,
            "onions": 0.6,
            "peppers": 0.7,
            "olives": 0.4,
        },
    },
    {
        "pizza": "BBQ Chicken",
        "price": 16.5,
        "veg": False,
        "prep_min": 15,
        "popularity": 0.13,
        "recipe": {
            "dough": 1,
            "mozzarella": 1.0,
            "chicken": 1.1,
            "onions": 0.5,
            "bbq_sauce": 0.9,
        },
    },
    {
        "pizza": "Hawaiian",
        "price": 14.0,
        "veg": False,
        "prep_min": 12,
        "popularity": 0.09,
        "recipe": {
            "dough": 1,
            "tomato_sauce": 0.8,
            "mozzarella": 0.9,
            "ham": 0.9,
            "pineapple": 0.8,
        },
    },
    {
        "pizza": "Meat Lovers",
        "price": 18.0,
        "veg": False,
        "prep_min": 16,
        "popularity": 0.14,
        "recipe": {
            "dough": 1.1,
            "tomato_sauce": 1,
            "mozzarella": 1.1,
            "pepperoni": 0.8,
            "ham": 0.6,
            "sausage": 0.9,
        },
    },
    {
        "pizza": "Truffle Mushroom",
        "price": 19.5,
        "veg": True,
        "prep_min": 14,
        "popularity": 0.08,
        "recipe": {
            "dough": 1,
            "mozzarella": 1.2,
            "mushrooms": 1.3,
            "truffle_oil": 0.7,
        },
    },
]


def _recipe_matrix() -> tuple[np.ndarray, list[str]]:
    names = [row["pizza"] for row in MENU_SPEC]
    idx = {name: i for i, name in enumerate(INGREDIENTS)}
    recipes = np.zeros((len(MENU_SPEC), len(INGREDIENTS)), dtype=np.float64)
    for r, spec in enumerate(MENU_SPEC):
        for ingredient, qty in spec["recipe"].items():
            recipes[r, idx[ingredient]] = float(qty)
    return recipes, names


@dataclass
class Shop:
    """The whole pizzeria: matrices for NumPy, frames for Pandas, labels for ML."""

    recipes: np.ndarray
    ingredient_cost: np.ndarray
    inventory: np.ndarray
    pizza_names: list[str]
    ingredient_names: list[str]
    stores: list[str]
    menu: pd.DataFrame
    customers: pd.DataFrame
    orders: pd.DataFrame
    daily: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def cogs_per_medium(self) -> np.ndarray:
        return self.recipes @ self.ingredient_cost

    def pizza_index(self, name: str) -> int:
        return self.pizza_names.index(name)

    def store_index(self, name: str) -> int:
        return self.stores.index(name)


def _menu_frame() -> pd.DataFrame:
    recipes, _ = _recipe_matrix()
    cogs = recipes @ INGREDIENT_COST
    rows = []
    for i, spec in enumerate(MENU_SPEC):
        rows.append(
            {
                "pizza": spec["pizza"],
                "base_price": spec["price"],
                "veg": spec["veg"],
                "prep_min": spec["prep_min"],
                "popularity": spec["popularity"],
                "cogs_medium": round(float(cogs[i]), 3),
                "margin_medium": round(float(spec["price"] - cogs[i]), 3),
            }
        )
    return pd.DataFrame(rows)


def _customers(rng: np.random.Generator, n: int = 1400) -> pd.DataFrame:
    neighborhoods = {
        "Downtown": ["Midtown", "Arts District", "Old Port"],
        "Airport": ["Hangar Row", "Transit Village", "Cargo Park"],
        "Campus": ["North Quad", "Sorority Row", "Faculty Hill"],
        "Harbor": ["Pier 4", "Fisherman's", "Marina"],
        "Suburb": ["Oak Lane", "Maple Court", "Ridgeview"],
    }
    home_store = rng.choice(STORES, size=n, p=[0.28, 0.14, 0.22, 0.16, 0.20])
    hoods = [rng.choice(neighborhoods[s]) for s in home_store]
    signup = pd.to_datetime("2025-11-01") + pd.to_timedelta(rng.integers(0, 220, size=n), unit="D")
    loyalty_p = [0.42, 0.38, 0.15, 0.05]
    tier = rng.choice(TIERS, size=n, p=loyalty_p)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(n)],
            "home_store": home_store,
            "neighborhood": hoods,
            "signup_date": signup,
            "loyalty_tier": pd.Categorical(tier, categories=TIERS, ordered=True),
            "marketing_opt_in": rng.random(n) > 0.35,
        }
    )


def _hourly_lambda(hour: int, store: str, is_weekend: bool, rain: bool) -> float:
    lunch = np.exp(-0.5 * ((hour - 12.5) / 1.6) ** 2)
    dinner = np.exp(-0.5 * ((hour - 19.0) / 2.0) ** 2)
    late = 0.35 * np.exp(-0.5 * ((hour - 22.5) / 1.2) ** 2)
    base = 1.2 * lunch + 1.8 * dinner + late
    store_k = {"Downtown": 1.35, "Airport": 1.15, "Campus": 1.25, "Harbor": 0.95, "Suburb": 0.85}[store]
    if store == "Campus" and is_weekend:
        store_k *= 0.55
    if store == "Airport":
        base += 0.35  # relatively flat day
    if is_weekend and store != "Campus":
        store_k *= 1.18
    if rain:
        store_k *= 1.12  # more delivery, slightly more tickets
    if hour < 11 or hour > 23:
        return 0.08 * store_k
    return float(base * store_k * 3.4)


def build_shop(days: int = 56, seed: int = SEED) -> Shop:
    rng = np.random.default_rng(seed)
    recipes, pizza_names = _recipe_matrix()
    menu = _menu_frame()
    customers = _customers(rng)

    # Stores x ingredients. Downtown turns faster → slightly leaner stock.
    turn_k = np.array([0.92, 1.05, 1.00, 1.10, 1.15])[:, None]
    weekly_need = recipes.T @ (menu["popularity"].to_numpy() * 420.0)
    inventory = np.clip(weekly_need * turn_k * rng.uniform(0.85, 1.25, size=(len(STORES), 1)), 40, None)
    inventory = inventory.astype(np.float64)

    start = pd.Timestamp("2026-05-04")  # Monday
    dates = pd.date_range(start, periods=days, freq="D")
    pizza_p = menu["popularity"].to_numpy()
    pizza_p = pizza_p / pizza_p.sum()

    store_channel_p = {
        "Downtown": [0.34, 0.22, 0.44],
        "Airport": [0.12, 0.55, 0.33],
        "Campus": [0.28, 0.40, 0.32],
        "Harbor": [0.40, 0.18, 0.42],
        "Suburb": [0.22, 0.18, 0.60],
    }
    store_delay = {"Downtown": 8.0, "Airport": 12.0, "Campus": 6.5, "Harbor": 10.0, "Suburb": 16.0}

    cust_ids = customers["customer_id"].to_numpy()
    tier_of = customers.set_index("customer_id")["loyalty_tier"].to_dict()
    spec_of = {s["pizza"]: s for s in MENU_SPEC}
    home_groups = {s: customers.loc[customers.home_store == s, "customer_id"].to_numpy() for s in STORES}

    records: list[dict[str, Any]] = []
    order_id = 10000
    weather_by_day: dict[pd.Timestamp, str] = {}

    for day in dates:
        is_weekend = day.dayofweek >= 5
        w = rng.choice(WEATHER, p=[0.70, 0.22, 0.08])
        weather_by_day[day] = w
        rain = w == "rain"
        heat = w == "heat"
        for store in STORES:
            for hour in range(11, 24):
                lam = _hourly_lambda(hour, store, is_weekend, rain)
                n = int(rng.poisson(lam))
                if n == 0:
                    continue
                local = home_groups[store]
                pick_local = rng.random(n) < 0.82
                chosen = []
                for loc in pick_local:
                    if loc and len(local):
                        chosen.append(rng.choice(local))
                    else:
                        chosen.append(rng.choice(cust_ids))
                pizzas = rng.choice(pizza_names, size=n, p=pizza_p)
                sizes = rng.choice(SIZES, size=n, p=[0.12, 0.46, 0.30, 0.12])
                qtys = rng.choice([1, 2, 3], size=n, p=[0.78, 0.18, 0.04])
                ch_p = np.array(store_channel_p[store], dtype=float)
                if rain:
                    ch_p = ch_p * np.array([0.7, 0.9, 1.35])
                    ch_p = ch_p / ch_p.sum()
                channels = rng.choice(CHANNELS, size=n, p=ch_p)
                minute = rng.integers(0, 60, size=n)
                ts = pd.to_datetime(day) + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")

                for i in range(n):
                    pizza = str(pizzas[i])
                    spec = spec_of[pizza]
                    size = str(sizes[i])
                    qty = int(qtys[i])
                    channel = str(channels[i])
                    unit = spec["price"] * SIZE_MULT[size]
                    cid = str(chosen[i])
                    tier = tier_of[cid]
                    discount = 0.0
                    if tier == "gold":
                        discount = 0.08
                    elif tier == "founders":
                        discount = 0.12
                    if store == "Campus" and hour >= 21:
                        discount = max(discount, 0.10)
                    if rng.random() < 0.04:
                        discount = max(discount, 0.15)

                    prep = spec["prep_min"] * (0.9 if size == "S" else 1.0 if size == "M" else 1.12 if size == "L" else 1.25)
                    rush = 9.0 if hour in (12, 13, 18, 19, 20) else 0.0
                    traffic = store_delay[store]
                    weather_pen = 13.0 if rain else (6.0 if heat else 0.0)
                    if channel != "delivery":
                        delivery_min = np.nan
                        late = False
                    else:
                        mu = prep + traffic + rush + weather_pen
                        delivery_min = float(np.clip(rng.normal(mu, 6.2), 12, 85))
                        late = delivery_min > 40.0

                    rating = rng.choice([3, 4, 5], p=[0.08, 0.32, 0.60])
                    if late:
                        rating = int(np.clip(rating - rng.choice([1, 2], p=[0.6, 0.4]), 1, 5))
                    tip = 0.0
                    if channel == "delivery":
                        tip_base = 2.0 + 0.08 * unit * qty
                        if late:
                            tip_base *= 0.35
                        if tier in ("gold", "founders"):
                            tip_base *= 1.15
                        tip = float(max(0.0, rng.normal(tip_base, 0.8)))

                    # ~4% missing delivery times (tablet glitch) — Pandas lesson.
                    if channel == "delivery" and rng.random() < 0.04:
                        delivery_min = np.nan

                    records.append(
                        {
                            "order_id": order_id,
                            "ts": ts[i],
                            "date": day.normalize(),
                            "hour": hour,
                            "dow": int(day.dayofweek),
                            "is_weekend": is_weekend,
                            "store": store,
                            "customer_id": cid,
                            "pizza": pizza,
                            "size": size,
                            "qty": qty,
                            "channel": channel,
                            "unit_price": round(unit, 2),
                            "discount": round(discount, 2),
                            "revenue": round(unit * qty * (1.0 - discount), 2),
                            "delivery_min": None if np.isnan(delivery_min) else round(delivery_min, 1),
                            "late": bool(late) if channel == "delivery" and not np.isnan(delivery_min) else pd.NA,
                            "rating": rating,
                            "tip": round(tip, 2),
                            "weather": w,
                            "veg": bool(spec["veg"]),
                        }
                    )
                    order_id += 1

    orders = pd.DataFrame.from_records(records)
    orders["ts"] = pd.to_datetime(orders["ts"])
    orders["date"] = pd.to_datetime(orders["date"])
    orders["size"] = pd.Categorical(orders["size"], categories=SIZES, ordered=True)
    orders["channel"] = pd.Categorical(orders["channel"], categories=CHANNELS)
    orders["weather"] = pd.Categorical(orders["weather"], categories=WEATHER)
    orders["store"] = pd.Categorical(orders["store"], categories=STORES)
    orders["late"] = orders["late"].astype("boolean")

    daily = (
        orders.groupby(["date", "store"], observed=True)
        .agg(
            tickets=("order_id", "count"),
            pies=("qty", "sum"),
            revenue=("revenue", "sum"),
            avg_ticket=("revenue", "mean"),
            delivery_share=("channel", lambda s: float((s == "delivery").mean())),
            late_rate=("late", lambda s: float(pd.Series(s).fillna(False).mean())),
            avg_delivery=("delivery_min", "mean"),
            avg_rating=("rating", "mean"),
            tips=("tip", "sum"),
        )
        .reset_index()
    )
    weather_df = pd.DataFrame({"date": list(weather_by_day), "weather": list(weather_by_day.values())})
    daily = daily.merge(weather_df, on="date", how="left")
    daily["dow"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = daily["dow"] >= 5

    # Attach a few customer rollups used by clustering.
    cust_roll = (
        orders.groupby("customer_id")
        .agg(
            orders=("order_id", "count"),
            spend=("revenue", "sum"),
            avg_rating=("rating", "mean"),
            last_order=("ts", "max"),
            veg_share=("veg", "mean"),
            delivery_share=("channel", lambda s: float((s == "delivery").mean())),
        )
        .reset_index()
    )
    customers = customers.merge(cust_roll, on="customer_id", how="left")
    customers["orders"] = customers["orders"].fillna(0).astype(int)
    customers["spend"] = customers["spend"].fillna(0.0)
    last_day = orders["ts"].max()
    customers["recency_days"] = (last_day - customers["last_order"]).dt.days
    customers["recency_days"] = customers["recency_days"].fillna(999).astype(int)

    return Shop(
        recipes=recipes,
        ingredient_cost=INGREDIENT_COST.copy(),
        inventory=inventory,
        pizza_names=pizza_names,
        ingredient_names=list(INGREDIENTS),
        stores=list(STORES),
        menu=menu,
        customers=customers,
        orders=orders,
        daily=daily,
        meta={
            "seed": seed,
            "days": days,
            "start": str(dates[0].date()),
            "end": str(dates[-1].date()),
            "n_orders": int(len(orders)),
            "size_mult": SIZE_MULT,
        },
    )
