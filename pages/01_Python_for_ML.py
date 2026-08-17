"""Python topics — every construct runs on Slice & Dice ticket/menu data."""

from __future__ import annotations

import io
import itertools
import json
from collections import Counter, defaultdict, deque, namedtuple
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Callable, Iterable, Literal

import pandas as pd
import streamlit as st

from pizza.quiz import ask
from pizza.ui import header, inject, load_shop, ok, pitfall, why

inject()
shop = load_shop()
orders = shop.orders
menu = shop.menu
SIZE_MULT = shop.meta["size_mult"]

# Shared working sample so every tab shows *data*, not just snippets.
TICKETS = (
    orders.loc[orders["date"] == orders["date"].min(), ["order_id", "ts", "store", "customer_id", "pizza", "size", "qty", "channel", "unit_price", "discount", "revenue", "weather", "rating"]]
    .head(12)
    .copy()
)
TICKETS["ts"] = pd.to_datetime(TICKETS["ts"])

header(
    "Module 01 · Python (all topics)",
    "Python on the pizza ledger",
    "Types, collections, functions, OOP, errors, files, datetime, stdlib — each one computed from the same shop tables.",
)

why(
    "Before NumPy/Pandas, you still shape records, maps, and loops. Every tab below uses real Slice & Dice tickets "
    f"from {shop.meta['start']} (first 12 of opening day) plus the 8-pie menu."
)

c1, c2, c3 = st.columns(3)
c1.metric("Working tickets", len(TICKETS))
c2.metric("Menu pies", len(menu))
c3.metric("Full ledger", f"{len(orders):,} rows")

st.markdown("**Example data this page uses**")
st.dataframe(TICKETS, hide_index=True, width="stretch")
with st.expander("Menu (example data)"):
    st.dataframe(menu, hide_index=True, width="stretch")

tabs = st.tabs(
    [
        "Types",
        "Collections",
        "Strings & numbers",
        "Control flow",
        "Functions",
        "Comprehensions",
        "OOP",
        "Exceptions & files",
        "Datetime",
        "Stdlib",
        "Generators & itertools",
        "Typing & decorators",
    ]
)

# ---------------------------------------------------------------------------
with tabs[0]:
    st.markdown("### Scalar types — one ticket, one value each")
    row = TICKETS.iloc[0]
    types_tbl = pd.DataFrame(
        [
            {"field": "order_id", "python value": str(int(row.order_id)), "type": "int", "ml use": "id, never a feature as-is"},
            {"field": "qty", "python value": str(int(row.qty)), "type": "int", "ml use": "numeric feature"},
            {"field": "unit_price", "python value": str(float(row.unit_price)), "type": "float", "ml use": "numeric / target cousin"},
            {"field": "pizza", "python value": str(row.pizza), "type": "str", "ml use": "categorical → one-hot"},
            {"field": "is_delivery", "python value": str(str(row.channel) == "delivery"), "type": "bool", "ml use": "0/1 flag"},
            {"field": "missing_note", "python value": "None", "type": "NoneType", "ml use": "NaN / optional label"},
        ]
    )
    st.dataframe(types_tbl, hide_index=True, width="stretch")
    st.code(
        """qty: int = 2
price: float = 13.5
name: str = "Pepperoni"
late: bool = False
note = None          # not 0, not "", not NaN until you put it in Pandas
""",
        language="python",
    )
    pitfall("`False`, `0`, `[]`, `None` are all falsy — but they mean different things in a feature table. Don't collapse them.")

# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("### list · tuple · set · dict — built from the menu")
    pies = menu["pizza"].tolist()
    prices = dict(zip(menu["pizza"], menu["base_price"]))
    veg = set(menu.loc[menu.veg, "pizza"])
    meat = set(pies) - veg
    first_combo = (pies[0], "M", 1)  # tuple = immutable record

    left, right = st.columns(2)
    with left:
        st.caption("list — ordered, mutable (menu order)")
        st.dataframe(pd.DataFrame({"index": range(len(pies)), "pizza": pies}), hide_index=True)
        st.caption("set — unique veg pies (hash membership)")
        st.dataframe(pd.DataFrame({"veg_set": sorted(veg)}), hide_index=True)
    with right:
        st.caption("dict — pizza → base price (O(1) lookup)")
        st.dataframe(pd.DataFrame({"pizza": list(prices), "base_price": list(prices.values())}), hide_index=True)
        st.caption("tuple — one immutable ticket key")
        st.dataframe(pd.DataFrame([{"pizza": first_combo[0], "size": first_combo[1], "qty": first_combo[2]}]), hide_index=True)

    st.markdown("**Set algebra on tonight's tickets vs the veg menu**")
    tonight = set(TICKETS["pizza"])
    st.dataframe(
        pd.DataFrame(
            {
                "operation": ["tonight ∩ veg", "tonight − veg", "veg − tonight"],
                "meaning": ["veg pies that sold", "meat pies that sold", "veg pies nobody ordered"],
                "result": [
                    ", ".join(sorted(tonight & veg)) or "—",
                    ", ".join(sorted(tonight - veg)) or "—",
                    ", ".join(sorted(veg - tonight)) or "—",
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.code(
        """pies = ["Margherita", "Pepperoni", ...]      # list
prices = {"Pepperoni": 13.5, ...}            # dict
veg = {"Margherita", "Veggie", ...}          # set
key = ("Pepperoni", "L", 1)                  # tuple
tonight & veg                                # intersection
""",
        language="python",
    )
    ok("dict/set for lookups and uniqueness. list when order matters. tuple as a dict key (lists cannot be keys).")

# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("### Strings, numbers, formatting — ticket line items")
    rows = []
    for r in TICKETS.itertuples(index=False):
        sku = f"{r.pizza[:3].upper()}-{r.size}-{r.qty}"
        label = f"{r.qty}× {r.pizza} ({r.size}) @ ${r.unit_price:.2f}"
        net = round(r.unit_price * r.qty * (1 - float(r.discount)), 2)
        rows.append(
            {
                "sku": sku,
                "label": label,
                "slug": r.pizza.lower().replace(" ", "_"),
                "discount_pct": f"{float(r.discount):.0%}",
                "net": net,
                "matches_ledger": abs(net - float(r.revenue)) < 0.02,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.code(
        """sku = f"{pizza[:3].upper()}-{size}-{qty}"
slug = pizza.lower().replace(" ", "_")
net = round(unit_price * qty * (1 - discount), 2)
label = f"{qty}× {pizza} ({size}) @ ${unit_price:.2f}"
""",
        language="python",
    )
    n = st.number_input("Integer math vs float (pies to box)", min_value=1, max_value=20, value=5)
    st.dataframe(
        pd.DataFrame(
            [
                {"expr": "n / 2", "value": str(n / 2), "type": type(n / 2).__name__},
                {"expr": "n // 2", "value": str(n // 2), "type": type(n // 2).__name__},
                {"expr": "n % 2", "value": str(n % 2), "type": type(n % 2).__name__},
                {"expr": "round(13.5 * 1.3, 2)", "value": str(round(13.5 * 1.3, 2)), "type": "float"},
            ]
        ),
        hide_index=True,
    )
    pitfall("`0.1 + 0.2 != 0.3`. Money: round to cents at display; keep float64 in the frame until then, or use integer cents.")

# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown("### if / elif / for / while — classify opening-day tickets")
    st.markdown("Python `if` on each row is the slow cousin of a Pandas mask. Learn it, then vectorize.")

    def bucket(channel: str, hour: int, weather: str) -> str:
        if channel != "delivery":
            return "counter"
        if weather == "rain" and hour >= 18:
            return "sla_risk"
        if hour in (12, 13, 18, 19, 20):
            return "rush"
        return "normal_delivery"

    classified = TICKETS.copy()
    classified["hour"] = classified["ts"].dt.hour
    classified["bucket"] = [
        bucket(str(c), int(h), str(w))
        for c, h, w in zip(classified["channel"], classified["hour"], classified["weather"])
    ]
    st.dataframe(
        classified[["order_id", "store", "pizza", "channel", "hour", "weather", "bucket"]],
        hide_index=True,
        width="stretch",
    )
    counts = classified["bucket"].value_counts().rename_axis("bucket").reset_index(name="n")
    st.dataframe(counts, hide_index=True)
    st.code(
        """def bucket(channel, hour, weather):
    if channel != "delivery":
        return "counter"
    elif weather == "rain" and hour >= 18:
        return "sla_risk"
    elif hour in (12, 13, 18, 19, 20):
        return "rush"
    else:
        return "normal_delivery"
""",
        language="python",
    )
    st.markdown("`for` over rows vs `while` (rare in ML — batch iterators use `for`).")
    total, i = 0.0, 0
    revs = TICKETS["revenue"].tolist()
    while i < len(revs) and total < 80:
        total += revs[i]
        i += 1
    st.dataframe(
        pd.DataFrame([{"loop": "while total < 80", "tickets_consumed": i, "running_revenue": round(total, 2)}]),
        hide_index=True,
    )

# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("### Functions — price a pie like the ledger does")

    def price_pie(base: float, size: str, qty: int = 1, discount: float = 0.0, **fees: float) -> float:
        """*qty* default 1; **fees* catch delivery_fee=1.5 etc."""
        sub = base * SIZE_MULT[size] * qty * (1.0 - discount)
        return round(sub + sum(fees.values()), 2)

    def apply_tier(base_discount: float, *stack: float) -> float:
        """*stack extra promos, cap at 25%."""
        return min(0.25, base_discount + sum(stack))

    m1, m2, m3, m4 = st.columns(4)
    pie = m1.selectbox("Pizza", menu["pizza"].tolist(), key="fn_pie")
    size = m2.selectbox("Size", list(SIZE_MULT), index=1, key="fn_size")
    qty = m3.slider("Qty", 1, 4, 1)
    tier = m4.selectbox("Loyalty extra", [0.0, 0.08, 0.12], format_func=lambda x: f"{x:.0%}")
    base = float(menu.loc[menu.pizza.eq(pie), "base_price"].iloc[0])
    disc = apply_tier(0.0, float(tier))
    delivery_fee = st.checkbox("Add $1.50 delivery fee", value=True)
    fees = {"delivery_fee": 1.5} if delivery_fee else {}
    quoted = price_pie(base, size, qty, disc, **fees)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "pizza": pie,
                    "size": size,
                    "qty": qty,
                    "base": base,
                    "size_mult": SIZE_MULT[size],
                    "discount": disc,
                    "fees": fees,
                    "quoted": quoted,
                }
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.code(
        """def price_pie(base, size, qty=1, discount=0.0, **fees):
    sub = base * SIZE_MULT[size] * qty * (1 - discount)
    return round(sub + sum(fees.values()), 2)

def apply_tier(base_discount, *stack):
    return min(0.25, base_discount + sum(stack))

price_pie(13.5, "L", 2, 0.08, delivery_fee=1.5)
""",
        language="python",
    )
    lamb = sorted(TICKETS.itertuples(index=False), key=lambda r: r.revenue, reverse=True)[:5]
    st.caption("lambda as a sort key — top 5 tickets by revenue")
    st.dataframe(
        pd.DataFrame([{"order_id": r.order_id, "pizza": r.pizza, "revenue": r.revenue} for r in lamb]),
        hide_index=True,
    )
    pitfall("`def f(extras=[])` mutates the same list across calls. Default to `None` and allocate inside, or use `field(default_factory=list)`.")

# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown("### Comprehensions, zip, enumerate, unpacking")
    price_map = {p: pr for p, pr in zip(menu["pizza"], menu["base_price"])}
    veg_under_15 = [p for p, pr, v in zip(menu["pizza"], menu["base_price"], menu["veg"]) if v and pr < 15]
    enum_tbl = pd.DataFrame(
        [{"rank": i, "pizza": p, "base_price": price_map[p]} for i, p in enumerate(menu["pizza"], start=1)]
    )
    st.dataframe(enum_tbl, hide_index=True, width="stretch")
    st.markdown(f"Veg pies under $15: **{', '.join(veg_under_15)}**")

    a, *mid, z = menu["pizza"].tolist()
    st.dataframe(
        pd.DataFrame({"unpack": ["first", "middle*", "last"], "value": [a, ", ".join(mid), z]}),
        hide_index=True,
    )

    nested = [[str(p), str(s)] for p in TICKETS["pizza"].unique()[:3] for s in ["M", "L"]]
    st.caption("Nested comprehension = pizza × size (same idea as OneHot crosses)")
    st.dataframe(pd.DataFrame(nested, columns=["pizza", "size"]), hide_index=True)

    st.code(
        """price_map = {p: pr for p, pr in zip(pizzas, prices)}
veg_under_15 = [p for p, pr, v in zip(pizzas, prices, veg) if v and pr < 15]
for i, name in enumerate(pizzas, start=1):
    ...
first, *rest, last = pizzas
cross = [[p, s] for p in pies for s in sizes]
""",
        language="python",
    )

# ---------------------------------------------------------------------------
with tabs[6]:
    st.markdown("### OOP — Ticket record + a tiny OrderBook")

    @dataclass
    class Ticket:
        pizza: str
        size: str
        qty: int = 1
        toppings: list[str] = field(default_factory=list)
        channel: str = "dine_in"

        @property
        def veg_requested(self) -> bool:
            meat = {"pepperoni", "ham", "sausage", "chicken"}
            return meat.isdisjoint(self.toppings)

        def line_total(self, base: float, size_mult: dict[str, float]) -> float:
            return round(base * size_mult[self.size] * self.qty, 2)

    class OrderBook:
        def __init__(self) -> None:
            self._rows: list[Ticket] = []

        def add(self, ticket: Ticket) -> None:
            self._rows.append(ticket)

        def __len__(self) -> int:
            return len(self._rows)

        def __iter__(self):
            return iter(self._rows)

        def revenue(self, price_of: dict[str, float], size_mult: dict[str, float]) -> float:
            return sum(t.line_total(price_of[t.pizza], size_mult) for t in self._rows)

    book = OrderBook()
    price_of = dict(zip(menu["pizza"].astype(str), menu["base_price"].astype(float)))
    built = []
    for r in TICKETS.itertuples(index=False):
        t = Ticket(str(r.pizza), str(r.size), int(r.qty), channel=str(r.channel))
        book.add(t)
        built.append(
            {
                "pizza": t.pizza,
                "size": t.size,
                "qty": t.qty,
                "channel": t.channel,
                "veg_requested": t.veg_requested,
                "line_total": t.line_total(price_of[t.pizza], SIZE_MULT),
                "asdict": str(asdict(t)),
            }
        )
    st.dataframe(pd.DataFrame(built), hide_index=True, width="stretch")
    st.metric("OrderBook revenue (no discounts)", f"${book.revenue(price_of, SIZE_MULT):,.2f}")
    st.caption(f"len(book) = {len(book)}  ·  iterating a class that implements __iter__")
    st.code(
        """@dataclass
class Ticket:
    pizza: str
    size: str
    qty: int = 1
    toppings: list[str] = field(default_factory=list)

    def line_total(self, base, size_mult):
        return round(base * size_mult[self.size] * self.qty, 2)

class OrderBook:
    def add(self, ticket): ...
    def __len__(self): ...
    def __iter__(self): ...
""",
        language="python",
    )
    ok("Dataclass = record. Class with __len__/__iter__ = tiny collection. sklearn estimators are just classes with fit/predict.")

# ---------------------------------------------------------------------------
with tabs[7]:
    st.markdown("### Exceptions, JSON, pathlib — a ticket payload")

    def parse_size(raw: str) -> str:
        allowed = {"S", "M", "L", "XL"}
        if raw not in allowed:
            raise ValueError(f"size {raw!r} not in {sorted(allowed)}")
        return raw

    trial = st.selectbox("Try parsing size", ["M", "L", "XXL", ""], key="ex_size")
    try:
        ok_size = parse_size(trial)
        err_tbl = pd.DataFrame([{"input": trial, "status": "ok", "parsed": str(ok_size), "error": ""}])
    except ValueError as exc:
        err_tbl = pd.DataFrame([{"input": trial, "status": "ValueError", "parsed": "", "error": str(exc)}])
    st.dataframe(err_tbl, hide_index=True, width="stretch")

    payload = {
        "shop": "Slice & Dice",
        "n_tickets": int(len(TICKETS)),
        "tickets": [
            {"id": int(r.order_id), "pizza": str(r.pizza), "rev": float(r.revenue)}
            for r in TICKETS.itertuples(index=False)
        ],
    }
    blob = json.dumps(payload, indent=2)
    st.markdown("**json.dumps → str, json.loads → dict, then back to a table**")
    st.code(blob[:500] + ("\n…" if len(blob) > 500 else ""), language="json")
    loaded = json.loads(blob)
    st.dataframe(pd.DataFrame(loaded["tickets"]), hide_index=True, width="stretch")

    buf = io.StringIO()
    TICKETS.to_csv(buf, index=False)
    st.download_button("Download these 12 tickets as CSV", buf.getvalue(), "opening_day_sample.csv", "text/csv")
    st.caption(f"pathlib example: Path('data') / 'orders.csv' → `{Path('data') / 'orders.csv'}` (this lab generates data in memory, no file required).")
    st.code(
        """try:
    size = parse_size(raw)
except ValueError as exc:
    log(exc); size = "M"          # fallback — be explicit

json.dumps(payload)
json.loads(text)
Path("data") / "orders.csv"
""",
        language="python",
    )

# ---------------------------------------------------------------------------
with tabs[8]:
    st.markdown("### datetime — rush windows from timestamps")
    ts = TICKETS.copy()
    ts["hour"] = ts["ts"].dt.hour
    ts["dow"] = ts["ts"].dt.day_name()
    ts["iso"] = ts["ts"].dt.strftime("%Y-%m-%dT%H:%M")
    ts["plus_40m"] = ts["ts"] + pd.to_timedelta(40, unit="m")
    ts["is_weekend"] = ts["ts"].dt.dayofweek >= 5
    st.dataframe(ts[["order_id", "ts", "iso", "hour", "dow", "is_weekend", "plus_40m"]], hide_index=True, width="stretch")

    now = datetime.fromisoformat("2026-05-04T19:10:00")
    sla = now + timedelta(minutes=40)
    st.dataframe(
        pd.DataFrame(
            [
                {"clock": "quoted_at", "value": now.isoformat(sep=" ")},
                {"clock": "sla_deadline", "value": sla.isoformat(sep=" ")},
                {"clock": "weekday", "value": now.strftime("%A")},
            ]
        ),
        hide_index=True,
    )
    st.code(
        """from datetime import datetime, timedelta
quoted = datetime.fromisoformat("2026-05-04T19:10:00")
deadline = quoted + timedelta(minutes=40)
hour = quoted.hour
is_weekend = quoted.weekday() >= 5
""",
        language="python",
    )
    ok("In Pandas you use `.dt` on a column; in raw Python you use `datetime`. Same fields become ML calendar features.")

# ---------------------------------------------------------------------------
with tabs[9]:
    st.markdown("### collections — Counter, defaultdict, deque, namedtuple")
    c = Counter(TICKETS["pizza"].tolist())
    by_store: dict[str, list[float]] = defaultdict(list)
    for r in TICKETS.itertuples(index=False):
        by_store[str(r.store)].append(float(r.revenue))
    store_tbl = pd.DataFrame(
        {
            "store": list(by_store),
            "n": [len(v) for v in by_store.values()],
            "revenue": [round(sum(v), 2) for v in by_store.values()],
        }
    )
    st.caption("Counter — pizza histogram on the 12 tickets")
    st.dataframe(
        pd.DataFrame(c.most_common(), columns=["pizza", "n"]),
        hide_index=True,
        width="stretch",
    )
    st.caption("defaultdict(list) — store → revenues")
    st.dataframe(store_tbl, hide_index=True, width="stretch")

    Line = namedtuple("Line", "pizza size qty")
    named = [Line(str(r.pizza), str(r.size), int(r.qty)) for r in TICKETS.itertuples(index=False)]
    st.caption("namedtuple — lightweight record (attribute access, no methods)")
    st.dataframe(pd.DataFrame(named), hide_index=True, width="stretch")

    q = deque(TICKETS["order_id"].tolist(), maxlen=5)
    st.caption("deque(maxlen=5) — last five order ids (rolling window / streaming)")
    st.dataframe(pd.DataFrame({"last_5_order_id": list(q)}), hide_index=True)

    st.code(
        """from collections import Counter, defaultdict, deque, namedtuple
Counter(pizzas).most_common()
by_store = defaultdict(list); by_store[store].append(rev)
Line = namedtuple("Line", "pizza size qty")
recent = deque(maxlen=5)
""",
        language="python",
    )

# ---------------------------------------------------------------------------
with tabs[10]:
    st.markdown("### Generators & itertools — stream and cross")

    def stream_revenue(frame: pd.DataFrame, store: str) -> Iterable[float]:
        for r in frame.itertuples(index=False):
            if r.store == store:
                yield float(r.revenue)

    store = st.selectbox("Generator: stream this store from the FULL ledger", shop.stores, key="gen_store")
    gen_vals = list(itertools.islice(stream_revenue(orders, store), 8))
    st.caption("islice(generator, 8) — first 8 yields, without materializing the night")
    st.dataframe(pd.DataFrame({"yield_i": range(1, len(gen_vals) + 1), "revenue": gen_vals}), hide_index=True)
    st.metric(f"{store} full-window revenue (sum of generator)", f"${sum(stream_revenue(orders, store)):,.0f}")

    st.markdown("**itertools.product — pizza × size × channel (OneHot explosion)**")
    pn = st.slider("Pizzas", 2, 8, 3, key="it_p")
    sn = st.slider("Sizes", 1, 4, 2, key="it_s")
    grid = list(itertools.product(shop.pizza_names[:pn], ["S", "M", "L", "XL"][:sn], ["dine_in", "delivery"]))
    st.dataframe(pd.DataFrame(grid, columns=["pizza", "size", "channel"]), hide_index=True, width="stretch", height=260)
    st.caption(f"{len(grid)} cells = {pn} × {sn} × 2")

    st.markdown("**groupby (itertools) on opening-day pizzas**")
    sorted_p = sorted(TICKETS.itertuples(index=False), key=lambda r: r.pizza)
    gb_rows = []
    for pizza, grp in itertools.groupby(sorted_p, key=lambda r: r.pizza):
        items = list(grp)
        gb_rows.append({"pizza": pizza, "n": len(items), "revenue": round(sum(x.revenue for x in items), 2)})
    st.dataframe(pd.DataFrame(gb_rows), hide_index=True, width="stretch")
    st.code(
        """def stream_revenue(orders, store):
    for row in orders:
        if row.store == store:
            yield row.revenue

itertools.islice(stream, 8)
itertools.product(pizzas, sizes, channels)
itertools.groupby(sorted(rows, key=pizza), key=pizza)
""",
        language="python",
    )

# ---------------------------------------------------------------------------
with tabs[11]:
    st.markdown("### Typing, decorators, context managers")

    Channel = Literal["dine_in", "takeout", "delivery"]

    def quote(pizza: str, size: str, qty: int = 1, channel: Channel = "dine_in") -> float:
        base = float(menu.loc[menu.pizza.eq(pizza), "base_price"].iloc[0])
        return round(base * SIZE_MULT[size] * qty, 2)

    def timed(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            wrapper.calls = getattr(wrapper, "calls", 0) + 1  # type: ignore[attr-defined]
            return out

        return wrapper

    @timed
    def top_pizza(frame: pd.DataFrame) -> str:
        return str(Counter(frame["pizza"]).most_common(1)[0][0])

    @contextmanager
    def oven(temp_c: int):
        yield {"temp_c": temp_c, "status": "hot"}

    with oven(315) as state:
        oven_state = state

    typed_rows = []
    for r in TICKETS.itertuples(index=False):
        typed_rows.append(
            {
                "pizza": r.pizza,
                "size": r.size,
                "qty": r.qty,
                "channel": r.channel,
                "typed_quote": quote(str(r.pizza), str(r.size), int(r.qty), str(r.channel)),  # type: ignore[arg-type]
                "ledger_unit_x_qty": round(float(r.unit_price) * int(r.qty), 2),
            }
        )
    st.dataframe(pd.DataFrame(typed_rows), hide_index=True, width="stretch")
    _ = top_pizza(TICKETS)
    _ = top_pizza(TICKETS)
    st.dataframe(
        pd.DataFrame(
            [
                {"idea": "Literal[channel]", "example": "dine_in | takeout | delivery"},
                {"idea": "@timed calls", "example": str(getattr(top_pizza, "calls", 0))},
                {"idea": "contextmanager oven", "example": str(oven_state)},
                {"idea": "top_pizza()", "example": str(top_pizza(TICKETS))},
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.code(
        """from typing import Literal, Callable
Channel = Literal["dine_in", "takeout", "delivery"]

def quote(pizza: str, size: str, qty: int = 1, channel: Channel = "dine_in") -> float:
    ...

@timed
def top_pizza(frame) -> str:
    ...

with oven(315) as state:
    bake(state)
""",
        language="python",
    )
    pitfall("Type hints are not runtime checks (unless you use pydantic). A wrong channel still runs — sklearn will one-hot an unexpected level via handle_unknown.")

st.markdown("---")
ok(
    "You now have Python coverage on **the same 12 opening-day tickets + menu**: types, collections, strings, control flow, "
    "functions, comprehensions, OOP, exceptions/JSON, datetime, collections/itertools/generators, typing/decorators."
)
ask(
    "q1_python",
    "You need a {customer_id → last order timestamp} map from 10M log lines, once. First tool?",
    [
        "A pandas merge on the full frame, then unique",
        "A dict or defaultdict updated in one pass (or a generator + dict)",
        "A nested list of lists, then .index()",
    ],
    "A dict or defaultdict updated in one pass (or a generator + dict)",
    "One pass, O(n) memory for keys you actually saw. Pandas is right when you also need time-based windows and joins — not for a single lookup table.",
)
