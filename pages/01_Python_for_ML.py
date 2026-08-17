"""Python topics — every construct runs on Northshore encounter/atlas data."""

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

from clinic.quiz import ask
from clinic.ui import header, inject, load_clinic, ok, pitfall, warn, why

inject()
clinic = load_clinic()
enc = clinic.encounters
atlas = clinic.atlas

CASES = (
    enc.loc[enc["date"] == enc["date"].min(),
            ["encounter_id", "ts", "site", "patient_id", "age", "condition", "esi", "arrival",
             "wait_min", "admit", "spo2", "hr", "season"]]
    .head(12)
    .copy()
)
CASES["ts"] = pd.to_datetime(CASES["ts"])

header(
    "Module 01 · Python (all topics)",
    "Python on the encounter chart",
    "Types through typing — each tab computed from the same 12 opening-day encounters plus the condition atlas.",
)
warn("Synthetic educational data. Not medical advice.")
why("sklearn wants numeric arrays. The glue — records, maps, generators, JSON payloads — is still Python. Agents are just that glue with a loop.")

c1, c2, c3 = st.columns(3)
c1.metric("Working encounters", len(CASES))
c2.metric("Atlas rows", len(atlas))
c3.metric("Full ledger", f"{len(enc):,}")
st.markdown("**Example data this page uses**")
st.dataframe(CASES, hide_index=True, width="stretch")
with st.expander("Condition atlas"):
    st.dataframe(atlas, hide_index=True, width="stretch")

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

with tabs[0]:
    row = CASES.iloc[0]
    st.dataframe(
        pd.DataFrame(
            [
                {"field": "encounter_id", "python value": str(int(row.encounter_id)), "type": "int", "ml use": "id, never a raw feature"},
                {"field": "age", "python value": str(int(row.age)), "type": "int", "ml use": "numeric feature"},
                {"field": "spo2", "python value": str(float(row.spo2)), "type": "float", "ml use": "vital / risk"},
                {"field": "condition", "python value": str(row.condition), "type": "str", "ml use": "label or one-hot (careful)"},
                {"field": "admit", "python value": str(bool(row.admit)), "type": "bool", "ml use": "target"},
                {"field": "pending_note", "python value": "None", "type": "NoneType", "ml use": "missing lab ≠ 0"},
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    pitfall("`False`, `0`, `[]`, `None` are all falsy. A missing troponin is not a negative troponin.")

with tabs[1]:
    names = atlas["condition"].tolist()
    los = dict(zip(atlas["condition"], atlas["typical_los_h"]))
    resp = set(atlas.loc[atlas.respiratory, "condition"])
    st.dataframe(pd.DataFrame({"index": range(len(names)), "condition": names}), hide_index=True)
    left, right = st.columns(2)
    with left:
        st.caption("dict — condition → typical LOS")
        st.dataframe(pd.DataFrame({"condition": list(los), "typical_los_h": list(los.values())}), hide_index=True)
    with right:
        st.caption("set — respiratory conditions")
        st.dataframe(pd.DataFrame({"respiratory": sorted(resp)}), hide_index=True)
    tonight = set(CASES["condition"].astype(str))
    st.dataframe(
        pd.DataFrame(
            {
                "operation": ["tonight ∩ respiratory", "tonight − respiratory", "respiratory − tonight"],
                "result": [
                    ", ".join(sorted(tonight & resp)) or "—",
                    ", ".join(sorted(tonight - resp)) or "—",
                    ", ".join(sorted(resp - tonight)) or "—",
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )

with tabs[2]:
    rows = []
    for r in CASES.itertuples(index=False):
        sku = f"{str(r.condition)[:3].upper()}-E{int(r.esi)}-{str(r.arrival)[:3].upper()}"
        rows.append(
            {
                "sku": sku,
                "slug": str(r.condition).lower(),
                "hypoxia": float(r.spo2) < 92,
                "wait_label": f"{r.wait_min:.0f} min",
                "admit": bool(r.admit),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    n = st.number_input("Integer vs float (beds)", min_value=1, max_value=20, value=5)
    st.dataframe(
        pd.DataFrame(
            [
                {"expr": "n / 2", "value": str(n / 2), "type": type(n / 2).__name__},
                {"expr": "n // 2", "value": str(n // 2), "type": type(n // 2).__name__},
                {"expr": "n % 2", "value": str(n % 2), "type": type(n % 2).__name__},
            ]
        ),
        hide_index=True,
    )

with tabs[3]:
    def bucket(esi: int, spo2: float, arrival: str) -> str:
        if spo2 < 92 or esi == 1:
            return "resusc"
        if esi == 2 or arrival == "ambulance":
            return "urgent"
        if esi >= 4:
            return "fast_track"
        return "standard"

    classified = CASES.copy()
    classified["bucket"] = [
        bucket(int(e), float(s), str(a))
        for e, s, a in zip(classified["esi"], classified["spo2"], classified["arrival"])
    ]
    st.dataframe(classified[["encounter_id", "condition", "esi", "spo2", "arrival", "bucket"]], hide_index=True, width="stretch")
    st.dataframe(classified["bucket"].value_counts().rename_axis("bucket").reset_index(name="n"), hide_index=True)
    st.code(
        """def bucket(esi, spo2, arrival):
    if spo2 < 92 or esi == 1:
        return "resusc"
    elif esi == 2 or arrival == "ambulance":
        return "urgent"
    elif esi >= 4:
        return "fast_track"
    else:
        return "standard"
""",
        language="python",
    )

with tabs[4]:
    def acuity_weight(esi: int, *bumps: float) -> float:
        base = {1: 1.0, 2: 0.8, 3: 0.45, 4: 0.2, 5: 0.1}[esi]
        return min(1.0, base + sum(bumps))

    def wait_quote(site_base: float, esi: int, **mods: float) -> float:
        return round(max(1.0, site_base * (0.5 if esi <= 2 else 1.0) + sum(mods.values())), 1)

    esi = st.select_slider("ESI", options=[1, 2, 3, 4, 5], value=3)
    hypoxia = st.checkbox("Hypoxia bump +0.25", value=False)
    w = acuity_weight(int(esi), 0.25 if hypoxia else 0.0)
    q = wait_quote(18.0, int(esi), flu=4.0 if st.checkbox("Flu wave +4 min") else 0.0)
    st.dataframe(pd.DataFrame([{"esi": esi, "acuity_weight": w, "quoted_wait": q}]), hide_index=True)
    pitfall("`def f(flags=[])` mutates across calls. Agents will leak state. Use `None` or `default_factory`.")

with tabs[5]:
    los_map = {c: lo for c, lo in zip(atlas["condition"], atlas["typical_los_h"])}
    high_admit = [c for c, a in zip(atlas["condition"], atlas["admit_base"]) if a >= 0.25]
    st.dataframe(
        pd.DataFrame([{"rank": i, "condition": c, "typical_los_h": los_map[c]} for i, c in enumerate(atlas["condition"], 1)]),
        hide_index=True,
        width="stretch",
    )
    st.markdown(f"High baseline-admit conditions: **{', '.join(high_admit)}**")
    a, *mid, z = atlas["condition"].tolist()
    st.dataframe(pd.DataFrame({"unpack": ["first", "middle*", "last"], "value": [a, ", ".join(mid), z]}), hide_index=True)

with tabs[6]:
    @dataclass
    class Encounter:
        condition: str
        esi: int
        spo2: float
        symptoms: list[str] = field(default_factory=list)

        @property
        def hypoxia(self) -> bool:
            return self.spo2 < 92

        def acuity(self) -> str:
            if self.hypoxia or self.esi <= 2:
                return "high"
            return "standard"

    class Board:
        def __init__(self) -> None:
            self._rows: list[Encounter] = []

        def add(self, e: Encounter) -> None:
            self._rows.append(e)

        def __len__(self) -> int:
            return len(self._rows)

        def high(self) -> int:
            return sum(1 for e in self._rows if e.acuity() == "high")

    board = Board()
    built = []
    for r in CASES.itertuples(index=False):
        e = Encounter(str(r.condition), int(r.esi), float(r.spo2), [])
        board.add(e)
        built.append({**asdict(e), "hypoxia": e.hypoxia, "acuity": e.acuity()})
    st.dataframe(pd.DataFrame(built), hide_index=True, width="stretch")
    st.metric("Board high-acuity", f"{board.high()} / {len(board)}")
    ok("Dataclass = chart row. A class with methods = tiny agent state. sklearn estimators are classes with fit/predict.")

with tabs[7]:
    def parse_esi(raw: str) -> int:
        allowed = {"1", "2", "3", "4", "5"}
        if raw not in allowed:
            raise ValueError(f"ESI {raw!r} not in 1–5")
        return int(raw)

    trial = st.selectbox("Try parsing ESI", ["3", "2", "0", "stat"], key="esi_parse")
    try:
        parsed = parse_esi(trial)
        err = pd.DataFrame([{"input": trial, "status": "ok", "parsed": str(parsed), "error": ""}])
    except ValueError as exc:
        err = pd.DataFrame([{"input": trial, "status": "ValueError", "parsed": "", "error": str(exc)}])
    st.dataframe(err, hide_index=True, width="stretch")
    payload = {
        "clinic": "Northshore",
        "n": int(len(CASES)),
        "encounters": [{"id": int(r.encounter_id), "dx": str(r.condition), "admit": bool(r.admit)} for r in CASES.itertuples(index=False)],
    }
    blob = json.dumps(payload, indent=2)
    st.code(blob[:450] + "\n…", language="json")
    st.dataframe(pd.DataFrame(json.loads(blob)["encounters"]), hide_index=True, width="stretch")
    buf = io.StringIO()
    CASES.to_csv(buf, index=False)
    st.download_button("Download these 12 encounters as CSV", buf.getvalue(), "opening_day_encounters.csv", "text/csv")
    st.caption(f"pathlib: `{Path('data') / 'encounters.csv'}` — this lab generates data in memory.")

with tabs[8]:
    ts = CASES.copy()
    ts["hour"] = ts["ts"].dt.hour
    ts["dow"] = ts["ts"].dt.day_name()
    ts["plus_20m"] = ts["ts"] + pd.to_timedelta(20, unit="m")
    st.dataframe(ts[["encounter_id", "ts", "hour", "dow", "plus_20m", "esi"]], hide_index=True, width="stretch")
    now = datetime.fromisoformat("2026-05-04T18:10:00")
    st.dataframe(
        pd.DataFrame(
            [
                {"clock": "door", "value": now.isoformat(sep=" ")},
                {"clock": "esi2_sla", "value": (now + timedelta(minutes=20)).isoformat(sep=" ")},
            ]
        ),
        hide_index=True,
    )

with tabs[9]:
    c = Counter(CASES["condition"].astype(str).tolist())
    st.dataframe(pd.DataFrame(c.most_common(), columns=["condition", "n"]), hide_index=True)
    by_site: dict[str, list[float]] = defaultdict(list)
    for r in CASES.itertuples(index=False):
        by_site[str(r.site)].append(float(r.wait_min))
    st.dataframe(
        pd.DataFrame({"site": list(by_site), "n": [len(v) for v in by_site.values()], "avg_wait": [round(sum(v) / len(v), 1) for v in by_site.values()]}),
        hide_index=True,
        width="stretch",
    )
    Line = namedtuple("Line", "condition esi spo2")
    st.dataframe(pd.DataFrame([Line(str(r.condition), int(r.esi), float(r.spo2)) for r in CASES.itertuples(index=False)]), hide_index=True)
    q = deque(CASES["encounter_id"].tolist(), maxlen=5)
    st.caption("deque(maxlen=5) — last five encounter ids")
    st.dataframe(pd.DataFrame({"last_5": list(q)}), hide_index=True)

with tabs[10]:
    def stream_wait(frame: pd.DataFrame, site: str) -> Iterable[float]:
        for r in frame.itertuples(index=False):
            if r.site == site:
                yield float(r.wait_min)

    site = st.selectbox("Stream waits for site (full ledger)", clinic.sites, key="gen_site")
    first = list(itertools.islice(stream_wait(enc, site), 8))
    st.dataframe(pd.DataFrame({"yield_i": range(1, len(first) + 1), "wait_min": first}), hide_index=True)
    st.metric(f"{site} mean wait", f"{float(enc.loc[enc.site.eq(site), 'wait_min'].mean()):.1f} min")
    pn = st.slider("Conditions in grid", 2, 8, 3)
    grid = list(itertools.product(clinic.condition_names[:pn], [2, 3, 4], ["walk_in", "ambulance"]))
    st.dataframe(pd.DataFrame(grid, columns=["condition", "esi", "arrival"]), hide_index=True, height=240, width="stretch")
    st.caption(f"{len(grid)} cells — same explosion OneHotEncoder will create. Agents that enumerate differentials hit this too.")

with tabs[11]:
    Arrival = Literal["walk_in", "ambulance", "referral"]

    def timed(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            wrapper.calls = getattr(wrapper, "calls", 0) + 1  # type: ignore[attr-defined]
            return out
        return wrapper

    @timed
    def top_dx(frame: pd.DataFrame) -> str:
        return str(Counter(frame["condition"].astype(str)).most_common(1)[0][0])

    @contextmanager
    def isolation(on: bool):
        yield {"isolation": on, "ppe": "droplet" if on else "standard"}

    with isolation(True) as iso:
        iso_state = iso
    _ = top_dx(CASES)
    _ = top_dx(CASES)
    st.dataframe(
        pd.DataFrame(
            [
                {"idea": "Literal[arrival]", "example": "walk_in | ambulance | referral"},
                {"idea": "@timed calls", "example": str(getattr(top_dx, "calls", 0))},
                {"idea": "contextmanager isolation", "example": str(iso_state)},
                {"idea": "top_dx()", "example": top_dx(CASES)},
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    ok("An agent is a loop + tools + state. You already have all three in this tab.")

st.markdown("---")
ask(
    "q1_python",
    "You need {patient_id → last encounter timestamp} from 10M HL7 messages, once. First tool?",
    [
        "A pandas merge on the full frame, then unique",
        "A dict or defaultdict updated in one pass (or a generator + dict)",
        "A nested list of lists, then .index()",
    ],
    "A dict or defaultdict updated in one pass (or a generator + dict)",
    "One pass, O(n) for keys you saw. Pandas is right when you also need windows and joins.",
)
