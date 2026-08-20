"""Catalog of every topic on the site — concept, clinic example, interview Q/A.

Same shape as the .NET interview map: Topic / Concept / Example / Where, plus a drill answer.
"""

from __future__ import annotations

CATALOG_COLS = ["area", "topic", "what", "example", "where"]


def T(
    area: str,
    topic: str,
    what: str,
    where: str,
    example: str,
    interview_q: str,
    interview_a: str,
    choices: list[str],
    answer: str,
) -> dict[str, object]:
    return {
        "area": area,
        "topic": topic,
        "what": what,
        "where": where,
        "example": example.strip(),
        "interview_q": interview_q,
        "interview_a": interview_a,
        "choices": choices,
        "answer": answer,
    }


TOPICS: list[dict[str, object]] = [
    T(
        "Python",
        "Scalar types",
        "int, float, str, bool, None — missing lab is None, not 0",
        "Python → Types",
        """age, spo2, condition, admit = 67, 88.0, "Pneumonia", True
pending_troponin = None
print(type(age), type(spo2), type(condition), type(admit), type(pending_troponin))
# type(67) → int          type(88.0) → float
# type("Pneumonia") → str type(True) → bool
# type(None) → NoneType   — assay not run, not 0.0 (measured normal)""",
        "Why is a missing troponin None (or NaN) instead of 0?",
        "0 is a measured value (troponin not elevated). None/NaN means the assay was not run. Imputing 0 trains the model that 'no test' looks like 'normal', which is leakage of the ordering decision.",
        [
            "0 is a measured normal; None/NaN means the assay was not run",
            "Python cannot store 0.0 in a float column",
            "sklearn rejects zeros as features",
        ],
        "0 is a measured normal; None/NaN means the assay was not run",
    ),
    T(
        "Python",
        "Collections",
        "list, tuple, set, dict; set algebra ∩ ∪ −; tuple as an immutable key",
        "Python → Collections",
        """resp = {"flu", "pneumonia", "covid_like"}
tonight = {"flu", "uti"}
print(tonight & resp)              # {'flu'}
print(tonight - resp)              # {'uti'}
cache_key = ("Campus", 2)          # tuple key, not a list
# wait_by[("Campus", 2)]  → 11.0
# wait_by[["Campus", 2]]  → TypeError (list is not hashable)""",
        "When do you use a tuple vs a list as a dict key?",
        "Dict keys must be hashable. Tuples of immutables are; lists are not. Composite keys like (site, esi) belong in a tuple. Use a list when you need to append.",
        [
            "Tuples are hashable (if items are); lists are not — use tuple keys like (site, esi)",
            "Tuples are faster at append",
            "Lists cannot hold strings",
        ],
        "Tuples are hashable (if items are); lists are not — use tuple keys like (site, esi)",
    ),
    T(
        "Python",
        "Strings & numbers",
        "f-strings, slicing, slugs, // vs /, %, round",
        "Python → Strings & numbers",
        """sku = f\"{condition[:3].upper()}-E{esi}-{arrival[:3].upper()}\"
beds_left = n_beds // 2   # int; n_beds / 2 is float
# "Pneumonia", esi=2, "ambulance" → "PNE-E2-AMB"
# 5 / 2 → 2.5 (float)    5 // 2 → 2 (int beds)    5 % 2 → 1""",
        "What is the difference between / and // ?",
        "`/` is true division (always float in Python 3). `//` is floor division (int if both ints). Mixing them in a bed-count formula is a classic off-by-type bug.",
        [
            "/ is true division (float); // is floor division",
            "// is only for floats",
            "/ truncates toward zero like C",
        ],
        "/ is true division (float); // is floor division",
    ),
    T(
        "Python",
        "Control flow",
        "if / elif / else, for, while, in",
        "Python → Control flow",
        """def bucket(esi, spo2, arrival):
    if spo2 < 92 or esi == 1:
        return "resusc"
    if esi == 2 or arrival == "ambulance":
        return "urgent"
    if esi >= 4:
        return "fast_track"
    return "standard"

# bucket(3, 88, "walk_in") → first if True → "resusc"
# bucket(2, 98, "walk_in") → esi == 2      → "urgent"
# bucket(5, 99, "walk_in") → esi >= 4      → "fast_track"
# bucket(3, 98, "walk_in") → else          → "standard" """,
        "Why elif instead of stacked independent ifs for mutually exclusive buckets?",
        "Stacked `if`s can assign twice. `if/elif/else` encodes a single decision tree — what you want for triage buckets and for a deterministic agent planner.",
        [
            "elif makes the branches mutually exclusive so a case cannot match twice",
            "elif is faster than if",
            "else is illegal after elif",
        ],
        "elif makes the branches mutually exclusive so a case cannot match twice",
    ),
    T(
        "Python",
        "Functions",
        "defaults, *args, **kwargs, return, lambda sort key",
        "Python → Functions",
        """def acuity_weight(esi, *bumps):
    base = {1: 1.0, 2: 0.8, 3: 0.45, 4: 0.2, 5: 0.1}[esi]
    return min(1.0, base + sum(bumps))

# acuity_weight(3)              → bumps=()           → 0.45
# acuity_weight(3, 0.25)        → bumps=(0.25,)      → 0.70
# acuity_weight(3, 0.25, 0.10)  → bumps=(0.25, 0.10) → 0.80
# acuity_weight(1, 0.25)        → 1.0+0.25           → min → 1.0
# BAD: acuity_weight(3, [0.25])  # one list, not two numbers
# BAD: def f(flags=[])           # shared mutable default""",
        "Why is `def f(xs=[])` a bug?",
        "Default args are evaluated once, at function definition. The same list is reused. An agent that `flags.append(...)` will leak state into the next encounter. Use `xs=None` and `xs = []` inside, or `field(default_factory=list)` on a dataclass.",
        [
            "The default list is created once and shared across calls",
            "Empty lists are illegal defaults",
            "Python copies the list on every call",
        ],
        "The default list is created once and shared across calls",
    ),
    T(
        "Python",
        "Comprehensions",
        "list/dict comps, zip, enumerate, unpacking first, *rest, last",
        "Python → Comprehensions",
        """high = [c for c, a in zip(conditions, admit_base) if a >= 0.25]
los_map = {c: h for c, h in zip(conditions, typical_los_h)}
first, *mid, last = conditions
# Pneumonia 0.42 kept; URI 0.03 dropped → high includes "Pneumonia"
# los_map["UTI"] → 3.5
# 8 names: first=URI, last=Cardiac_rule_out, mid = the six in between""",
        "When is a list comprehension the wrong tool?",
        "When the body has side effects, nested logic, or you need early exit. Prefer a generator if you only iterate once. Interviewers want: comps for maps/filters, loops for procedures.",
        [
            "When the body has side effects, early exit, or you only need a one-pass generator",
            "Comprehensions cannot filter",
            "Never use them in production",
        ],
        "When the body has side effects, early exit, or you only need a one-pass generator",
    ),
    T(
        "Python",
        "OOP",
        "class, @dataclass, field(default_factory), @property, methods, __len__",
        "Python → OOP",
        """@dataclass
class Encounter:
    condition: str
    esi: int
    spo2: float
    symptoms: list[str] = field(default_factory=list)

    @property
    def hypoxia(self) -> bool:
        return self.spo2 < 92

# Encounter("Pneumonia", 2, 88.0).hypoxia → True  (88 < 92)
# each Encounter() gets its own symptoms list — not symptoms=[]""",
        "Why default_factory=list instead of symptoms=[] on a dataclass?",
        "Same shared-mutable trap as function defaults. `default_factory` runs per instance so each Encounter gets its own list.",
        [
            "default_factory runs per instance; [] would be shared across instances",
            "list is not a valid type",
            "dataclass forbids default values",
        ],
        "default_factory runs per instance; [] would be shared across instances",
    ),
    T(
        "Python",
        "Exceptions",
        "raise, try / except ValueError, explicit fallbacks",
        "Python → Exceptions & files",
        """def parse_esi(raw: str) -> int:
    if raw not in {"1", "2", "3", "4", "5"}:
        raise ValueError(f"ESI {raw!r} not in 1–5")
    return int(raw)

# parse_esi("3") → 3
# parse_esi("0") → ValueError: ESI '0' not in 1–5
# except ValueError: esi = 3   # explicit fallback, you chose it""",
        "Do you catch Exception around a model call?",
        "No — catch the specific errors you can recover from (ValueError, KeyError). Bare `except Exception` hides leakage bugs and schema drift. Agents should fail the tool and record the observation, not swallow it.",
        [
            "Catch specific recoverable errors; do not swallow Exception around fit/predict",
            "Always catch Exception and return None",
            "Never raise in library code",
        ],
        "Catch specific recoverable errors; do not swallow Exception around fit/predict",
    ),
    T(
        "Python",
        "Files & JSON",
        "json.dumps / loads, pathlib.Path, CSV export",
        "Python → Exceptions & files",
        """payload = {"clinic": "Northshore", "n": len(rows), "encounters": rows}
blob = json.dumps(payload)
path = Path("data") / "encounters.parquet"
# dumps: dict → str     loads: str → dict  (True stays bool)
# Path("data") / "encounters.csv"  — no "data" + "/" + name""",
        "pathlib vs os.path — what do you say?",
        "`pathlib.Path` is the current default: objects, `/` join, `.read_text()`. `os.path` still appears in older code. For ML artifacts prefer parquet over CSV (types, NaNs, size).",
        [
            "Path is the current API (objects, / join); parquet beats CSV for typed ML tables",
            "os.path is required on Linux",
            "JSON is a replacement for parquet",
        ],
        "Path is the current API (objects, / join); parquet beats CSV for typed ML tables",
    ),
    T(
        "Python",
        "Datetime",
        "datetime, timedelta, weekday, ISO; Pandas .dt",
        "Python → Datetime",
        """cutoff = start + timedelta(days=int(len(dates) * 0.75))
enc["hour"] = pd.to_datetime(enc["ts"]).dt.hour
enc["dow"] = pd.to_datetime(enc["date"]).dt.dayofweek
# 2026-05-04 19:12 → .dt.hour == 19  (rush feature)
# Monday 00:00 + timedelta(days=1) → Tuesday 00:00
# naive (no tz) mixed with UTC silently shifts the train cutoff""",
        "Naive vs aware datetimes — why does it matter for a time split?",
        "Naive timestamps have no tz. Mixing UTC logs with local clinic hours silently shifts your cutoff. Store UTC, convert for display, split on a timezone-consistent clock.",
        [
            "Naive has no tz; mixing UTC and local shifts the train/test cutoff",
            "datetime cannot store dates",
            "Pandas .dt always converts to UTC",
        ],
        "Naive has no tz; mixing UTC and local shifts the train/test cutoff",
    ),
    T(
        "Python",
        "collections",
        "Counter, defaultdict, namedtuple, deque(maxlen)",
        "Python → Stdlib",
        """from collections import Counter, defaultdict, deque
by_site = Counter(enc["site"])
flags = defaultdict(list)
trace = deque(maxlen=8)   # agent last-N thoughts
# Counter(['Downtown']*5 + ['Campus']*2) → {'Downtown': 5, 'Campus': 2}
# flags["spo2"].append("low") works even if "spo2" was missing
# maxlen=3, four pushes → oldest dropped; last three remain""",
        "When is defaultdict better than dict.setdefault?",
        "When you are grouping into lists/counters in a loop. `defaultdict(list)` makes the insert path obvious. Prefer Counter for tallies. Interview plus: deque(maxlen=N) is O(1) rolling memory for an agent trace.",
        [
            "Grouping into lists/counters in a loop; deque(maxlen=N) for a bounded agent trace",
            "defaultdict is thread-safe and dict is not",
            "Counter cannot count strings",
        ],
        "Grouping into lists/counters in a loop; deque(maxlen=N) for a bounded agent trace",
    ),
    T(
        "Python",
        "Generators",
        "yield, one-pass streams, itertools.islice",
        "Python → Generators & itertools",
        """def stream_encounters(rows):
    for r in rows:
        yield r   # one row in memory

from itertools import islice
preview = list(islice(stream_encounters(rows), 12))
# next(gen) → row 1; next again → row 2; rest not in memory yet
# list(gen) pulls everything; list(gen) again is empty (one-shot)
# islice(gen, 12) is head — 12 rows, then stop""",
        "Generator vs list — memory and reuse?",
        "A generator is lazy and one-shot. `list(gen)` materializes. If you iterate twice, a generator is empty the second time. Use a list when you need len/reuse; a generator when the stream is large and single-pass (logs, NDJSON).",
        [
            "Generators are lazy and one-shot; lists are reusable and have len",
            "Generators are always faster",
            "yield returns a list",
        ],
        "Generators are lazy and one-shot; lists are reusable and have len",
    ),
    T(
        "Python",
        "itertools",
        "product (OneHot-style crosses), groupby",
        "Python → Generators & itertools",
        """from itertools import product
cross = list(product(["Downtown", "Campus"], [1, 2]))
# → [('Downtown', 1), ('Downtown', 2), ('Campus', 1), ('Campus', 2)]
# 2 sites × 2 ESI = 4 pairs  (dummy grid / test matrix)
# 5 sites × 5 ESI = 25 rows — same idea as a site × ESI one-hot""",
        "How is itertools.product related to one-hot / feature crosses?",
        "`product(A, B)` is the Cartesian product — the same idea as a site×ESI dummy grid. sklearn OneHotEncoder builds columns; product builds the explicit pairs for a lookup table or a test matrix.",
        [
            "product is the Cartesian product — the same idea as a site × ESI dummy grid",
            "product multiplies DataFrame columns in place",
            "product is only for integers",
        ],
        "product is the Cartesian product — the same idea as a site × ESI dummy grid",
    ),
    T(
        "Python",
        "Typing",
        "Literal, Callable, annotations (not runtime checks)",
        "Python → Typing & decorators",
        """from typing import Callable, Literal
ToolFn = Callable[..., dict]
Band = Literal["low", "mid", "high"]
# band = "high"  → checker OK
# band = "HOT"   → checker error; runtime still a str (hints are not locks)
# plan(...) -> tuple[str, str, dict] documents the planner; Python will not enforce it""",
        "Do type hints run at runtime?",
        "No (unless you add pydantic/beartype). They document contracts and feed mypy/pyright. `Literal` is an interview favorite: the admit band is not an arbitrary str.",
        [
            "No — they are for checkers/docs unless you add a runtime validator",
            "Yes, Python 3.12 enforces them",
            "Only in dataclasses",
        ],
        "No — they are for checkers/docs unless you add a runtime validator",
    ),
    T(
        "Python",
        "Decorators",
        "@wraps, call counters — same idea as tool wrappers",
        "Python → Typing & decorators",
        """from functools import wraps
def counted(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        wrapper.n += 1
        return fn(*args, **kwargs)
    wrapper.n = 0
    return wrapper

@counted
def get_chart(eid):
    return eid
# get_chart(1); get_chart(2) → get_chart.n == 2
# @wraps keeps __name__ == "get_chart", not "wrapper" """,
        "Why @wraps on a decorator?",
        "Without it, `__name__`/`__doc__` become `wrapper`, which breaks debugging and some registries. Agent tool wrappers are decorators: log, time, catch. Same pattern as .NET action filters.",
        [
            "Preserves __name__/__doc__ of the wrapped function",
            "Makes the decorator async",
            "Required for class methods only",
        ],
        "Preserves __name__/__doc__ of the wrapped function",
    ),
    T(
        "Python",
        "Context managers",
        "with, @contextmanager (isolation / resources)",
        "Python → Typing & decorators",
        """from contextlib import contextmanager
@contextmanager
def seed_scope(rng_seed):
    # enter: isolate RNG / temp dir
    yield
    # exit: restore
# with open(path) as f: closes the file even if json.loads raises
# enter → body → exit     (happy path)
# enter → body raises → exit still runs, then the error propagates""",
        "What does `with` guarantee that try/finally also does?",
        "`__exit__` runs even on exception — close files, release locks, restore RNG. Prefer `with` for readability. Interview: `with Path.open()` vs leaking handles in a long agent loop.",
        [
            "__exit__ runs even if the body raises — resources get released",
            "with disables exceptions",
            "with is only for files",
        ],
        "__exit__ runs even if the body raises — resources get released",
    ),
    T(
        "NumPy",
        "ndarray shape / dtype",
        "lab_means (8×14), protocol (8×14), inventory (5×14), cost (14,)",
        "NumPy labs",
        """lab_means.shape   # (8, 14)  conditions × biomarkers
protocol.shape    # (8, 14)
assay_cost.shape  # (14,)
# len(lab_means) → 8  (axis 0 only, not 8×14=112)
# (14,) vs (14, 1) — both 14 numbers; @ and broadcast care
# lab_means.dtype → float64; object/None kills vectorization""",
        "Why does shape matter more than len() for ML arrays?",
        "`len` is only axis 0. Broadcasting and matmul fail on the trailing dims. Always print `.shape` and `.dtype` before a model. A (14,) vs (14,1) mismatch is the #1 NumPy interview trap.",
        [
            "len is only axis 0; matmul/broadcast need the full shape (14,) vs (14,1)",
            "dtype is ignored by sklearn",
            "shape is only for images",
        ],
        "len is only axis 0; matmul/broadcast need the full shape (14,) vs (14,1)",
    ),
    T(
        "NumPy",
        "matmul / @",
        "panel_cost = protocol @ assay_cost; pull = counts @ protocol",
        "NumPy labs · Capstone reagents",
        """panel_cost = protocol @ assay_cost     # (8,14) @ (14,) -> (8,) dollars per condition
pull = condition_counts @ protocol     # (n_conditions,) @ (8,14) -> (14,) assays tonight
# Pneumonia row · assay_cost → one dollar figure for that panel
# protocol * assay_cost broadcasts cost onto each assay but does NOT sum
# (8,14) @ (14,) is legal — @ is not "square matrices only" """,
        "When do you use @ vs * ?",
        "`*` is elementwise (Hadamard). `@` is matrix multiply / dot. `protocol * assay_cost` broadcasts a cost onto each indicated assay; `protocol @ assay_cost` sums those into one dollar figure per condition. Say both; know which one you meant.",
        [
            "* is elementwise; @ is matrix multiply — panel dollars are a dot product",
            "@ is only for square matrices",
            "* and @ are aliases in NumPy 2",
        ],
        "* is elementwise; @ is matrix multiply — panel dollars are a dot product",
    ),
    T(
        "NumPy",
        "Broadcasting",
        "patient vector vs ref_low/ref_high; z vs band",
        "NumPy labs",
        """z = (x - ref_low) / (ref_high - ref_low)   # x (14,) or (n,14) vs (14,)
# (14,) vs (14,) → elementwise flags for one Pneumonia vector
# (n, 14) vs (14,) → same band stretched across every patient  (broadcast)
# (8, 14) vs (8,)  → FAIL unless you reshape ref to (8, 1)""",
        "State the broadcasting rule in one sentence.",
        "Compare shapes right to left: dims equal, or one of them is 1, or one is missing. `(n,14)` vs `(14,)` works; `(8,14)` vs `(8,)` does not unless you reshape to `(8,1)`.",
        [
            "Align from the right: equal, 1, or missing — (n,14) vs (14,) works; (8,14) vs (8,) needs (8,1)",
            "NumPy always tiles the shorter array on the left",
            "Broadcasting copies the array in memory every time",
        ],
        "Align from the right: equal, 1, or missing — (n,14) vs (14,) works; (8,14) vs (8,) needs (8,1)",
    ),
    T(
        "NumPy",
        "Boolean masks",
        "abnormal = (x < low) | (x > high); reagent reorder mask",
        "NumPy labs · flag_labs",
        """abnormal = (x < ref_low) | (x > ref_high)
names = np.array(biomarker_names)[abnormal]
# spo2=88 vs low=92 → True on that slot; wbc in band → False
# names might be ["spo2", "lactate", "crp"] — flag_labs is this mask
# no Python for-loop over the 14 assays""",
        "Why masks instead of a Python for-loop over biomarkers?",
        "Vectorized compares are SIMD, stay in C, and compose (`low | high & ~missing`). The agent's `flag_labs` tool is this mask behind a name. Loops are for control flow, not 14 (or 14k) assays.",
        [
            "Vectorized compares stay in C and compose; flag_labs is a mask behind a tool name",
            "Boolean masks cannot index arrays",
            "Python loops are required for |",
        ],
        "Vectorized compares stay in C and compose; flag_labs is a mask behind a tool name",
    ),
    T(
        "NumPy",
        "Fancy / integer index",
        "bincount + matmul vs Python loop over conditions",
        "NumPy labs",
        """idx = condition_id_per_encounter          # (n,)
counts = np.bincount(idx, minlength=8)    # (8,)
pull = counts @ protocol                  # (14,)
# 100 encounters, 40 coded as Pneumonia (id=2) → counts[2] == 40
# counts @ protocol → 14 assay pulls for tonight, no Python loop""",
        "bincount vs a Python counter loop?",
        "`np.bincount` is the histogram of integer codes in one pass. Combined with `@ protocol` you get tonight's assay pull without iterating encounters. Same idea as a sparse one-hot then matmul.",
        [
            "bincount histograms integer codes in one pass; then counts @ protocol is the assay pull",
            "bincount only works on floats",
            "You must loop to multiply protocol",
        ],
        "bincount histograms integer codes in one pass; then counts @ protocol is the assay pull",
    ),
    T(
        "NumPy",
        "View vs copy",
        "slicing can mutate the atlas; .copy() before write",
        "NumPy labs",
        """view = lab_means[0]       # may share memory with atlas
safe = lab_means[0].copy()
view[:] = 0               # can wipe pneumonia's row in the atlas
# view.base is lab_means  → mutating view mutates the parent
# fancy index lab_means[[0, 3]] usually copies — write is safe""",
        "How do you know if a slice is a view?",
        "`.base is not None` usually means a view. Slices are often views; fancy integer/boolean index usually copies. Mutating a view mutates the parent — disastrous if that parent is `lab_means` used by the agent.",
        [
            "Slices are often views (.base is not None); fancy index usually copies — mutating a view mutates the atlas",
            "All NumPy indexing copies",
            ".copy() is never needed after a slice",
        ],
        "Slices are often views (.base is not None); fancy index usually copies — mutating a view mutates the atlas",
    ),
    T(
        "NumPy",
        "Axis reductions",
        "mean(axis=0) by assay; sum(axis=1) by site",
        "NumPy labs",
        """lab_means.mean(axis=0)   # (14,) mean per biomarker
inventory.sum(axis=1)    # (5,)  total vials per site
# (8, 14) axis=0 collapses 8 conditions → 14 numbers
# (8, 14) axis=1 collapses 14 assays    → 8 numbers""",
        "What does axis=0 mean?",
        "Collapse that axis. `axis=0` on (8,14) reduces conditions → one value per biomarker. `axis=1` reduces biomarkers → one value per condition. Draw the shape; don't memorize 'rows vs columns' without the array orientation.",
        [
            "Collapse that axis: axis=0 on (8,14) → (14,) per biomarker",
            "axis=0 always means columns in pandas sense",
            "axis is only for 1-D arrays",
        ],
        "Collapse that axis: axis=0 on (8,14) → (14,) per biomarker",
    ),
    T(
        "NumPy",
        "clip / nan",
        "clip labs to physiologic ranges; missing lactate/troponin as nan",
        "clinic/data.py · flag_labs",
        """x = np.clip(x, ref_low, ref_high * 1.5)
missing = np.isnan(x)
z = (x - ref_low) / np.clip(ref_high - ref_low, 1e-6, None)
# spo2=200 clipped to the physiologic cap; spo2=np.nan stays missing
# None in a float array → object dtype; use np.nan then np.isnan""",
        "NaN vs None in a numeric array?",
        "NumPy float arrays cannot hold None without becoming object dtype (kills vectorization). Use np.nan, then `np.isnan`. sklearn SimpleImputer talks NaN, not None.",
        [
            "Float arrays use np.nan; None forces object dtype and kills vectorization",
            "NaN and None compare equal",
            "clip removes NaNs",
        ],
        "Float arrays use np.nan; None forces object dtype and kills vectorization",
    ),
    T(
        "NumPy",
        "L2 nearest prototype",
        "argmin ||lab_means - x|| → nearest_condition",
        "Agent",
        """dist = np.linalg.norm(lab_means - x, axis=1)  # (8,)
guess = condition_names[int(np.argmin(dist))]
# x closer to Pneumonia mean than UTI mean → argmin → "Pneumonia"
# that name fetches a protocol — it is not a diagnosis""",
        "Is nearest prototype a diagnosis?",
        "No. It is a retriever: closest mean vector in lab space. The agent uses it to fetch a protocol, not to name a disease. Say 'prototype / k-NN in 14-D', not 'the model diagnosed pneumonia'.",
        [
            "No — it retrieves the closest lab-mean protocol, it is not a diagnosis",
            "Yes — argmin is a Bayes classifier",
            "L2 is illegal on labs",
        ],
        "No — it retrieves the closest lab-mean protocol, it is not a diagnosis",
    ),
    T(
        "Pandas",
        "DataFrame / Series",
        "encounters ledger, dtypes, memory, head",
        "Pandas chart",
        """enc.head()
enc.dtypes
enc.memory_usage(deep=True).sum()
# enc["spo2"] is a Series (1-D, labeled)
# enc[["spo2", "hr"]] is a DataFrame (aligned columns, shared index)""",
        "Series vs DataFrame in one line?",
        "Series is one labeled 1-D array (a column). DataFrame is a dict of Series sharing an index. Models usually want a DataFrame in, a Series (target) beside it.",
        [
            "Series is 1-D labeled; DataFrame is aligned columns sharing an index",
            "Series cannot hold floats",
            "DataFrame is a NumPy matrix with no index",
        ],
        "Series is 1-D labeled; DataFrame is aligned columns sharing an index",
    ),
    T(
        "Pandas",
        "loc / boolean filter",
        "site, arrival, ESI masks; hypoxia spo2 < 92",
        "Pandas chart",
        """enc.loc[enc.site.eq("Campus") & enc.esi.isin([1, 2]), "spo2"]
hypoxia = enc[enc.spo2 < 92]
# Campus AND esi in {1,2} → those rows' spo2 column
# spo2=88 → in hypoxia; spo2=98 → out
# loc = labels/masks; iloc = positions; [] is ambiguous""",
        "loc vs iloc vs [] ?",
        "`loc` is labels (including boolean masks). `iloc` is integer positions. `[]` is convenient but ambiguous (sometimes columns, sometimes rows). In production filters, prefer `enc.loc[mask, cols]` so you cannot set-on-copy by accident.",
        [
            "loc is labels/masks; iloc is positions; [] is ambiguous — prefer loc[mask, cols]",
            "iloc is for column names",
            "loc cannot take a boolean mask",
        ],
        "loc is labels/masks; iloc is positions; [] is ambiguous — prefer loc[mask, cols]",
    ),
    T(
        "Pandas",
        "groupby agg",
        "n, admit_rate, avg_wait by site × condition",
        "Pandas chart",
        """enc.groupby(["site", "condition"], observed=True).agg(
    n=("encounter_id", "count"),
    admit_rate=("admit", "mean"),
)
# Downtown × Pneumonia → n=40, admit_rate=0.42  (one row per pair)
# observed=True: skip site×condition pairs that never happened""",
        "What does observed=True do on a categorical groupby?",
        "Categoricals have a declared set of levels. `observed=False` emits empty combinations (every site×condition even if n=0), which inflates the grid and can leak unused dummy levels. `observed=True` keeps combinations that actually appear.",
        [
            "observed=True keeps combinations that actually appear; False emits the full categorical grid",
            "observed skips NaN targets",
            "observed=True is required for merge",
        ],
        "observed=True keeps combinations that actually appear; False emits the full categorical grid",
    ),
    T(
        "Pandas",
        "transform",
        "wait / site_mean aligned to original index",
        "Pandas chart",
        """enc["site_wait"] = enc.groupby("site", observed=True)["wait_min"].transform("mean")
enc["vs_site"] = enc["wait_min"] / enc["site_wait"]
# Downtown mean wait 22 min; this row 11 min → vs_site = 0.5
# agg would be 5 rows (one per site); transform keeps 10k rows""",
        "agg vs transform?",
        "`agg` reduces (one row per group). `transform` broadcasts the group statistic back to the original index — what you need to compare each encounter to its site mean. Interviewers love this pair.",
        [
            "agg reduces to one row per group; transform aligns the statistic back to every row",
            "transform is just a faster agg",
            "agg cannot compute mean",
        ],
        "agg reduces to one row per group; transform aligns the statistic back to every row",
    ),
    T(
        "Pandas",
        "merge",
        "encounters ⨝ patients on patient_id (left join)",
        "Pandas chart",
        """chart = enc.merge(patients[["patient_id", "copd", "cad"]], on="patient_id", how="left")
# 6 encounters, 5 patient IDs, one first-visit P99 not in patients:
# how="left"  → 6 rows (P99 keeps a chart; copd is NaN)
# how="inner" → 5 rows (P99 dropped — not random; biases admit)
# how="right" → keeps P05 with no visit tonight (usually wrong for a board)
# always: n_before, n_after, n_unmatched = chart["copd"].isna().sum()""",
        "left vs inner join on patient_id — what can go wrong?",
        "`inner` silently drops encounters whose MRN is missing in the patient table (first visit, identity mismatch). That drop is not random — it biases admit rate. Always `how='left'` then count nulls. Same as a SQL left join.",
        [
            "inner drops unmatched MRNs (often first visits) and biases admit rate — left then count nulls",
            "left join duplicates columns and must be avoided",
            "merge cannot join on strings",
        ],
        "inner drops unmatched MRNs (often first visits) and biases admit rate — left then count nulls",
    ),
    T(
        "Pandas",
        "Missing data",
        "dropna vs site median vs site+ESI median (lactate/troponin)",
        "Pandas chart",
        """# dropna shrinks the chart and is not MCAR
enc["lactate"] = enc["lactate"].fillna(
    enc.groupby(["site", "esi"], observed=True)["lactate"].transform("median")
)
# missing lactate because it was not ordered — related to suspicion
# dropna removes those rows; site+ESI median keeps them with a number""",
        "Is dropna safe before fit?",
        "Only if missingness is MCAR and rare. Lactate is missing because it was not ordered — related to suspicion of sepsis — so dropping it removes the very patients you care about. Impute inside a Pipeline on train only, or model missingness as a feature.",
        [
            "Usually no — lab missingness is informative; dropna removes the patients you care about",
            "Yes, sklearn requires complete cases",
            "Median impute on the full frame is always unbiased",
        ],
        "Usually no — lab missingness is informative; dropna removes the patients you care about",
    ),
    T(
        "Pandas",
        "Time series",
        "shift(7), rolling(7), .dt hour/dow",
        "Pandas chart",
        """g = daily.groupby("site", observed=True)["encounters"]
daily["enc_lag7"] = g.shift(7)
daily["enc_roll7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
# Tuesday this week ← Tuesday last week (lag-7), not a random row
# shift(1) before rolling: today's census is not inside the feature for today""",
        "Why shift(1) before rolling(7)?",
        "Without the shift, today's census is inside the rolling mean you will use to predict today — leakage. Lag-7 is last week's same weekday; roll-7 is the past week only.",
        [
            "Otherwise today's target leaks into the feature used to predict today",
            "shift is required for groupby to work",
            "rolling cannot run without shift for math reasons",
        ],
        "Otherwise today's target leaks into the feature used to predict today",
    ),
    T(
        "Pandas",
        "crosstab / pivot",
        "site × condition counts",
        "Pandas chart",
        """pd.crosstab(enc["site"], enc["condition"])
enc.pivot_table(index="site", columns="hour", values="admit", aggfunc="mean")
# Downtown × Pneumonia → 40 visits (count)
# Downtown × hour 19 admit mean → 0.18 (rate; pair with n)""",
        "crosstab vs groupby.size().unstack()?",
        "Same idea: two-way counts. `crosstab` is the readable helper; `pivot_table` is for arbitrary aggfunc (mean admit). Watch `dropna`/`fill_value` so zeros stay zeros, not NaN that later get imputed.",
        [
            "Same two-way table; crosstab for counts, pivot_table when you need mean admit etc.",
            "crosstab only works on integers",
            "pivot_table cannot use mean",
        ],
        "Same two-way table; crosstab for counts, pivot_table when you need mean admit etc.",
    ),
    T(
        "Pandas",
        "Categorical",
        "site, arrival, season, ordered ESI; observed=True",
        "clinic/data.py",
        """enc["esi"] = pd.Categorical(enc["esi"], categories=[1, 2, 3, 4, 5], ordered=True)
enc["site"] = enc["site"].astype("category")
# esi 2 < esi 3 is meaningful because ordered=True
# do not one-hot patient_id the same way — that is identity, not a level""",
        "Why category dtype in an ML table?",
        "Smaller memory, defined levels for groupby, and a contract for OneHotEncoder. Ordered categories (ESI) make comparisons meaningful. Do not one-hot an unordered high-cardinality ID.",
        [
            "Memory, declared levels, ordered ESI; still do not one-hot high-cardinality IDs",
            "category makes sklearn skip the column",
            "category is required for floats",
        ],
        "Memory, declared levels, ordered ESI; still do not one-hot high-cardinality IDs",
    ),
    T(
        "EDA",
        "Line / bar / box",
        "census over time; hour × arrival; wait vs ESI × season",
        "EDA & charts",
        """daily.groupby("date")["encounters"].sum().plot()  # line
enc.groupby("hour")["arrival"].value_counts().unstack().plot.bar()
# line: flu week is a hill, not a flat mean — that becomes flu_wave
# bar: 19:00 stacked ambulance vs walk_in — that becomes rush / arrival
# box wait vs ESI: ESI 2 should sit under the 20 min SLA line""",
        "What is EDA for if you already know the features?",
        "To find the features you did not know: flu waves, hour×arrival mix, ESI wait SLA. If you cannot see it on a chart, do not put it in the model. Interview: EDA is hypothesis generation, not pretty pictures.",
        [
            "To discover features you can see (flu wave, hour mix) before you encode them",
            "EDA replaces a test set",
            "Charts are only for executives",
        ],
        "To discover features you can see (flu wave, hour mix) before you encode them",
    ),
    T(
        "EDA",
        "Heatmap",
        "admit rate site × hour",
        "EDA & charts",
        """rate = pd.crosstab(enc["site"], enc["hour"], values=enc["admit"], aggfunc="mean")
n = pd.crosstab(enc["site"], enc["hour"])  # pair every rate with its count
# Harbor hour 3, rate=1.00, n=2 → noise, not a policy
# Downtown hour 19, rate=0.18, n=80 → real mix""",
        "What can a heatmap hide?",
        "Sample size. A 100% admit cell with n=2 looks like a pattern. Always pair a rate heatmap with a count heatmap, or annotate n.",
        [
            "Small-n cells look like real rates — pair rates with counts",
            "Heatmaps cannot show rates",
            "Heatmaps leak labels by design",
        ],
        "Small-n cells look like real rates — pair rates with counts",
    ),
    T(
        "EDA",
        "Scatter",
        "volume vs admit rate; wait vs SpO2",
        "EDA & charts",
        """# color by site so Simpson's paradox is visible
px.scatter(daily, x="encounters", y="admit_rate", color="site")
# pooled: busier days look safer
# colored: Harbor slope can reverse Downtown — do not fit one line on the pile""",
        "Why color by site on a scatter?",
        "Pooled scatter can reverse the within-site slope (Simpson). Color/facet is the cheapest causal-sanity check before you fit.",
        [
            "Pooled scatter can reverse the within-site slope (Simpson) — color/facet first",
            "Color is only aesthetic",
            "Scatter cannot take a third variable",
        ],
        "Pooled scatter can reverse the within-site slope (Simpson) — color/facet first",
    ),
    T(
        "EDA",
        "Correlation",
        "Pearson on site-day numerics; Simpson warning",
        "EDA & charts",
        """feat[["encounters", "enc_lag7", "flu_wave", "admit_rate"]].corr()
# lag-7 vs encounters is signal; same-day admit_rate vs encounters is leakage""",
        "Does high correlation mean we should drop a feature?",
        "Not automatically. Lag-7 will correlate with today — that is the signal. Drop when two features are the same information and unstable (admit_rate vs encounters on the same day is leakage, not just collinearity).",
        [
            "No — lag-7 correlation is signal; drop same-day outcome cousins (leakage), not every correlated pair",
            "Yes, corr > 0.3 must be dropped",
            "Pearson implies causation",
        ],
        "No — lag-7 correlation is signal; drop same-day outcome cousins (leakage), not every correlated pair",
    ),
    T(
        "EDA",
        "Plotly + seaborn",
        "interactive Plotly; seaborn scatter sample",
        "EDA & charts",
        """px.line(daily, x="date", y="encounters", color="site")
sns.scatterplot(data=enc.sample(800), x="spo2", y="wait_min", hue="esi")
# 10k points hide the cloud — sample 800 or hexbin for EDA
# neither plot belongs inside the training Pipeline""",
        "Why sample for seaborn?",
        "Overplotting: 10k points hide the density. Sample or hexbin for exploration; Plotly for interactive site filters. Neither belongs inside the training Pipeline.",
        [
            "Overplotting hides density — sample/hexbin for EDA; charts are not pipeline steps",
            "seaborn cannot plot more than 100 rows",
            "Plotly fits models",
        ],
        "Overplotting hides density — sample/hexbin for EDA; charts are not pipeline steps",
    ),
    T(
        "Features",
        "Calendar features",
        "dow, is_weekend, flu_wave, is_heat",
        "Feature engineering",
        """feat["is_weekend"] = feat["dow"].isin([5, 6]).astype(int)
feat["is_heat"] = (feat["season"] == "heat").astype(int)
# Saturday → is_weekend=1  (known at dawn)
# heat flag: only legal if you already know season at 9am, not from today's census""",
        "Are season flags known at prediction time?",
        "Calendar dow/weekend: yes. Flu/heat: only if you have a forecast or a lag, not the same-day realized wave if that is measured from today's census. Known-at-inference is the leakage test.",
        [
            "dow/weekend yes; flu/heat only if known at inference — not a same-day realized wave from today's census",
            "All calendar features leak",
            "Season must be one-hot with patient_id",
        ],
        "dow/weekend yes; flu/heat only if known at inference — not a same-day realized wave from today's census",
    ),
    T(
        "Features",
        "Lag / roll",
        "enc_lag7, enc_roll7 = shift(1).rolling(7) per site",
        "Feature engineering",
        """g = d.groupby("site", observed=True)["encounters"]
d["enc_lag7"] = g.shift(7)
d["enc_roll7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
# Harbor Monday lags Harbor last Monday — not Downtown Sunday
# shift(1) before rolling: today's y is not inside the feature for today""",
        "Why groupby site before shift?",
        "Otherwise Harbor's Monday lags Downtown's Sunday. Panel time series: always lag within entity. Same rule as a store, a ticker, a server.",
        [
            "Lag within entity — otherwise Harbor Monday lags Downtown Sunday",
            "groupby is only for aggregation",
            "shift is global so groupby does nothing",
        ],
        "Lag within entity — otherwise Harbor Monday lags Downtown Sunday",
    ),
    T(
        "Features",
        "Time split vs random",
        "same model, two splits; random inflates R²",
        "Feature engineering",
        """# BAD
train_test_split(X, y, test_size=0.25, random_state=0)
# GOOD
cutoff = df["date"].quantile(0.75)
train, test = df[df.date <= cutoff], df[df.date > cutoff]
# random: Tuesday in train, Monday in test — same flu week, inflated R²
# time: train through cutoff, test is next week — what the floor will actually face""",
        "Why does a random split inflate R² on census?",
        "Adjacent days share season, flu, and lag features. Randomly putting Tuesday in train and Monday in test lets the model peek at the same local weather. A time split asks: would this have worked last week for next week?",
        [
            "Adjacent days share season/flu/lags — random split peeks at the same local regime",
            "Random split is unbiased for all data",
            "Time split is only for NLP",
        ],
        "Adjacent days share season/flu/lags — random split peeks at the same local regime",
    ),
    T(
        "Features",
        "Leakage",
        "same-day admit_rate as a feature; encoder fit on full frame",
        "Feature engineering",
        """# illegal at 9am
X = np.hstack([calendar_lags, df[["admit_rate"]]])
# illegal encoder
enc.fit(full["site"])   # saw test frequencies
# 1) same-day admit_rate to predict census
# 2) OneHot fit on the full frame
# 3) gold_label_condition inside the agent planner""",
        "Give three leakage examples from this clinic.",
        "1) same-day admit_rate to predict census. 2) OneHot/target-encoder fit on the full frame. 3) Using gold_label_condition inside the agent planner. Bonus: lactate imputed with the test-set median.",
        [
            "Same-day admit_rate; encoder fit on full frame; gold labels in the agent planner",
            "Using lag-7; using ESI; using site",
            "StandardScaler on train only",
        ],
        "Same-day admit_rate; encoder fit on full frame; gold labels in the agent planner",
    ),
    T(
        "Features",
        "Target encoding leak",
        "mean admit by condition on the full table",
        "Feature engineering",
        """# LEAK — mean includes this row's label and the test set
enc["leaky"] = enc.groupby("condition")["admit"].transform("mean")
# Pneumonia row sees its own admit in the mean → target leakage
# legal: fold means on train only, then map to test with smoothing""",
        "How do you target-encode without leaking?",
        "Fit means on train folds only (out-of-fold / CV encoding), apply to test with train maps, and add smoothing for rare conditions. Never `groupby(target).transform` on the full frame.",
        [
            "Out-of-fold means on train only, then map to test — never transform the full frame",
            "Always encode on the full dataset for stability",
            "Target encoding is illegal in sklearn",
        ],
        "Out-of-fold means on train only, then map to test — never transform the full frame",
    ),
    T(
        "Features",
        "High-cardinality IDs",
        "do not one-hot patient_id / MRN",
        "Feature engineering",
        """# BAD
OneHotEncoder().fit_transform(enc[["patient_id"]])  # ~1.6k columns of identity
# new MRN at test time → unknown dummy, model never saw that person
# use past-only aggregates (prior visits), never the raw ID""",
        "Why not one-hot patient_id to predict admit?",
        "It memorizes who was admitted before, will not generalize to new MRNs, and usually leaks historical admit frequency. Use aggregates (prior visits) computed from the past only — never the raw ID as a dummy.",
        [
            "It overfits identity, fails on new MRNs, and leaks historical admit frequency",
            "sklearn cannot one-hot strings",
            "patient_id is numeric so it is a safe feature",
        ],
        "It overfits identity, fails on new MRNs, and leaks historical admit frequency",
    ),
    T(
        "Regression",
        "Problem",
        "predict site-day encounter census",
        "Regression",
        """y = feat["encounters"]
X = feat[["dow", "is_weekend", "flu_wave", "is_heat", "enc_lag7", "enc_roll7", "site", "season"]]
# one row = one site-day (Downtown, 2026-05-04) → y=42 encounters
# not one row per visit — mixing grains is a design bug""",
        "What is the unit of prediction?",
        "Site-day, not encounter. Mixing levels (predicting census with encounter-level ESI) is a design bug. State the grain before the model.",
        [
            "Site-day census — state the grain before the estimator",
            "One row per patient lifetime",
            "One row per biomarker",
        ],
        "Site-day census — state the grain before the estimator",
    ),
    T(
        "Regression",
        "Naive baseline",
        "lag-7; beat it or you have no model",
        "Regression",
        """naive = test["enc_lag7"]
mae_naive = (test["encounters"] - naive).abs().mean()
# lag-7 MAE 12, Ridge MAE 11.8 → you have not earned the complexity
# quote MAE vs this number, not R² alone""",
        "Why a naive baseline before Ridge?",
        "If lag-7 already has MAE 12 and Ridge has 11.8, you have not earned the complexity. Interviewers want the baseline first; it is also what the floor already uses ('same as last week').",
        [
            "If you cannot beat last week, you have no model — and the floor already uses that heuristic",
            "Baselines are only for classification",
            "Ridge cannot be compared to lag-7",
        ],
        "If you cannot beat last week, you have no model — and the floor already uses that heuristic",
    ),
    T(
        "Regression",
        "Linear / Ridge / RF",
        "estimators behind the same Pipeline",
        "Regression",
        """from sklearn.pipeline import Pipeline
pipe = Pipeline([("prep", daily_transformer()), ("model", Ridge(alpha=1.0))])
pipe.fit(X_train, y_train)
# OLS on one-hot site + collinear lags → wild coefficients
# Ridge α=1 shrinks them; same Pipeline shape as score_admit later""",
        "Why Ridge over OLS here?",
        "One-hot site + season + collinear lags. Ridge shrinks coefficients (L2) so the model does not explode on correlated calendar dummies. RF would capture interactions but needs a time-aware split still.",
        [
            "Collinear lags + one-hots — L2 shrinkage beats unstable OLS coefficients",
            "Ridge is a classifier",
            "OLS cannot run inside a Pipeline",
        ],
        "Collinear lags + one-hots — L2 shrinkage beats unstable OLS coefficients",
    ),
    T(
        "Regression",
        "MAE RMSE R²",
        "metrics on the time-split test window",
        "Regression",
        """from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(y_test, y_hat)
r2 = r2_score(y_test, y_hat)
# MAE=8.2 encounters/day vs lag-7 MAE=12 → quote that to ops
# R²=0.71 on a random split is not the same number on a time split""",
        "MAE vs RMSE vs R² — which do you quote to ops?",
        "MAE in encounters/day — same unit as the board. RMSE punishes large misses (Friday spikes). R² is relative to predicting the mean; easy to inflate with a random split. Quote MAE vs lag-7.",
        [
            "MAE in encounters/day vs lag-7; RMSE for big misses; R² is easy to inflate",
            "R² is the only metric that matters",
            "MAE is biased so never use it",
        ],
        "MAE in encounters/day vs lag-7; RMSE for big misses; R² is easy to inflate",
    ),
    T(
        "Regression",
        "Residuals",
        "residual vs predicted, colored by site",
        "Regression",
        """resid = y_test - y_hat
# plot resid vs y_hat, color=site — fan shape = heteroscedasticity
# yhat=20, resid≈±4; yhat=60, resid≈±15 → error grows with volume
# one site all positive resid → missing a site shift, not a new estimator""",
        "What would a fan-shaped residual plot tell you?",
        "Variance grows with predicted volume (heteroscedasticity). Linear MAE still useful; consider log target or a count model. Also check one site is systematically high — missing a site feature or a shift in mix.",
        [
            "Heteroscedasticity — error grows with volume; check site bias too",
            "The model is correctly specified",
            "R² must be negative",
        ],
        "Heteroscedasticity — error grows with volume; check site bias too",
    ),
    T(
        "Regression",
        "Coefficients / importance",
        "linear coefs after scaling; RF impurity importance (debug only)",
        "Regression",
        """# linear: coefs are in scaled units — do not compare to raw age
# RF impurity importance is biased to high-cardinality features""",
        "Can you rank features from RandomForest impurity importance?",
        "Cautiously. It is biased toward high-cardinality and correlated features, and it is not causal. Prefer permutation importance on the time-split test set, or just read Ridge coefficients after scaling.",
        [
            "Impurity importance is biased; prefer permutation on the time-split test set or scaled Ridge coefs",
            "Impurity importance is the causal effect",
            "Coefficients are comparable without scaling",
        ],
        "Impurity importance is biased; prefer permutation on the time-split test set or scaled Ridge coefs",
    ),
    T(
        "Classification",
        "Problem",
        "P(admit) at encounter level",
        "Classification",
        """y = df["admit_int"]
X = df[num_cols + cat_cols]
pipe.predict_proba(X_test)[:, 1]   # P(admit)
# predict() uses 0.5 → not a bed budget
# predict_proba → 0.41; ops cuts at 0.35 or 'top 6 scores'""",
        "Why predict_proba instead of predict?",
        "The threshold is a bed budget, not a math constant. `predict` uses 0.5. Ops wants a ranked list: score, then cut to the number of beds. The agent consumes `p_admit` as a tool output.",
        [
            "Threshold is a bed budget — rank P(admit), do not ship 0.5",
            "predict_proba is slower so avoid it",
            "predict already uses the optimal threshold",
        ],
        "Threshold is a bed budget — rank P(admit), do not ship 0.5",
    ),
    T(
        "Classification",
        "Class imbalance",
        "majority discharge; dummy accuracy is a trap",
        "Classification",
        """majority_acc = 1 - y_test.mean()   # always-discharge accuracy
# admit rate 12% → dummy accuracy 88%
# a model with 88% accuracy has learned nothing — quote admit recall/PR-AUC""",
        "Admit rate is 12%. A model with 88% accuracy — good?",
        "No. Always discharging matches 88%. Quote recall/precision on the admit class, PR-AUC, and a confusion matrix at an ops threshold. Accuracy is the dummy.",
        [
            "No — always discharge already gets ~88%; quote admit-class recall/precision/PR-AUC",
            "Yes, 88% exceeds 80%",
            "Accuracy is invariant to base rate",
        ],
        "No — always discharge already gets ~88%; quote admit-class recall/precision/PR-AUC",
    ),
    T(
        "Classification",
        "LogReg / RandomForest",
        "class_weight none vs balanced",
        "Classification",
        """LogisticRegression(max_iter=500, class_weight="balanced")
RandomForestClassifier(class_weight="balanced", min_samples_leaf=4)
# admit ~12% → balanced up-weights those rows in the loss
# scores shift; you still pick a threshold after""",
        "What does class_weight='balanced' do?",
        "Reweights the loss so the minority (admit) counts more — roughly inverse frequency. It changes the score scale; you still choose a threshold. It is not magic for bad features.",
        [
            "Reweights the loss toward the minority class; you still pick a threshold",
            "It oversamples the CSV on disk",
            "It equalizes precision and recall",
        ],
        "Reweights the loss toward the minority class; you still pick a threshold",
    ),
    T(
        "Classification",
        "Threshold",
        "cut on P(admit); bed budget, not 0.5",
        "Classification · Agent",
        """thr = 0.35   # slider on the floor
pred = (proba >= thr).astype(int)
# p_admit=0.41, thr=0.35 → predicted admit (recall up, more false alarms)
# p_admit=0.41, thr=0.50 → predicted discharge (sklearn default, not a policy)
# beds tonight = 6 → you can only take the top 6 scores, whatever 0.5 says""",
        "Who owns the threshold?",
        "The operator (beds, risk tolerance), not sklearn. Lower threshold: more predicted admits, higher recall, more false alarms. The agent can expose `p_admit` and let a human pick the cut.",
        [
            "Ops / bed budget — lower cut raises recall and false alarms; sklearn's 0.5 is not a policy",
            "Always 0.5 by Bayes",
            "RandomForest has no threshold",
        ],
        "Ops / bed budget — lower cut raises recall and false alarms; sklearn's 0.5 is not a policy",
    ),
    T(
        "Classification",
        "Confusion matrix",
        "discharge vs admit, labeled table of precision/recall/F1",
        "Classification",
        """# rows true, cols predicted (sklearn default)
# FP: flagged admit, went home — wasted bed
# FN: sent home, should have admitted — the expensive miss""",
        "For admit, which error is worse — FP or FN?",
        "Product decision. Clinically FN (missed admit) is usually costlier; operationally too many FPs board the hallway. State both costs; do not let the interviewer trap you into 'F1 is enough'.",
        [
            "FN is usually the clinical miss; FP wastes a bed — name both costs, do not hide behind F1",
            "FP and FN are always equal",
            "Confusion matrices are only for multiclass",
        ],
        "FN is usually the clinical miss; FP wastes a bed — name both costs, do not hide behind F1",
    ),
    T(
        "Classification",
        "ROC and PR",
        "AUC; precision–recall for the minority class",
        "Classification",
        """roc_auc_score(y_test, proba)
precision, recall, _ = precision_recall_curve(y_test, proba)
# 12% admit: ROC can look strong from true negatives
# PR-AUC asks: when you flag admit, how often were you right, across cuts""",
        "ROC-AUC vs PR-AUC under 12% admit rate?",
        "ROC-AUC can look strong while the admit ranking is weak (many TNs). PR-AUC focuses on the positive class. Prefer PR-AUC (and a curve at operating points) for imbalanced admit.",
        [
            "ROC can look fine from TNs; PR-AUC is the minority-class ranking metric",
            "They are identical",
            "PR-AUC ignores precision",
        ],
        "ROC can look fine from TNs; PR-AUC is the minority-class ranking metric",
    ),
    T(
        "Classification",
        "Imputation in Pipeline",
        "SimpleImputer median on train only (lactate, troponin)",
        "clinic/ml.py",
        """num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
ColumnTransformer([("num", num, num_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
# fit on train → median lactate stored from train only
# agent score_admit replays that median — not a test-set peek""",
        "Why impute inside the Pipeline, not before split?",
        "A global median sees the test distribution. Pipeline.fit on train stores train medians and applies them at predict time — including in the agent's `score_admit`. That is the whole point of Pipeline.",
        [
            "Train medians only — a global median sees test; the agent then reuses the same fitted object",
            "SimpleImputer cannot sit in a Pipeline",
            "You must impute before train_test_split",
        ],
        "Train medians only — a global median sees test; the agent then reuses the same fitted object",
    ),
    T(
        "Clusters & pipelines",
        "K-means",
        "k groups on scaled patient features; k is a choice",
        "Clusters",
        """labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X_scaled)
# k=4 on scaled [age, log_visits, admits, recency, comorbid, avg_esi]
# output: segment id per patient — not an ICD code""",
        "Does k-means produce diagnoses?",
        "No. It partitions scaled space into k Voronoi cells. You name the clusters in English (older high-util, …). k is a product choice, not a statistical discovery of 'true diseases'.",
        [
            "No — k Voronoi cells on scaled features; you name phenotypes; k is a choice",
            "k-means is a supervised diagnosis model",
            "k is selected by accuracy",
        ],
        "No — k Voronoi cells on scaled features; you name phenotypes; k is a choice",
    ),
    T(
        "Clusters & pipelines",
        "Scaling",
        "StandardScaler so age does not drown comorbid",
        "Clusters",
        """Xz = StandardScaler().fit_transform(X)
# age ~ 70 would dominate comorbid in {0,1,2,3,4} under Euclidean k-means
print(X.std(axis=0), Xz.std(axis=0))""",
        "Why scale before k-means?",
        "k-means is Euclidean. Age in years (~70) dominates comorbid count (~0–4). StandardScaler puts features on comparable axes. Same reason Ridge coefficients need scaling to be comparable.",
        [
            "Euclidean distance — age in years would dominate 0–4 comorbid counts",
            "Scaling is only for neural nets",
            "k-means is invariant to scale",
        ],
        "Euclidean distance — age in years would dominate 0–4 comorbid counts",
    ),
    T(
        "Clusters & pipelines",
        "PCA",
        "2D view of segments (visualization, not a diagnosis)",
        "Clusters",
        """pca = PCA(n_components=2, random_state=0)
xy = pca.fit_transform(Xz)
# scatter pc1 vs pc2 colored by segment — visualization only""",
        "Is PCA a clustering algorithm?",
        "No. It rotates to variance axes for visualization (or as a preprocessor). Clusters in a 2-D PCA plot can look cleaner than they are in 6-D. Do not cluster on 2 PCs just because the scatter looks nice unless you meant to.",
        [
            "No — variance rotation for a view; pretty 2-D plots can lie about 6-D separation",
            "PCA assigns cluster labels",
            "PCA is supervised",
        ],
        "No — variance rotation for a view; pretty 2-D plots can lie about 6-D separation",
    ),
    T(
        "Clusters & pipelines",
        "Silhouette",
        "tightness hint; name the cluster in English",
        "Clusters",
        """sil = silhouette_score(Xz, labels)
# 0.25 vs 0.45 is a hint. If care management cannot name the cluster, k is wrong.""",
        "High silhouette — ship it?",
        "No. Silhouette is a geometric hint. If care management cannot name the cluster, k is wrong. Also: silhouette prefers convex blobs; it will not save a bad feature set.",
        [
            "No — if nobody can name the cluster in English, k is wrong regardless of silhouette",
            "Silhouette > 0.5 is a diagnosis",
            "Silhouette replaces a test set",
        ],
        "No — if nobody can name the cluster in English, k is wrong regardless of silhouette",
    ),
    T(
        "Clusters & pipelines",
        "Phenotypes",
        "unsupervised segments (older high-util, …)",
        "Clusters · Capstone",
        """profile = (
    pat.groupby("segment")
    .agg(n=("patient_id", "count"), age=("age", "mean"),
         visits=("visits", "mean"), comorbid=("comorbid", "mean"))
)
# name each row in English: "older high-util", "young low-touch", …""",
        "How do you validate a phenotype?",
        "Stability (bootstrap / time split: does the same story reappear?), actionability (a panel or outreach), and no use of the admit label in the features if you will later predict admit — that would leak the target into the segment.",
        [
            "Stability over time, a name/action, and do not bake the target into the features",
            "Accuracy against ICD codes",
            "Pick the k with the highest admit rate",
        ],
        "Stability over time, a name/action, and do not bake the target into the features",
    ),
    T(
        "Clusters & pipelines",
        "Pipeline",
        "ColumnTransformer + estimator; one object to dump",
        "Clusters",
        """pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])),
    ("model", Ridge()),
])
joblib.dump(pipe, "admit.joblib")
# ticket → prep (train stats) → model → yhat
# five pickle files that disagree is the failure this prevents""",
        "What problem does Pipeline solve in one sentence?",
        "Preprocess and model fit on the same rows and replay the same transforms at predict time — so the agent's `score_admit` cannot apply a different scaler than training.",
        [
            "Fit preprocess+model on train rows and replay the same transforms at predict (including in the agent)",
            "Pipeline parallelizes across GPUs",
            "Pipeline replaces ColumnTransformer",
        ],
        "Fit preprocess+model on train rows and replay the same transforms at predict (including in the agent)",
    ),
    T(
        "Clusters & pipelines",
        "TimeSeriesSplit + GridSearch",
        "walk-forward α grid; no shuffled Fridays",
        "Clusters",
        """GridSearchCV(pipe, {"model__alpha": [0.1, 1, 10]},
             cv=TimeSeriesSplit(n_splits=4), scoring="neg_mean_absolute_error")
# fold1: train days 1–10, val 11–14
# fold2: train days 1–14, val 15–18
# shuffled KFold: future Friday in train, past Monday in test — cheat""",
        "Why not KFold shuffle for census?",
        "Shuffled KFold puts future Fridays in train and past Mondays in test. TimeSeriesSplit walks forward: train on the past, validate on the next fold. Same philosophy as the holdout time split.",
        [
            "Shuffled KFold leaks future days; TimeSeriesSplit trains past → validates next",
            "GridSearchCV cannot take TimeSeriesSplit",
            "KFold is required for Ridge",
        ],
        "Shuffled KFold leaks future days; TimeSeriesSplit trains past → validates next",
    ),
    T(
        "Agent",
        "Tool",
        "named callable: get_chart, flag_labs, score_admit, nearest_condition, retrieve_protocol",
        "Agent workflow",
        """tools = {
    "get_chart": Tool("get_chart", "Load ESI, arrival, comorbidities.", fn),
    "flag_labs": Tool("flag_labs", "NumPy masks vs reference ranges.", fn),
    "score_admit": Tool("score_admit", "sklearn Pipeline P(admit).", fn),
}
# get_chart(20000) → {esi: 2, arrival: "ambulance", ...}  (gold stripped)
# flag_labs(20000) → {n_abnormal: 3, names: ["spo2", ...]}
# score_admit(20000) → {p_admit: 0.41, band: "mid"}""",
        "What is a tool vs the model?",
        "A tool is a named function with a schema (args in, JSON-ish out). The sklearn Pipeline is one tool (`score_admit`). The product is the loop that calls tools, not the AUC.",
        [
            "A named callable with a schema; the Pipeline is one tool, not the product",
            "A tool is a GPU kernel",
            "Tools replace metrics",
        ],
        "A named callable with a schema; the Pipeline is one tool, not the product",
    ),
    T(
        "Agent",
        "Planner",
        "deterministic plan(state); swap for LLM {tool, args} later",
        "Agent workflow",
        """def plan(state):
    if state.chart is None:
        return "Need the chart", "get_chart", {"encounter_id": state.encounter_id}
    if state.lab_flags is None:
        return "Flag labs", "flag_labs", {"encounter_id": state.encounter_id}
    return "Stop", "stop", {}
# empty state → first return is always get_chart
# after chart is filled → next is flag_labs
# after protocol is in state → stop (not another tool forever)""",
        "Why start with a deterministic planner instead of an LLM?",
        "You can unit-test the workflow, log a trace, and teach the architecture without a vendor. Later swap `plan()` for a model that emits `{tool, args}`. Same as .NET: interface first, implementation later.",
        [
            "Testable workflow and a clear swap point — LLM emits {tool, args} later, same interface",
            "LLMs cannot call functions",
            "Deterministic planners cannot stop",
        ],
        "Testable workflow and a clear swap point — LLM emits {tool, args} later, same interface",
    ),
    T(
        "Agent",
        "State / memory",
        "chart, lab_flags, admit_score, protocol, red_flags",
        "clinic/agent.py",
        """@dataclass
class AgentState:
    encounter_id: int
    chart: dict | None = None
    lab_flags: dict | None = None
    admit_score: dict | None = None
    done: bool = False
# state.chart is None → we have not called get_chart yet
# stuffing the same fields into a prompt hides whether labs already ran""",
        "Why a state object instead of stuffing everything into the prompt?",
        "Structured state is typed, inspectable, and cheap. Prompts dump strings and hide whether `flag_labs` already ran. This is the same idea as a saga / workflow document in backend systems.",
        [
            "Typed, inspectable, no 'did we already flag labs?' ambiguity — same idea as a workflow document",
            "State objects cannot be serialized",
            "Prompts are always smaller",
        ],
        "Typed, inspectable, no 'did we already flag labs?' ambiguity — same idea as a workflow document",
    ),
    T(
        "Agent",
        "Trace",
        "thought → tool → observation audit log",
        "Agent workflow",
        """TraceRow(step, thought, tool, observation)
# thought: "Chart in hand. Flag labs."
# tool: flag_labs
# observation: n_abnormal=3, spo2=88""",
        "Why keep a trace?",
        "Audit and debugging. When the suggestion is wrong you see which tool lied. Same as distributed tracing. Also your eval: compare guess vs gold without putting gold in the thought.",
        [
            "Audit: which tool lied; eval without stuffing gold into the thought",
            "Traces train the sklearn model",
            "You only need the final sentence",
        ],
        "Audit: which tool lied; eval without stuffing gold into the thought",
    ),
    T(
        "Agent",
        "Stop + recommend",
        "suggested plan for a clinician; not a diagnosis",
        "Agent workflow",
        """state.recommendation = (
    f"Suggested (not a diagnosis): nearest prototype {cond}. "
    f"Admit-risk band {band}. Clinician must confirm."
)
# after protocol is in state, plan() returns stop — not another tool
# max_steps=8 is a fuse; infinite tools are a production incident""",
        "When should the loop stop?",
        "When the state has enough to write a suggestion, or max_steps. Infinite tool loops are a production incident. Stop is a first-class tool, not an afterthought.",
        [
            "When state is sufficient or max_steps — stop is a first-class action, not an infinite loop",
            "After exactly one tool",
            "Never stop if recall < 1",
        ],
        "When state is sufficient or max_steps — stop is a first-class action, not an infinite loop",
    ),
    T(
        "Agent",
        "Gold labels",
        "hidden from planner; eval only (no RAG leakage)",
        "Agent workflow",
        """chart.pop("gold_label_condition")  # tools never see this
chart.pop("gold_admit")
# eval later: guess == gold_label_condition
# tools see: esi, spo2, labs
# tools never see: true condition, true admit""",
        "Why hide gold_label_condition from get_chart?",
        "Otherwise the planner (or an LLM) copies the answer — the agent 'cheat' analog of target leakage. Gold is for your scorecard after the run, like a held-out test label.",
        [
            "Otherwise the planner copies the answer — same bug as target leakage",
            "Gold labels improve recall so they belong in every tool",
            "Streamlit cannot hide columns",
        ],
        "Otherwise the planner copies the answer — same bug as target leakage",
    ),
    T(
        "Agent",
        "Model as a tool",
        "sklearn Pipeline is score_admit, not the product",
        "Agent · Capstone",
        """def score_admit(encounter_id):
    proba = float(pipe.predict_proba(row[cols])[0, 1])
    return {"p_admit": round(proba, 3), "band": band(proba)}
# joblib.dump(pipe) once; this tool loads it
# same columns as training or you get train/serving skew""",
        "How would you ship this in production?",
        "joblib.dump the Pipeline, version it, call it behind score_admit with the same columns as training, log traces, never let the model write the chart. The HTTP analog is one POST /score plus a workflow engine — same as the pizza-store interview APIs wrapping services.",
        [
            "Versioned Pipeline behind score_admit, same columns, traced workflow — model is not the app",
            "Pickle the whole Streamlit session",
            "Retrain on every request with the full warehouse",
        ],
        "Versioned Pipeline behind score_admit, same columns, traced workflow — model is not the app",
    ),
    T(
        "Capstone",
        "Census board",
        "forecast vs actual vs lag-7; season what-if",
        "Capstone",
        """# last night vs model vs last week (same site)
board = daily[daily.date == daily.date.max()][["site", "encounters"]]
board["yhat"] = pipe.predict(X_tonight)
board["naive_lag7"] = feat.loc[feat.date == feat.date.max(), "enc_lag7"].values
# Harbor losing while Downtown wins → site shift, not a generic ML win""",
        "What does beating lag-7 on the board prove?",
        "That the model is adding something the charge nurse does not already know. If it loses on Harbor only, you have a site-shift problem, not a generic ML win.",
        [
            "The model adds information beyond last week; slice by site before you celebrate",
            "Lag-7 is the sklearn default estimator",
            "The board replaces a test set",
        ],
        "The model adds information beyond last week; slice by site before you celebrate",
    ),
    T(
        "Capstone",
        "Reagent pull",
        "counts @ protocol vs on-hand cover",
        "Capstone",
        """pull = counts @ protocol          # (14,)
cover = inventory[site] / pull    # hours/days of cover
# lots of UTI tonight → more ua_nitrite in pull
# cover < 1 → that assay runs out at this site
# counts (8) @ protocol (8,14) → pull (14)""",
        "Why is this a matmul and not a SQL groupby?",
        "It can be either; the interview answer is: condition mix is a vector, protocol is a matrix, pull is one multiply. SQL aggregates first, then you still multiply. Same linear algebra as a recipe @ cost in the pizza-store labs.",
        [
            "Condition mix × protocol matrix is one multiply — SQL still does the same math after the aggregate",
            "SQL cannot count conditions",
            "Matmul requires a square protocol",
        ],
        "Condition mix × protocol matrix is one multiply — SQL still does the same math after the aggregate",
    ),
    T(
        "Capstone",
        "Admit desk",
        "score a live ticket (ESI, SpO2, arrival, …)",
        "Capstone",
        """ticket = pd.DataFrame([{
    "age": 67, "esi_n": 2, "hour": 19, "spo2": 88, "hr": 112,
    "temp_c": 38.4, "sbp": 98, "wbc": 14.2, "lactate_f": 3.1,
    "troponin_f": np.nan, "rush": 1, "site": "Downtown",
    "arrival": "ambulance", "season": "flu_wave",
}])
p_admit = float(pipe.predict_proba(ticket)[0, 1])  # same columns as training
# missing rush or a new arrival code → train/serving skew
# handle_unknown="ignore" zeros that dummy — seatbelt, not a feature""",
        "What must the live ticket match?",
        "Training columns and dtypes: esi_n, rush, lactate_f, site categories. Train/serving skew (new arrival code, missing rush) is the usual production break. handle_unknown='ignore' is a seatbelt, not a feature.",
        [
            "The same columns/dtypes as training — new categories and missing engineered fields are train/serving skew",
            "Any JSON is fine if keys are pretty",
            "The agent will impute column names",
        ],
        "The same columns/dtypes as training — new categories and missing engineered fields are train/serving skew",
    ),
    T(
        "Capstone",
        "Run agent",
        "same loop on tonight’s board",
        "Capstone",
        """state, trace, eval_rows = run_agent(clinic, admit_pipe, encounter_id)
# trace: get_chart → flag_labs → score_admit → nearest_condition → retrieve_protocol → stop
# eval_rows: guess vs gold_label_condition (gold was never in the tools)""",
        "What is the capstone testing that Module 07 did not?",
        "Integration: census + reagents + admit scores + the tool loop on the same universe. Module 07 is a model. Capstone is the product workflow. Same as the .NET pizza app: topics map to live calls, not slides.",
        [
            "Integration of forecast, reagents, scoring, and the tool loop — the product, not just AUC",
            "A new loss function",
            "That Streamlit can plot",
        ],
        "Integration of forecast, reagents, scoring, and the tool loop — the product, not just AUC",
    ),
]


def catalog_frame():
    import pandas as pd

    return pd.DataFrame([{k: t[k] for k in CATALOG_COLS} for t in TOPICS])


def areas() -> list[str]:
    return list(dict.fromkeys(str(t["area"]) for t in TOPICS))


from clinic.walkthroughs import attach_walkthroughs

attach_walkthroughs(TOPICS)
