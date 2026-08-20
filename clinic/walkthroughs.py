"""Step-by-step explanations in the acuity_weight style: English, numbered calls, picture, what it is not."""

from __future__ import annotations

WALKS: dict[tuple[str, str], str] = {}


def _put(area: str, topic: str, text: str) -> None:
    WALKS[(area, topic)] = text.strip() + "\n"


def explain(area: str, topic: str) -> str:
    return WALKS.get((area, topic), "")


def attach_walkthroughs(topics: list[dict]) -> None:
    for t in topics:
        t["explain"] = explain(str(t["area"]), str(t["topic"]))


_put(
    "Python",
    "Functions",
    r"""
Think of it as a **score from 0 to 1 for “how urgent is this visit?”** ESI gives the starting score. Extra problems (low oxygen, etc.) add a little. It is never allowed to go above 1.

```python
def acuity_weight(esi, *bumps):
    base = {1: 1.0, 2: 0.8, 3: 0.45, 4: 0.2, 5: 0.1}[esi]
    return min(1.0, base + sum(bumps))
```

1. Look up a **starting number** from ESI.
2. **Add** every extra number you passed in.
3. If the total is bigger than 1, **use 1**.

ESI here: **1 = most urgent**, **5 = least**.

| ESI | Starting score (`base`) |
|---|---|
| 1 | 1.00 (already max) |
| 2 | 0.80 |
| 3 | 0.45 |
| 4 | 0.20 |
| 5 | 0.10 |

`*bumps` means: after `esi`, you may pass **zero, one, or many extra numbers**. Python stuffs those extras into a **tuple** named `bumps`.

**Example 1 — no extras**

`acuity_weight(3)` → `esi=3`, `bumps=()`, `sum(())=0` → `0.45 + 0` → **0.45**

**Example 2 — one extra (hypoxia +0.25)**

`acuity_weight(3, 0.25)` → `bumps=(0.25,)` → `0.45 + 0.25` → **0.70**

**Example 3 — two extras**

`acuity_weight(3, 0.25, 0.10)` → `bumps=(0.25, 0.10)` → `0.45 + 0.35` → **0.80**

**Example 4 — cap**

`acuity_weight(1, 0.25)` → `1.0 + 0.25 = 1.25` → `min(1.0, 1.25)` → **1.0**

```
acuity_weight(  3  ,  0.25  ,  0.10 )
               esi    bump1    bump2
                │       └────┬────┘
                ▼            ▼
           dict → 0.45    *bumps → (0.25, 0.10)
                └──── sum 0.80 → min(1, 0.80) → 0.80
```

**What it is not:** `acuity_weight(3, [0.25, 0.10])` passes **one list**. Unpack: `acuity_weight(3, *extras)`.

**Sibling:** `wait_quote(18.0, esi, flu=4.0)` uses `**mods` — extras by **name**, not position.
""",
)

_put(
    "Python",
    "Scalar types",
    r"""
Think of one chart row as **five different kinds of Python value**. The type is the rule for what you can do with it.

```python
age, spo2, condition, admit = 67, 88.0, "Pneumonia", True
pending_troponin = None
```

| Field | Example | Type | Meaning |
|---|---|---|---|
| age | `67` | `int` | whole number, a feature |
| spo2 | `88.0` | `float` | measurement with a decimal |
| condition | `"Pneumonia"` | `str` | text label |
| admit | `True` | `bool` | yes/no target |
| pending_troponin | `None` | `NoneType` | **not done**, not zero |

**Example 1** — `type(67)` is `int`. `67 + 1` works. `"67" + 1` does not.

**Example 2** — `bool(True)` is the admit target. `1` is also truthy, but it is not the same as a lab.

**Example 3 — the trap**

```python
troponin = None    # assay not run
troponin = 0.0     # assay ran, value was zero (normal)
```

If you write `0.0` for missing, the admit model learns “no test looks like normal.” That is leakage of the ordering decision.

```
one encounter
  age=67        → int
  spo2=88.0     → float
  admit=True    → bool
  troponin=?    → None  (missing)  not  0.0 (measured)
```

**What it is not:** “Python has no types.” It does; they show up at runtime (`type(x)`). sklearn later wants numbers, so `None` becomes `NaN` in a float column.
""",
)

_put(
    "Python",
    "Collections",
    r"""
Think of four boxes for many values:

| Box | Ordered? | Duplicate? | Typical clinic use |
|---|---|---|---|
| `list` | yes | yes | tonight’s condition names |
| `tuple` | yes, **frozen** | yes | dict key `(site, esi)` |
| `set` | no | **no** | respiratory conditions |
| `dict` | keys unique | — | condition → typical LOS |

```python
resp = {"flu", "pneumonia", "covid_like"}
tonight = {"flu", "uti"}
print(tonight & resp)          # {'flu'}
cache_key = ("Campus", 2)      # tuple key
```

**Example 1 — set algebra**

- `tonight & resp` → in both → `{'flu'}`
- `tonight - resp` → tonight but not respiratory → `{'uti'}`
- `resp - tonight` → respiratory not here tonight

**Example 2 — tuple as a key**

```python
wait_by = {("Campus", 2): 11.0, ("Downtown", 3): 24.0}
wait_by[("Campus", 2)]   # 11.0
# wait_by[["Campus", 2]]  # TypeError: list is not hashable
```

**Example 3 — dict lookup**

`los["Pneumonia"]` is typical hours. Missing key → `KeyError` (same family as bad ESI in the dict).

```
tonight {flu, uti}     resp {flu, pneumonia, …}
         └──── ∩ ────┘
              {flu}
```

**What it is not:** a `list` as a dict key. Freeze it to a `tuple` first.
""",
)

_put(
    "Python",
    "Strings & numbers",
    r"""
Think of two jobs: **build a label** (string) and **divide beds** (numbers).

```python
sku = f"{condition[:3].upper()}-E{esi}-{arrival[:3].upper()}"
beds_left = n_beds // 2
```

**Example 1 — f-string**

`condition="Pneumonia"`, `esi=2`, `arrival="ambulance"`

- `condition[:3]` → `"Pne"` (first three characters)
- `.upper()` → `"PNE"`
- result: **`PNE-E2-AMB`**

**Example 2 — `/` vs `//`**

| Expr | `n_beds=5` | Type |
|---|---|---|
| `5 / 2` | `2.5` | `float` |
| `5 // 2` | `2` | `int` (floor) |
| `5 % 2` | `1` | remainder |

You cannot have 2.5 beds as an integer count. Use `//` when you need a whole number.

**Example 3 — slice**

`"ambulance"[:3]` → `"amb"`. `[:3]` means “start at 0, stop before 3.”

```
"Pneumonia"
 0 1 2 3 4 …
 P n e u m …
 └─┬─┘
 [:3] → Pne → PNE
```

**What it is not:** `//` is not “percent.” `%` is remainder (`5 % 2 == 1`), and also the old `"%s" % name` string style. Prefer f-strings.
""",
)

_put(
    "Python",
    "Control flow",
    r"""
Think of a **single decision tree**: each visit gets **one** bucket. `if / elif / else` so two branches cannot both win.

```python
def bucket(esi, spo2, arrival):
    if spo2 < 92 or esi == 1:
        return "resusc"
    if esi == 2 or arrival == "ambulance":
        return "urgent"
    if esi >= 4:
        return "fast_track"
    return "standard"
```

**Example 1** — `esi=3`, `spo2=88`, `arrival="walk_in"`

First `if`: `88 < 92` is True → **`resusc`**. Stop. Never look at ESI 3.

**Example 2** — `esi=2`, `spo2=98`, `arrival="walk_in"`

First `if` false. Second True (esi==2) → **`urgent`**.

**Example 3** — `esi=5`, `spo2=99`, `arrival="walk_in"`

First two false. `esi >= 4` → **`fast_track`**.

```
spo2<92 or esi==1 ? ──yes──► resusc
        │ no
esi==2 or ambulance ? ──yes──► urgent
        │ no
    esi>=4 ? ──yes──► fast_track
        │ no
        └──► standard
```

**What it is not:** two separate `if`s that can both assign. Stacked `if` without `elif` can overwrite. The agent planner is the same shape: one next tool, not two.
""",
)

_put(
    "Python",
    "Comprehensions",
    r"""
Think of a **one-line factory**: make a new list/dict from an old one, maybe filtered.

```python
high = [c for c, a in zip(conditions, admit_base) if a >= 0.25]
los_map = {c: h for c, h in zip(conditions, typical_los_h)}
first, *mid, last = conditions
```

**Example 1 — list comp**

`admit_base` for Pneumonia is 0.42, URI is 0.03.

`if a >= 0.25` keeps Pneumonia, drops URI. `high` might be `['Pneumonia', 'Cardiac_rule_out']`.

**Example 2 — dict comp**

Each condition name becomes a key, hours become the value. `los_map["UTI"]` → `3.5`.

**Example 3 — unpack**

`conditions` has 8 names. `first, *mid, last = conditions` → first = URI, last = Cardiac_rule_out, `mid` is the six in between.

```
zip(names, rates) → (URI, 0.03), (Pneumonia, 0.42), …
filter a>=0.25    → (Pneumonia, 0.42), …
keep the name     → ["Pneumonia", …]
```

**What it is not:** a place for `print` or `append` side effects. If you need early `break`, use a `for` loop. If you only iterate once, a generator `(x for x in …)` is enough.
""",
)

_put(
    "Python",
    "OOP",
    r"""
Think of a **chart row with behavior**. Data fields plus questions you can ask (`hypoxia`, `acuity`).

```python
@dataclass
class Encounter:
    condition: str
    esi: int
    spo2: float
    symptoms: list[str] = field(default_factory=list)

    @property
    def hypoxia(self) -> bool:
        return self.spo2 < 92
```

**Example 1**

`Encounter("Pneumonia", 2, 88.0)` → `e.hypoxia` is **True** (88 < 92). You did not store `hypoxia`; it is computed.

**Example 2 — default_factory**

Each `Encounter()` gets its **own** empty `symptoms` list. Two patients do not share one list.

**Example 3 — Board**

`len(board)` uses `__len__`. `board.high()` counts rows with `acuity()=="high"`. sklearn estimators are the same idea: an object with `fit` / `predict`.

```
Encounter(condition, esi, spo2)
        │
        ├── .hypoxia   (property: spo2 < 92)
        └── .acuity()  (method: high vs standard)
```

**What it is not:** `symptoms: list = []`. That empty list is created **once** and shared — the same bug as `def f(xs=[])`.
""",
)

_put(
    "Python",
    "Exceptions",
    r"""
Think of a **bouncer**: bad ESI does not silently become 3. You raise, then catch **that** error if you can recover.

```python
def parse_esi(raw: str) -> int:
    if raw not in {"1", "2", "3", "4", "5"}:
        raise ValueError(f"ESI {raw!r} not in 1–5")
    return int(raw)
```

**Example 1** — `parse_esi("3")` → **3**. Status ok.

**Example 2** — `parse_esi("0")` → **ValueError: ESI '0' not in 1–5**.

**Example 3 — recover**

```python
try:
    esi = parse_esi(raw)
except ValueError:
    esi = 3   # explicit fallback, you chose it
```

```
raw="stat" → not in {1..5} → raise ValueError
                │
         except ValueError → fallback
         except Exception  → too wide (hides bugs)
```

**What it is not:** `except Exception:` around `pipe.predict`. That swallows a missing column and looks like a model bug later. Agents should fail the tool and put the error in the trace.
""",
)

_put(
    "Python",
    "Files & JSON",
    r"""
Think of **take the chart off the page**: a dict → a text blob → a file.

```python
payload = {"clinic": "Northshore", "n": len(rows), "encounters": rows}
blob = json.dumps(payload)
path = Path("data") / "encounters.parquet"
```

**Example 1 — dumps**

`{"n": 12, ...}` becomes one string. `json.loads(blob)` gives the dict back. Types: `True` stays bool; keys are strings.

**Example 2 — Path**

`Path("data") / "encounters.csv"` joins folders. No `"data" + "/" + name` mistakes on Windows vs Mac.

**Example 3**

CSV download of 12 rows is `to_csv`. Parquet is better for ML (types, NaNs, size). The site bakes parquet in `data/`.

```
dict  --json.dumps-->  str  --write-->  file
dict  <--json.loads--  str  <--read---  file
```

**What it is not:** JSON as a replacement for parquet for 10k typed rows. JSON is the agent observation / API payload.
""",
)

_put(
    "Python",
    "Datetime",
    r"""
Think of **when** as an object, not a string. Split train/test on a clock, not on shuffled rows.

```python
cutoff = start + timedelta(days=21)
enc["hour"] = pd.to_datetime(enc["ts"]).dt.hour
```

**Example 1** — `timedelta(days=1)` added to Monday 00:00 → Tuesday 00:00.

**Example 2** — `.dt.hour` on `2026-05-04 19:12` → `19` (rush feature).

**Example 3 — naive vs aware**

A timestamp with no timezone is **naive**. Mixing UTC logs with local clinic hours **shifts** your cutoff. Store UTC, convert for display, split on one clock.

```
ts = 2026-05-04 19:12
         │
    .dt.hour → 19
    .dt.dayofweek → 0=Mon … 6=Sun
```

**What it is not:** comparing date strings like `"5/4/26" < "12/1/26"` (lexicographic). Parse first.
""",
)

_put(
    "Python",
    "collections",
    r"""
Think of **stdlib batteries** for counting and grouping so you do not write a slow dict by hand.

```python
from collections import Counter, defaultdict, deque
by_site = Counter(enc["site"])
flags = defaultdict(list)
trace = deque(maxlen=8)
```

**Example 1 — Counter**

Five Downtown and two Campus in a toy list → `Counter({'Downtown': 5, 'Campus': 2})`.

**Example 2 — defaultdict(list)**

`flags["spo2"].append("low")` works even if `"spo2"` was missing. No `if key not in d: d[key]=[]`.

**Example 3 — deque(maxlen=8)**

Agent last-N thoughts. Push 9th item, the oldest drops. O(1).

```
trace: [get_chart, flag_labs, score_admit, stop]
maxlen=3 → after four pushes: [flag_labs, score_admit, stop]
```

**What it is not:** `defaultdict` is not thread-safe magic. It is “missing key → call factory.”
""",
)

_put(
    "Python",
    "Generators",
    r"""
Think of a **faucet**, not a bucket. `yield` gives one row, then pauses. Memory stays small.

```python
def stream_encounters(rows):
    for r in rows:
        yield r

preview = list(islice(stream_encounters(rows), 12))
```

**Example 1** — `next(gen)` is row 1. Another `next` is row 2. The rest are not in memory yet.

**Example 2** — `list(gen)` pulls **everything**. Then `list(gen)` again is **empty** (one-shot).

**Example 3** — `islice(gen, 12)` is “only the first 12,” like `head`.

```
yield row1 → pause
yield row2 → pause
… until the loop ends
```

**What it is not:** a list. No `len(gen)` until you materialize. If you need two passes, use a list.
""",
)

_put(
    "Python",
    "itertools",
    r"""
Think of **every pair** without nested `for` loops you have to get right.

```python
from itertools import product
cross = list(product(["Downtown", "Campus"], [1, 2]))
# [('Downtown', 1), ('Downtown', 2), ('Campus', 1), ('Campus', 2)]
```

**Example 1** — 2 sites × 2 ESI values = **4** pairs. That is a dummy grid / test matrix.

**Example 2** — 5 sites × 5 ESI = 25 rows. Same idea as OneHot `site` × `esi` columns (sklearn builds columns; `product` builds the pairs).

**Example 3 — groupby** (itertools) needs **sorted** input. Pandas `groupby` does not. Different tools, same word.

```
sites × esi
Downtown ─ 1
Downtown ─ 2
Campus   ─ 1
Campus   ─ 2
```

**What it is not:** `product` multiplying DataFrame columns in place. It is Cartesian product of iterables.
""",
)

_put(
    "Python",
    "Typing",
    r"""
Think of a **sticky note for humans and checkers**, not a lock at runtime.

```python
from typing import Callable, Literal
ToolFn = Callable[..., dict]
Band = Literal["low", "mid", "high"]
```

**Example 1** — `Band = "high"` is fine. `Band = "urgent"` is a type-checker error, still runs in plain Python.

**Example 2** — `def plan(...) -> tuple[str, str, dict]:` documents the planner return. Python will not enforce it unless you add pydantic/beartype.

**Example 3** — `Literal` is the admit **band**, not a free-form string in the agent JSON.

```
p_admit=0.61 → band "high"   (you defined the three strings)
p_admit=0.61 → "HOT"         (checker complains; runtime still a str)
```

**What it is not:** “Python 3.12 checks types at runtime.” It does not, by default.
""",
)

_put(
    "Python",
    "Decorators",
    r"""
Think of **wrapping a tool** with a counter or a timer without changing the tool’s insides.

```python
from functools import wraps
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
```

**Example 1** — `get_chart(1)` then `get_chart(2)` → `get_chart.n` is **2**.

**Example 2** — `@wraps(fn)` keeps `__name__` as `"get_chart"`, not `"wrapper"`. Debugging and registries need that.

**Example 3** — Agent tools are the same pattern: log, time, catch, then call.

```
call get_chart(1)
   wrapper.n: 0 → 1
   then real get_chart(1)
```

**What it is not:** `@counted` making the function async. It only wraps.
""",
)

_put(
    "Python",
    "Context managers",
    r"""
Think of **borrow / return**. `with` always runs the cleanup, even if the body raises.

```python
from contextlib import contextmanager
@contextmanager
def seed_scope(rng_seed):
    # enter: isolate RNG
    yield
    # exit: restore
```

**Example 1** — `with open(path) as f:` closes the file if JSON parse fails.

**Example 2** — `try/finally` can do the same; `with` is the readable form.

**Example 3** — A long agent loop that opens files without `with` leaks handles.

```
enter → run body → exit     (happy path)
enter → body raises → exit  (still cleaned up)
```

**What it is not:** `with` disabling exceptions. The error still propagates after `__exit__`.
""",
)

_put(
    "NumPy",
    "ndarray shape / dtype",
    r"""
Think of an array as a **box with a size tag**. The tag is `.shape`. `len` only reads the first number.

```python
lab_means.shape    # (8, 14)  8 conditions × 14 biomarkers
assay_cost.shape   # (14,)
```

**Example 1** — `len(lab_means)` is **8**, not 112. The 14 is the second axis.

**Example 2** — `(14,)` vs `(14, 1)`: both have 14 numbers. `@` and broadcast **care**. Print both shape and dtype.

**Example 3** — `lab_means.dtype` is `float64`. An object array of `None` kills vectorization.

```
lab_means
  8 rows (conditions)
  14 cols (assays)
  shape tag: (8, 14)
```

**What it is not:** “shape is only for images.” Every ML array has one.
""",
)

_put(
    "NumPy",
    "matmul / @",
    r"""
Think of **@ as “times and add along the shared edge.”** `*` is “times each cell, no add.”

```python
panel_cost = protocol @ assay_cost     # (8,14) @ (14,) → (8,)
```

**Example 1** — Pneumonia row of `protocol` is which assays fire. Dot with `assay_cost` → **one dollar figure** for that condition.

**Example 2** — `protocol * assay_cost` broadcasts cost onto each indicated assay but **does not sum** the panel.

**Example 3** — `counts @ protocol` → tonight’s assay **pull** (14 numbers). Same multiply-add.

```
protocol row (14)  ·  cost (14)  =  one panel_cost
1 1 0 1 …          ·  4 6.5 …    =  dollars
```

**What it is not:** `@` only for square matrices. `(8,14) @ (14,)` is legal.
""",
)

_put(
    "NumPy",
    "Broadcasting",
    r"""
Think of a **short ruler reused on every row**. NumPy compares sizes **from the right**.

```python
z = (x - ref_low) / (ref_high - ref_low)
# x is (14,) or (n, 14); ref is (14,)
```

**Example 1** — `vec` Pneumonia `(14,)` minus `ref_low` `(14,)` → element 0 vs 0, …, 13 vs 13.

**Example 2** — `X` shape `(100, 14)` minus `ref_low` `(14,)` → the 14 limits apply to **every** of 100 rows. That is broadcast.

**Example 3 — fail** — `(8, 14) - (8,)` does **not** work. Need `(8, 1)` so the 8 lines up on the left.

```
(n, 14)
    14     ← ref_low (14,) stretches across n rows
align from the RIGHT
```

**What it is not:** making a giant copied tiled array first. And not `@` (that is matmul).
""",
)

_put(
    "NumPy",
    "Boolean masks",
    r"""
Think of a **row of True/False lights**, one per assay. True = out of range.

```python
abnormal = (vec < ref_low) | (vec > ref_high)
names = np.array(biomarker_names)[abnormal]
```

**Example 1** — Pneumonia typical SpO2 90.5 vs band 95–100 → that slot is **True** (low).

**Example 2** — WBC 14.5 vs 4–11 → **True** (high). SBP 108 vs 95–140 → **False**.

**Example 3** — `names[abnormal]` keeps only lit assays. The agent’s `flag_labs` is this mask.

```
vec:     14.5   90.5   108
band:    4–11   95–100 95–140
mask:    True   True   False
```

**What it is not:** a Python `for biomarker in …` loop. Vectorized `<` / `>` stay in C.
""",
)

_put(
    "NumPy",
    "Fancy / integer index",
    r"""
Think of **histogram then multiply**. Each encounter is a condition id 0–7. Count them, then `@ protocol`.

```python
counts = np.bincount(idx, minlength=8)   # (8,)
pull = counts @ protocol                 # (14,)
```

**Example 1** — twenty visits, eight Pneumonia (id 2). `counts[2]` is 8.

**Example 2** — `counts @ protocol` = how many of each assay to draw tonight.

**Example 3** — a Python `for` over 20_000 encounters is slower; this is one pass + one matmul.

```
ids:  0 2 2 5 2 …
bincount → [n_URI, n_flu, n_pneumonia, …]
         @ protocol → 14 assay counts
```

**What it is not:** `bincount` on floats (ids must be non-negative ints).
""",
)

_put(
    "NumPy",
    "View vs copy",
    r"""
Think of a **window vs a photocopy**. Changing a window changes the original atlas.

```python
view = lab_means[0]        # often a view
safe = lab_means[0].copy()
view[:] = 0                # can wipe Pneumonia’s row in lab_means
```

**Example 1** — `lab_means[0].base is not None` usually means view.

**Example 2** — `lab_means[[0]]` (fancy index) usually **copies**.

**Example 3** — Agent uses `lab_means` for nearest prototype. Mutating a view corrupts every later guess.

```
lab_means ──slice──► view  (same memory)
lab_means ──copy──► safe   (separate)
```

**What it is not:** “all indexing copies.” Slices often don’t.
""",
)

_put(
    "NumPy",
    "Axis reductions",
    r"""
Think of **collapse this direction**. `axis=0` on `(8,14)` eats the 8 (conditions) and leaves 14 assay means.

```python
lab_means.mean(axis=0)   # (14,) per biomarker
inventory.sum(axis=1)    # (5,)  vials per site
```

**Example 1** — mean WBC across 8 conditions: one number, first slot of `mean(axis=0)`.

**Example 2** — Downtown row of inventory summed across 14 assays: one total for that site (`axis=1`).

**Example 3** — Draw the shape, then cross out the axis you named.

```
(8, 14)  mean axis=0  →  (14,)
(8, 14)  mean axis=1  →  (8,)
```

**What it is not:** “axis=0 always means columns like Excel.” It means **the first index of this array**.
""",
)

_put(
    "NumPy",
    "clip / nan",
    r"""
Think of **NaN = empty float cell**. `None` does not fit in a float array.

```python
x = np.clip(x, ref_low, ref_high * 1.5)
missing = np.isnan(x)
```

**Example 1** — lactate not ordered → `np.nan` in that slot. `np.isnan` is True there.

**Example 2** — `None` in a float array → whole array becomes **object** dtype. Vectorization dies.

**Example 3** — `clip` bounds wild values; it does **not** remove NaNs.

```
[14.5, nan, 90.5]
          ↑
     assay not run  (not 0.0)
```

**What it is not:** `NaN == None`. They do not compare equal. sklearn SimpleImputer talks NaN.
""",
)

_put(
    "NumPy",
    "L2 nearest prototype",
    r"""
Think of **eight typical lab cards**. New vector: which card is closest in 14-D? That is retrieval, not a diagnosis.

```python
dist = np.linalg.norm(lab_means - x, axis=1)  # (8,)
guess = condition_names[int(np.argmin(dist))]
```

**Example 1** — `x` is typical Pneumonia → distance 0 to that row → guess **Pneumonia**.

**Example 2** — a mixed vector might be nearest Influenza even if gold is URI.

**Example 3** — agent uses the guess only to `retrieve_protocol`. Gold label stays hidden.

```
x  vs  8 mean vectors
        shortest arrow → guess
```

**What it is not:** “the model diagnosed pneumonia.” Say prototype / nearest mean.
""",
)

_put(
    "Pandas",
    "DataFrame / Series",
    r"""
Think of a **spreadsheet**: columns share a row index. One column alone is a Series.

```python
enc.head()
enc.dtypes
```

**Example 1** — `enc["spo2"]` is a Series (one column). `enc[["spo2","hr"]]` is a DataFrame.

**Example 2** — `len(enc)` is visits. `enc.shape[1]` is columns.

**Example 3** — models want a DataFrame of features and a Series `y` (admit).

```
DataFrame  = several Series glued by index
Series     = one labeled column
```

**What it is not:** a raw NumPy matrix with no index. The index is how `loc` and joins line up.
""",
)

_put(
    "Pandas",
    "loc / boolean filter",
    r"""
Think of a **highlighter**: True rows stay.

```python
hypoxia = enc[enc.spo2 < 92]
campus = enc.loc[enc.site.eq("Campus") & enc.esi.isin([1, 2]), "spo2"]
```

**Example 1** — `enc.spo2 < 92` is a Series of True/False the same length as `enc`. `enc[that]` keeps True rows.

**Example 2** — `loc[mask, "spo2"]` is only the SpO2 column of those rows.

**Example 3** — `iloc[0]` is “first row by position,” not “encounter_id 0.”

```
mask: F F T F T …
enc:  row row KEEP row KEEP
```

**What it is not:** `enc[enc.esi==1]["wait_min"] = 0` (chained). Use `enc.loc[mask, "wait_min"] = 0`.
""",
)

_put(
    "Pandas",
    "groupby agg",
    r"""
Think of **stack same keys, then summarize**. One row per group.

```python
enc.groupby(["site", "condition"], observed=True).agg(
    n=("encounter_id", "count"),
    admit_rate=("admit", "mean"),
)
```

**Example 1** — Downtown + Pneumonia → one row: how many visits, what fraction admitted.

**Example 2** — `observed=True` skips empty categorical combos (Campus × rare condition with n=0).

**Example 3** — without groupby you would `for site in …` and mess up the index.

```
many encounter rows
    group keys (site, condition)
        → one summary row each
```

**What it is not:** `transform` (that puts the summary **back on every original row**).
""",
)

_put(
    "Pandas",
    "transform",
    r"""
Think of **stamp every row with its group’s number**, same index as `enc`.

```python
enc["site_wait"] = enc.groupby("site")["wait_min"].transform("mean")
enc["vs_site"] = enc["wait_min"] / enc["site_wait"]
```

**Example 1** — Downtown mean wait is 22. A visit with wait 33 gets `site_wait=22`, `vs_site=1.5`.

**Example 2** — Harbor visit is compared to **Harbor** mean, not the whole network.

**Example 3** — `agg` would be one row per site. You could not divide row-by-row without a merge. `transform` skips that merge.

```
each row: wait / (mean wait of that site)
index unchanged
```

**What it is not:** `agg`. `agg` shrinks; `transform` does not.
""",
)

_put(
    "Pandas",
    "merge",
    r"""
Think of **tape comorbidities onto each visit** by `patient_id`. Count rows before and after.

Toy night: 6 encounters, P99 not in `patients`, P05 never visited.

**Example 1 — left (course default)**

Keep all 6 charts. P99 gets NaN copd/payer. `len` stays 6.

**Example 2 — inner**

Drop P99. `len` becomes 5. First visits are not random — admit rate shifts.

**Example 3 — duplicate patient row**

Two warehouse rows for P01 → E1 and E2 duplicate → `len` grows. Deduplicate the right side.

```
Venn keys:  P99 | P01–P04 | P05
left rows = left circle (6)
inner rows = overlap only (5)
```

**What it is not:** “left join duplicates columns so avoid it.” Left is the correct grain. Count `len`.
""",
)

_put(
    "Pandas",
    "Missing data",
    r"""
Think of **empty because nobody ordered the test**, not a random hole.

```python
enc["lactate"] = enc["lactate"].fillna(
    enc.groupby(["site", "esi"])["lactate"].transform("median")
)
```

**Example 1** — `dropna()` on lactate **removes** the sickest-looking charts (the ones you care about).

**Example 2** — site+ESI median is a better stamp than a global median.

**Example 3** — doing this on the **full** frame before split leaks the test distribution. Put `SimpleImputer` **inside** a Pipeline on train.

```
missing lactate  ≠  lactate was 0
drop those rows  ≠  MCAR
```

**What it is not:** “sklearn requires dropna first.” It requires an imputer in the Pipeline.
""",
)

_put(
    "Pandas",
    "Time series",
    r"""
Think of **yesterday’s census, not today’s**, as a feature.

```python
g = daily.groupby("site")["encounters"]
daily["enc_lag7"] = g.shift(7)
daily["enc_roll7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
```

**Example 1** — Monday’s `enc_lag7` is last Monday at that site.

**Example 2** — without `shift(1)` before `rolling(7)`, **today** is inside the average used to predict today. Leakage.

**Example 3** — always `groupby("site")` first or Harbor Monday lags Downtown Sunday.

```
days:  … D-7  D-6 … D-1  D(today)
lag7:            ↑______________│
roll:     mean of D-7..D-1 only
```

**What it is not:** a global `shift` on the stacked frame.
""",
)

_put(
    "Pandas",
    "crosstab / pivot",
    r"""
Think of a **grid**: rows × columns, cells are counts or means.

```python
pd.crosstab(enc["site"], enc["condition"])
enc.pivot_table(index="site", columns="hour", values="admit", aggfunc="mean")
```

**Example 1** — crosstab: how many Downtown Pneumonia visits (a count).

**Example 2** — pivot_table mean admit: Downtown at 19:00 admit **rate**. Pair with a count grid so n=2 is not 100%.

**Example 3** — `fill_value=0` so missing combos are zero, not NaN that later get imputed.

```
          URI  Pneumonia
Downtown   40     12
Campus     55      8
```

**What it is not:** only for integers. `pivot_table` can `mean` a bool admit column.
""",
)

_put(
    "Pandas",
    "Categorical",
    r"""
Think of a **menu of allowed labels**, not free text.

```python
enc["esi"] = pd.Categorical(enc["esi"], categories=[1,2,3,4,5], ordered=True)
enc["site"] = enc["site"].astype("category")
```

**Example 1** — ordered ESI: `esi == 1` vs `esi < 3` is meaningful.

**Example 2** — less memory than 10k copies of `"Downtown"`.

**Example 3** — `observed=True` on groupby uses combinations that **appear**, not the full menu.

```
site category = {Downtown, Airport, Campus, Harbor, Suburb}
not 10k separate strings
```

**What it is not:** a reason to one-hot `patient_id`. High cardinality IDs stay IDs.
""",
)

_put(
    "EDA",
    "Line / bar / box",
    r"""
Think of **looking before you model**. If you cannot see the flu wave on a line, do not encode it.

**Example 1 — line:** network census vs date. A bump in flu weeks is a feature candidate.

**Example 2 — bar:** hour × arrival. Ambulance mix at 19:00 is not the same as 09:00.

**Example 3 — box:** wait vs ESI. ESI 1–2 should sit lower (SLA). If not, the generator or the floor is lying.

```
date →  census
         ╱╲    flu week
        ╱  ╲
```

**What it is not:** a Pipeline step. Charts do not belong inside `fit`.
""",
)

_put(
    "EDA",
    "Heatmap",
    r"""
Think of a **colored grid of rates**. Always ask: how many visits in that cell?

**Example 1** — Downtown × hour 19 admit rate 0.4 with n=200 is a pattern.

**Example 2** — same 0.4 with n=2 is a coin flip. Pair rate heatmap with count heatmap.

**Example 3** — empty cells are missing combos, not zero risk.

```
rate without n  =  a lie that looks like science
```

**What it is not:** proof of causation.
""",
)

_put(
    "EDA",
    "Scatter",
    r"""
Think of **two numbers, one dot, color = site**. Color stops Simpson’s paradox from hiding.

**Example 1** — volume vs admit rate, all sites pooled, slope one way.

**Example 2** — color by site: each site’s slope can reverse the pooled slope.

**Example 3** — wait vs SpO2: hypoxia should not wait. Outliers are ESI mistakes or generator noise.

```
pooled cloud  vs  five colored clouds
```

**What it is not:** a third axis you ignore. Facet or color first.
""",
)

_put(
    "EDA",
    "Correlation",
    r"""
Think of **how two columns move together**, not “drop one.”

**Example 1** — `enc_lag7` vs today is **high** — that is the signal (last week predicts this week).

**Example 2** — same-day `admit_rate` vs volume is high **and illegal** as a 9am feature.

**Example 3** — Pearson on mixed sites can be Simpson again. Check by site.

```
high corr + known at dawn     → keep (lag-7)
high corr + outcome cousin    → leakage
```

**What it is not:** corr > 0.3 means drop. And not causation.
""",
)

_put(
    "EDA",
    "Plotly + seaborn",
    r"""
Think of **two cameras**: Plotly = interactive filter; seaborn = static sample.

**Example 1** — `px.line(..., color="site")` lets you hide Harbor.

**Example 2** — 10k seaborn points hide density. `sample(800)` or hexbin.

**Example 3** — neither camera is a training step.

```
explore  →  encode  →  Pipeline.fit
 (charts)     (code)      (not charts)
```

**What it is not:** Plotly fitting the admit model.
""",
)

_put(
    "Features",
    "Calendar features",
    r"""
Think of **numbers you already know at dawn**.

```python
feat["is_weekend"] = feat["dow"].isin([5, 6]).astype(int)
```

**Example 1** — Monday `dow=0`, `is_weekend=0`. Saturday `dow=5`, `is_weekend=1`.

**Example 2** — `flu_wave` is OK only if you have a **forecast** at 9am, not the same-day realized wave from today’s census.

**Example 3** — known-at-inference is the leakage test. If you would not have it on the live ticket, it is not a feature.

```
dawn:  dow, weekend, maybe flu forecast
not dawn:  today’s admit_rate, today’s realized wave from volume
```

**What it is not:** one-hot `patient_id` as a “calendar” trick.
""",
)

_put(
    "Features",
    "Lag / roll",
    r"""
Think of **last week** and **the past week’s average**, per site.

```python
d["enc_lag7"] = g.shift(7)
d["enc_roll7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
```

**Example 1** — Tuesday Harbor `enc_lag7` = last Tuesday Harbor.

**Example 2** — `shift(1)` before rolling so **today is not in the mean**.

**Example 3** — without `groupby("site")`, Harbor uses Downtown’s history.

```
D-7 is lag7
D-7 … D-1 mean is roll7
D is the target — stay out of the features
```

**What it is not:** rolling including today.
""",
)

_put(
    "Features",
    "Time split vs random",
    r"""
Think of **train on the past, test on the future**. Random split cheats on census.

**Example 1** — random 75/25: Tuesday in train, Monday in test, same flu week → inflated R².

**Example 2** — cutoff = 75th percentile of dates. Train ≤ cutoff, test after.

**Example 3** — same LinearRegression, two R²s. The honest one is time split.

```
time:  [ train days | test days ]
random: train and test mixed in the same week
```

**What it is not:** “random is unbiased for all data.” Not for time series.
""",
)

_put(
    "Features",
    "Leakage",
    r"""
Think of **the model saw the answer key**.

**Example 1** — same-day `admit_rate` as a feature for today’s census. You do not have that at 9am.

**Example 2** — `OneHotEncoder.fit` on the **full** frame (test frequencies leak).

**Example 3** — gold `condition` in `get_chart`. The planner copies the label.

```
illegal:  outcome cousins, full-frame fit, gold in the agent
legal:    lag-7, ESI, site, train-only Pipeline
```

**What it is not:** using ESI. ESI is known at triage.
""",
)

_put(
    "Features",
    "Target encoding leak",
    r"""
Think of **mean admit by condition including this row’s own admit**.

```python
# LEAK
enc["leaky"] = enc.groupby("condition")["admit"].transform("mean")
```

**Example 1** — a rare condition with 2 admits / 2 visits → leaky value 1.0, and both rows saw that.

**Example 2** — the test set’s admits are inside the mean. Test is not held out.

**Example 3 — legal** — means from **train folds only**, map to test, smooth rare levels.

```
full-table groupby(target)  =  the label wearing a costume
```

**What it is not:** “target encoding is always illegal.” Out-of-fold on train is OK.
""",
)

_put(
    "Features",
    "High-cardinality IDs",
    r"""
Think of **memorizing who the patient is**, not how they present.

**Example 1** — ~1.6k `patient_id` one-hots → 1.6k columns of identity.

**Example 2** — a new MRN at test time: all zeros. The model never saw them.

**Example 3** — even old MRNs leak **past admit frequency** as if it were a vital.

```
patient_id → one-hot  =  overfit + leak + fails on new people
past visit count from history only  =  maybe OK
```

**What it is not:** “sklearn cannot one-hot strings.” It can. You still must not.
""",
)

_put(
    "Regression",
    "Problem",
    r"""
Think of **how many visits at this site on this day?** One row = one site-day, not one patient.

**Example 1** — Downtown 2026-05-12 → y = 42 encounters.

**Example 2** — mixing encounter-level ESI into that row without aggregating is the wrong grain.

**Example 3** — features: dow, weekend, flu, lag7, roll7, site, season.

```
grain: site × date → census
not:   one row per encounter
```

**What it is not:** predicting a biomarker.
""",
)

_put(
    "Regression",
    "Naive baseline",
    r"""
Think of **last week’s same weekday**. If you cannot beat it, you have no model. The floor already uses it.

**Example 1** — test MAE lag-7 = 12. Ridge MAE = 11.8. Barely a win.

**Example 2** — Ridge MAE = 18 vs lag-7 = 12. You lost. Do not ship.

**Example 3** — always print **both** numbers.

```
yhat_naive = enc_lag7
MAE = mean(|y - yhat|)
```

**What it is not:** a sklearn estimator you must import. It is a column you already built.
""",
)

_put(
    "Regression",
    "Linear / Ridge / RF",
    r"""
Think of **the same Pipeline, swap the last step**.

**Example 1** — OLS coefficients explode when lag7 and roll7 and site dummies are collinear.

**Example 2** — Ridge (`L2`) shrinks those coefficients. Same features, stabler.

**Example 3** — RF can memorize last Friday’s flu spike unless `min_samples_leaf` is up and the split is time-aware.

```
X → ColumnTransformer → Linear or Ridge or RF → yhat
```

**What it is not:** Ridge as a classifier. That is LogisticRegression.
""",
)

_put(
    "Regression",
    "MAE RMSE R²",
    r"""
Think of **three rulers**. Quote MAE in visits/day vs lag-7 to ops.

**Example 1** — MAE 9 means “off by 9 visits on average.” Same unit as the board.

**Example 2** — RMSE punishes a Friday miss of 40 harder than MAE.

**Example 3** — R² vs predicting the mean. Easy to inflate with a random split. Always pair with time-split MAE.

```
ops cares: MAE vs lag-7
debug:     RMSE, residual plots
slides:    R² (with a grain of salt)
```

**What it is not:** R² as the only number.
""",
)

_put(
    "Regression",
    "Residuals",
    r"""
Think of **miss = actual − predicted**. Plot vs predicted, color by site.

**Example 1** — mean residual ~0: no overall bias.

**Example 2** — fan shape: errors grow with volume (heteroscedasticity). Consider log or a count model.

**Example 3** — Harbor always +15: missing a site shift, not a generic ML fail.

```
y - yhat  vs  yhat
color = site
```

**What it is not:** “residuals look random so we are done.” Slice by site first.
""",
)

_put(
    "Regression",
    "Coefficients / importance",
    r"""
Think of **a debug hint**, not a causal effect.

**Example 1** — Ridge coefs are in **scaled** units. Do not compare raw age to lag7 without the scaler.

**Example 2** — RF impurity importance prefers high-cardinality features. Biased.

**Example 3** — permutation importance on the **time-split test** is more honest.

```
scaled Ridge coef  →  direction after equalizing units
impurity RF        →  suspicious
```

**What it is not:** “this coefficient is the effect of flu on census.”
""",
)

_put(
    "Classification",
    "Problem",
    r"""
Think of **will this visit admit?** One row = one encounter. Output is `P(admit)`, not a hard 0/1 yet.

**Example 1** — `predict_proba(row)[0, 1]` → `0.41`.

**Example 2** — `predict()` uses 0.5. Beds do not.

**Example 3** — the agent tool `score_admit` returns that probability plus a band.

```
features at triage → Pipeline → P(admit)
```

**What it is not:** a diagnosis of Pneumonia.
""",
)

_put(
    "Classification",
    "Class imbalance",
    r"""
Think of **most people go home**. Accuracy is a trap.

**Example 1** — admit rate 12%. Always discharge → **88% accuracy**, zero useful admits.

**Example 2** — quote recall/precision **on the admit class**, and PR-AUC.

**Example 3** — DummyClassifier `most_frequent` is the number you must beat.

```
100 visits, 12 admits
always "no" → 88 correct, 0 of 12 admits caught
```

**What it is not:** 88% is a good model.
""",
)

_put(
    "Classification",
    "LogReg / RandomForest",
    r"""
Think of **two ways to turn features into P(admit)**. `class_weight="balanced"` reweights the loss; you still pick a threshold.

**Example 1** — LogReg: linear, coefficients you can read after scaling.

**Example 2** — RF: interactions (ESI × SpO2) without writing them.

**Example 3** — `balanced` does **not** oversample the CSV. It changes the loss weights.

```
same X, y
  LogReg(class_weight=…)
  RF(class_weight=…)
then threshold on proba
```

**What it is not:** balanced ⇒ precision equals recall.
""",
)

_put(
    "Classification",
    "Threshold",
    r"""
Think of a **bed budget slider**, not a law of math.

**Example 1** — `thr=0.20`: more predicted admits, higher recall, more false alarms (hallway).

**Example 2** — `thr=0.50`: sklearn `predict()` default. Not a policy.

**Example 3** — `thr=0.35`: the course slider default. Ops owns this number.

```
proba  0.12  0.33  0.61
thr=0.35      no    no   yes
```

**What it is not:** RandomForest having no threshold. It still outputs votes/probas.
""",
)

_put(
    "Classification",
    "Confusion matrix",
    r"""
Think of **four buckets**. Name costs, do not hide behind F1.

**Example 1 — FN:** sent home, should have admitted. Clinical miss.

**Example 2 — FP:** flagged admit, went home. Wasted bed.

**Example 3** — sklearn default: rows = true, cols = predicted.

```
              pred home   pred admit
true home        TN           FP
true admit       FN           TP
```

**What it is not:** FP and FN always equal.
""",
)

_put(
    "Classification",
    "ROC and PR",
    r"""
Think of **ranking quality**. ROC can look strong from all the true negatives. PR cares about admits.

**Example 1** — ROC-AUC 0.90 with 12% admits can still be a weak admit ranking.

**Example 2** — PR-AUC focuses on the positive class. Prefer it here.

**Example 3** — a curve is not a threshold. You still pick an operating point.

```
ROC: TPR vs FPR     (TNs help)
PR:  precision vs recall  (admits only)
```

**What it is not:** they are identical.
""",
)

_put(
    "Classification",
    "Imputation in Pipeline",
    r"""
Think of **learn the median on train, replay it on new charts**.

```python
num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
```

**Example 1** — train median lactate 1.4. Test missing lactate becomes 1.4, not the test median.

**Example 2** — a global median before split **sees test**. Leakage.

**Example 3** — the agent’s `score_admit` loads this **one** object. Same medians in production.

```
fit on train → remember medians
predict on ticket → fill, scale, model
```

**What it is not:** impute then `train_test_split`.
""",
)

_put(
    "Clusters & pipelines",
    "K-means",
    r"""
Think of **k piles in scaled space**. You chose k. The piles are not diseases.

**Example 1** — k=4 on age, visits, admits, comorbid. Four groups.

**Example 2** — a new patient is assigned to the nearest center (Voronoi).

**Example 3** — if care management cannot name a pile, k is wrong.

```
dots in 6-D (after scaling)
4 centers → 4 labels
```

**What it is not:** a supervised diagnosis. There is no accuracy vs ICD.
""",
)

_put(
    "Clusters & pipelines",
    "Scaling",
    r"""
Think of **putting features on the same ruler** before Euclidean k-means.

**Example 1** — age ~ 70, comorbid count 0–4. Without scaling, age **is** the clustering.

**Example 2** — StandardScaler: mean 0, std 1 **on train**.

**Example 3** — trees do not need this; k-means and Ridge do.

```
raw:   age 72, comorbid 1
scaled: both on ~N(0,1) axes
```

**What it is not:** “k-means is invariant to scale.”
""",
)

_put(
    "Clusters & pipelines",
    "PCA",
    r"""
Think of a **2-D photograph of 6-D patients**. Pretty, not a diagnosis.

**Example 1** — pc1 might be “age + visits.” pc2 something else.

**Example 2** — clusters that look clean in 2-D can overlap in 6-D.

**Example 3** — do not cluster on 2 PCs just because the scatter looks nice unless you meant to.

```
6-D  --PCA-->  2-D scatter (visualization)
```

**What it is not:** PCA assigning segment labels. K-means does that.
""",
)

_put(
    "Clusters & pipelines",
    "Silhouette",
    r"""
Think of a **geometry hint**: higher = tighter/separated. Not a KPI.

**Example 1** — silhouette 0.22 vs 0.41: 0.41 is tighter, not “more true.”

**Example 2** — a named, actionable pile with 0.22 beats an unnamed 0.41.

**Example 3** — it likes convex blobs. Odd shapes score worse even if useful.

```
name the cluster in English  >  chase silhouette
```

**What it is not:** a test set replacement.
""",
)

_put(
    "Clusters & pipelines",
    "Phenotypes",
    r"""
Think of **English names for the piles**: older high-util, young low-touch, …

**Example 1** — segment 0: mean age 71, visits 6, comorbid 3 → “older high-util.”

**Example 2** — validate: does the same story appear next month? Can outreach use it?

**Example 3** — do not put **admit** into the cluster features if you will later predict admit.

```
groupby(segment).mean() → name it or drop k
```

**What it is not:** picking k by highest admit rate (that bakes the target in).
""",
)

_put(
    "Clusters & pipelines",
    "Pipeline",
    r"""
Think of **one object**: preprocess + model. What you `joblib.dump`. What `score_admit` loads.

```python
pipe = Pipeline([("prep", ColumnTransformer(...)), ("model", Ridge())])
pipe.fit(X_train, y_train)
```

**Example 1** — scaler medians/means remembered from **train**.

**Example 2** — `predict` on a new ticket replays the same transforms. No notebook drift.

**Example 3** — five pickle files that disagree is the failure mode this prevents.

```
ticket → prep (train stats) → model → yhat
```

**What it is not:** Pipeline replacing ColumnTransformer. It **contains** one.
""",
)

_put(
    "Clusters & pipelines",
    "TimeSeriesSplit + GridSearch",
    r"""
Think of **walk forward**: train on the past, score the next fold. Then try the next α.

**Example 1** — shuffled KFold: future Friday in train, past Monday in test. Cheat.

**Example 2** — TimeSeriesSplit n=4: four expanding windows.

**Example 3** — `GridSearchCV(..., cv=TimeSeriesSplit(), scoring="neg_mean_absolute_error")`.

```
fold1: train [d1–d10] val [d11–d14]
fold2: train [d1–d14] val [d15–d18]
```

**What it is not:** GridSearch requiring shuffled KFold.
""",
)

_put(
    "Agent",
    "Tool",
    r"""
Think of a **named button**: args in, dict out. The sklearn model is **one** button.

**Example 1** — `get_chart(encounter_id=20000)` → ESI, arrival, comorbidities (no gold).

**Example 2** — `flag_labs(...)` → NumPy mask of abnormal assays.

**Example 3** — `score_admit(...)` → `{p_admit: 0.41, band: "mid"}`.

```
name + schema + function  =  tool
Pipeline is score_admit, not the product
```

**What it is not:** a GPU kernel. It is a callable.
""",
)

_put(
    "Agent",
    "Planner",
    r"""
Think of **if-then that picks the next button**. Later you can swap this for an LLM that emits `{tool, args}`.

```python
if state.chart is None: return get_chart
if state.lab_flags is None: return flag_labs
...
return stop
```

**Example 1** — empty state → first tool is always `get_chart`.

**Example 2** — after labs, next is `score_admit`.

**Example 3** — you can unit-test this without OpenAI.

```
state  →  plan()  →  (thought, tool, args)
```

**What it is not:** the LLM being required to learn the architecture.
""",
)

_put(
    "Agent",
    "State / memory",
    r"""
Think of a **folder for this visit**: chart, flags, score, protocol, done.

**Example 1** — `state.chart is None` means we have not called `get_chart` yet.

**Example 2** — stuffing everything into a prompt hides whether labs already ran.

**Example 3** — same idea as a saga / workflow document in backend systems.

```
AgentState
  encounter_id
  chart, lab_flags, admit_score, protocol
  done, recommendation
```

**What it is not:** “prompts are always smaller.” Structured state is inspectable.
""",
)

_put(
    "Agent",
    "Trace",
    r"""
Think of **the audit log**: thought → tool → observation.

**Example 1** — step 2: “Flag labs” / `flag_labs` / `n_abnormal=3, spo2=88`.

**Example 2** — when the suggestion is wrong, you see which tool lied.

**Example 3** — eval compares guess vs gold **after** the run, not inside the thought.

```
1 get_chart → {esi:2, …}
2 flag_labs → {n_abnormal:3}
3 score_admit → {p_admit:0.41}
4 stop → recommendation
```

**What it is not:** the sklearn training log.
""",
)

_put(
    "Agent",
    "Stop + recommend",
    r"""
Think of **enough context, write a suggestion, halt**. Stop is a first-class action.

**Example 1** — after protocol is in state, `plan` returns `stop`.

**Example 2** — `max_steps=8` is a fuse so the loop cannot run forever.

**Example 3** — text says “Suggested (not a diagnosis)… Clinician must confirm.”

```
enough state  →  stop  →  recommendation string
infinite tools →  production incident
```

**What it is not:** stopping only when recall is 1.
""",
)

_put(
    "Agent",
    "Gold labels",
    r"""
Think of **the answer key in a sealed envelope**. Tools never open it. You score after stop.

**Example 1** — `get_chart` pops `gold_label_condition` before the planner sees the dict.

**Example 2** — if gold is in the prompt, an LLM will copy it. Same bug as target leakage.

**Example 3** — eval: `guess == gold` and `p_admit` vs `gold_admit`.

```
tools see: esi, spo2, labs
tools never see: true condition, true admit
```

**What it is not:** putting gold in the system prompt “for accuracy.”
""",
)

_put(
    "Agent",
    "Model as a tool",
    r"""
Think of **sklearn as one employee**, not the company.

**Example 1** — `joblib.dump(pipe)` once. `score_admit` loads it.

**Example 2** — same columns as training or you get train/serving skew.

**Example 3** — HTTP analog: `POST /score` plus a workflow engine — like the .NET pizza interview APIs.

```
app = planner + tools
one tool = Pipeline
```

**What it is not:** pickling the whole Streamlit session.
""",
)

_put(
    "Capstone",
    "Census board",
    r"""
Think of **tonight vs last week vs the model**, by site.

**Example 1** — Downtown yhat 40, actual 42, lag-7 38. Model added something.

**Example 2** — Harbor yhat 30, actual 18, lag-7 17. Site shift — not a generic win.

**Example 3** — season what-if: flu on/off.

```
site | actual | yhat | lag-7
```

**What it is not:** the board replacing a test set. It is the integration view.
""",
)

_put(
    "Capstone",
    "Reagent pull",
    r"""
Think of **condition mix × protocol matrix = how many assays to draw**.

```python
pull = counts @ protocol          # (14,)
cover = inventory[site] / pull
```

**Example 1** — lots of UTI tonight → more ua_nitrite in `pull`.

**Example 2** — `cover < 1` → that assay runs out.

**Example 3** — SQL can count conditions first; you still multiply. Same as recipe @ cost.

```
counts (8)  @  protocol (8,14)  →  pull (14)
```

**What it is not:** needing a square protocol.
""",
)

_put(
    "Capstone",
    "Admit desk",
    r"""
Think of **one live ticket** scored with the **same** Pipeline.

**Example 1** — ticket must have `esi_n`, `rush`, `lactate_f`, `site`, … same as training.

**Example 2** — new arrival code → `handle_unknown="ignore"` zeros those dummies. Seatbelt, not a feature.

**Example 3** — missing `rush` is train/serving skew. Notebook AUC will not save you.

```
live JSON → DataFrame with training columns → p_admit
```

**What it is not:** “any pretty JSON is fine.”
""",
)

_put(
    "Capstone",
    "Run agent",
    r"""
Think of **the whole loop on tonight’s board**: chart → labs → score → protocol → stop.

**Example 1** — `run_agent(clinic, admit_pipe, encounter_id)` returns state, trace, eval vs gold.

**Example 2** — Module 07 is a model. Capstone is the **product workflow**.

**Example 3** — gold still hidden. You only score after stop.

```
same tools as Module 09
on the same census / reagent / admit numbers as 06–08
```

**What it is not:** a new loss function. It is integration.
""",
)

