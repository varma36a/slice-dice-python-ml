# Agentic Doctor — Python for ML

End-to-end **Python for machine learning + agent workflow**, taught as a five-site urgent care (**Northshore Urgent Care**).

**GitHub repo:** [https://github.com/varma36a/slice-dice-python-ml](https://github.com/varma36a/slice-dice-python-ml)  
**Live app:** [https://agenticdoctorv2.streamlit.app/](https://agenticdoctorv2.streamlit.app/)

Synthetic educational data. **Not medical advice. Not a diagnostic product. Not for clinical use.**

Aimed at engineers who already write production code. NumPy, Pandas, EDA, leakage-safe features, sklearn, then a diagnostic **agent** that calls those models as tools.

## Run locally

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501).

## Course map

| Page | What you can do after it |
|---|---|
| Home | Clinic universe + topic catalog with **examples** |
| All Topics | Topic / concept / example / interview card (Dotnet-InterviewQuestions layout) |
| Interview questions | Live clinic demos, flashcards, multiple choice |
| Python for ML | Types through typing, each with encounter tables |
| NumPy labs | Lab-mean matrix, reference-range masks, protocol @ cost |
| Pandas chart | groupby/transform, merge comorbidities, missing lactate |
| EDA & charts | Flu wave, hypoxia, ESI — then encode them |
| Feature engineering | Time split vs random split vs illegal same-day admit rate |
| Regression | Beat lag-7 naive census |
| Classification | Admit risk; threshold as a bed budget |
| Clusters & pipelines | Phenotypes, `Pipeline` + `TimeSeriesSplit` |
| Agent workflow | Tools, planner, trace: chart → labs → model → protocol → stop |
| Capstone | Triage board: forecast, reagents, admit desk, run the agent |

All pages share one seeded universe (`clinic/data.py`, baked in `data/`). The admit `Pipeline` the classifier page fits is the same shape of object the agent calls as `score_admit`.

## All topics covered

The full catalog lives **on the site**: home (module tabs with **example + result**), sidebar **All Topics → By module** (every topic, not a truncated table), and **Interview questions**. Source: `clinic/topics.py`, `clinic/worked.py`.

Each topic is the same shape as the .NET interview map: **Topic · Concept · Example · Where**, plus a question you should answer out loud.

| Area | Topics |
|---|---|
| Python | types, collections, strings/numbers, control flow, functions, comprehensions, OOP, exceptions, files/JSON, datetime, collections, generators, itertools, typing, decorators, context managers |
| NumPy | shape/dtype, matmul, broadcasting, boolean masks, fancy index, view vs copy, axis reductions, clip/nan, L2 nearest prototype |
| Pandas | DataFrame, loc/boolean, groupby/agg, transform, merge, missing data, shift/rolling, crosstab, Categorical |
| EDA | line/bar/box, heatmap, scatter, correlation, Plotly + seaborn |
| Features | calendar, lag/roll, time vs random split, leakage, target encoding leak, high-cardinality IDs |
| Regression | census target, lag-7 naive, Linear/Ridge/RF, MAE/RMSE/R², residuals, coefficients |
| Classification | P(admit), imbalance, LogReg/RF, threshold as bed budget, confusion matrix, ROC/PR, Pipeline imputer |
| Clusters & pipelines | K-means, scaling, PCA, silhouette, phenotypes, Pipeline, TimeSeriesSplit + GridSearch |
| Agent | tools, planner, state, trace, stop/recommend, gold labels, model-as-tool |
| Capstone | census board, reagent pull, admit desk, run the agent |

Mutable default args (`def f(xs=[])`) are called out as a pitfall; dataclasses use `default_factory` instead.

## Interview live demos

On the site: **Interview questions → Live examples** (and **All Topics → By module**). These two are the ones people ask about. Synthetic educational data — not a real lab or MPI.

### Broadcast flags (Pneumonia vs ref)

Take the **typical Pneumonia lab vector** and mark each of 14 biomarkers out of range against one shared reference band — no Python `for` loop. Same math as Module 02 Lab B and the agent’s `flag_labs`.

Three 1-D arrays, same assay order (`wbc`, `crp`, `lactate`, `spo2`, …):

| Array | Shape | Meaning |
|---|---|---|
| `typical_Pneumonia` (`vec`) | `(14,)` | Pneumonia row of `lab_means` |
| `ref_low` | `(14,)` | Lower edge of the normal band |
| `ref_high` | `(14,)` | Upper edge |

```python
abnormal = (vec < ref_low) | (vec > ref_high)   # (14,) True/False
# WBC 14.5 vs 4–11 → high; SpO2 90.5 vs 95–100 → low; CXR infiltrate 0.82 vs 0–0.5 → high
```

On this atlas that usually flags **WBC, CRP, lactate, SpO2 (low), HR, temp, sodium (just low), CXR infiltrate**.

Here `vec` and `ref_*` are both `(14,)`, so the compare is elementwise. The interview word **broadcast** is for the next shape:

```python
X            # (n_patients, 14)
abnormal = (X < ref_low) | (X > ref_high)   # (n, 14)
```

NumPy **aligns from the right**. `(14,)` stretches across every row of `(n, 14)`: one band, every patient. `(n, 14)` vs `(14,)` works; `(8, 14)` vs `(8,)` does not unless you reshape to `(8, 1)`.

`*` is elementwise; `@` is matmul (`protocol @ assay_cost`). Flags are compare + `|`, not `@`.

Z vs the band is the same broadcast: `z = (x - ref_low) / (ref_high - ref_low)`.

**Say this:** Pneumonia’s 14-vector vs a 14-vector reference; the boolean mask is out-of-range assays. For a board, `(n, 14) < (14,)` broadcasts the same band — that is `flag_labs`, not a diagnosis.

### Merge row counts (with Venn)

Join audit after `encounters ⨝ patients` on `patient_id`. **Venn diagrams are keys. The live demo counts rows.** Same lesson as Module 03: `how="left"`, then `len` before and after.

Toy night — 6 charts, 5 patient rows:

**encounters (left, 6 rows)**

| encounter | patient_id | note |
|---|---|---|
| E1 | P01 | Downtown |
| E2 | P01 | same MRN, second visit |
| E3 | P02 | |
| E4 | P03 | |
| E5 | **P99** | first visit — **not** in `patients` |
| E6 | P04 | |

**patients (right, 5 IDs)**

| patient_id | note |
|---|---|
| P01, P02, P03, P04 | overlap |
| **P05** | on file, **no visit tonight** |

#### Venn of `patient_id` keys (not rows)

```
          encounters                         patients
         (left keys)                        (right keys)

              ┌─────────────────┐       ┌─────────────────┐
              │                 │       │                 │
              │  P99            │       │            P05  │
              │     ┌───────────┼───────┼───────────┐     │
              │     │  P01      │       │           │     │
              │     │  P02      │ MATCH │           │     │
              │     │  P03      │       │           │     │
              │     │  P04      │       │           │     │
              │     └───────────┼───────┼───────────┘     │
              │  left-only      │       │  right-only     │
              └─────────────────┘       └─────────────────┘
```

- **Left-only:** P99 — chart exists, no comorbidity row.
- **Overlap:** P01, P02, P03, P04.
- **Right-only:** P05 — never appears in `enc.merge(...)` at all.

P01 is **one key** in the overlap and **two encounter rows** (E1, E2). The Venn shrinks IDs; `len()` counts visits.

#### The three live-demo numbers

```python
merged = enc.merge(patients, on="patient_id", how="left")
len(enc)                                              # 6  encounters
len(merged)                                           # 6  after left (if patient_id unique)
enc["patient_id"].isin(patients["patient_id"]).sum()  # 5  inner would keep
```

**1. Inner join = overlap only**

Keep visits whose `patient_id` is in both circles. **Drop P99 (E5).** P05 still unused.

Result: **5 rows** (E1, E2, E3, E4, E6).

That is the bug: first-time / unmatched MRNs are not a random 1/6. Admit rate and `get_chart` then train on a biased subset.

```
Inner = overlap only

   encounters ●════● patients
              ║ P01 P02 P03 P04 ║     P99 gone
                                P05 unused
   rows kept: E1 E2 E3 E4 E6     (5)
```

**2. Left join = whole left circle (what the course uses)**

Keep **every encounter**. Overlap gets `copd` / `payer`. P99 stays, comorbidities are **NaN**. P05 is still ignored.

Result: **6 rows**. `len(merged) == len(enc)`.

```
Left = entire left circle

   encounters ●════● patients
   P99 stays (nulls)  ║ match ║     P05 unused
   rows kept: E1–E6              (6)
```

```python
enc.merge(patients, on="patient_id", how="left")
# then count nulls on copd — do not drop those charts
assert len(merged) == len(enc)   # after patients.patient_id is unique
```

**3. Duplicate right key = Venn unchanged, row count explodes**

If `patients` has **two** rows for P01, the Venn still shows P01 in the overlap, but E1 and E2 each duplicate.

Result: **8 rows**. `len(merged) > len(enc)` — you double-count admits. Deduplicate `patients` on `patient_id` before the join.

```
Fan-out (P01 twice on the right)

   Venn keys: still P01 in the overlap
   rows: E1×2 + E2×2 + E3 + E4 + E5 + E6 = 8
```

On the **seeded** Northshore bake, encounters are sampled from the generated patient table, so the three counts are often **equal**. The demo is the habit. In production they diverge.

**Say this:** Venn is keys; `merge row counts` is rows. Left = left circle (nulls for P99). Inner = overlap (drops P99). If left `n` grows, the patient key wasn’t unique.

## Agent workflow

```
plan(state) → tool → observation → update(state) → stop
tools: get_chart | flag_labs | score_admit | nearest_condition | retrieve_protocol
```

No LLM is required to learn the architecture. Swap `plan()` for a model that emits `{tool, args}` JSON later if you want.

---

## Terminology

### Clinic setting

| Term | Meaning in this lab |
|---|---|
| **Northshore Urgent Care** | Fictional five-site network. All arrays and frames are generated from one seed. |
| **Site** | Clinic location: Downtown, Airport, Campus, Harbor, Suburb. |
| **Encounter** | One visit (one row). Not a lifetime patient record. |
| **Chart** | The encounter + demographics + comorbidities the agent reads first (`get_chart`). |
| **Atlas** | Eight-condition menu: typical labs, admit base rate, protocol weights, panel cost. |
| **Census** | Count of encounters per site-day. The regression target. |
| **Arrival** | How they got there: `walk_in`, `ambulance`, `referral`. |
| **Payer** | `self_pay`, `medicaid`, `commercial`, `medicare`. Used in merges / clustering, not as a diagnosis. |
| **Season** | `typical`, `flu_wave`, `heat`. A calendar/surge flag (known at dawn only if you have a forecast). |
| **Disposition** | Suggested next place: `likely_discharge` vs `consider_admit`. A hint, not an order. |
| **LOS (`los_h`)** | Length of stay in hours (synthetic). Longer if admitted. |
| **Wait (`wait_min`)** | Door-to-provider minutes. ESI 1–2 are generated shorter. |
| **Reagent / assay** | Lab test supply. Inventory is a NumPy matrix (sites × biomarkers). |
| **Panel cost** | `protocol @ assay_cost` — indicated tests times unit cost. |

### Triage and acuity

| Term | Meaning in this lab |
|---|---|
| **Triage** | First sort: who is seen sooner. Not a diagnosis. |
| **ESI** | **Emergency Severity Index**, 1–5. 1 = resuscitation now; 2 = high risk, don’t wait; 3 = several resources; 4 = one resource; 5 = none. Feature `esi` / `esi_n`. |
| **Late triage** | ESI ≤ 2 and wait > 20 minutes. The “SLA miss” analogue. |
| **Hypoxia** | SpO2 &lt; 92% in this lab. A red flag, not a disease name. |
| **Rush hour** | Hours 10–11 and 17–19. Extra wait / a binary feature `rush`. |
| **Fast track** | Low-ESI bucket in the Python control-flow lab. |
| **Bed budget** | Capacity constraint used to pick a classification **threshold** (false-positive admits cost beds). |

### Conditions (labels in the atlas)

| Code | Plain language |
|---|---|
| **URI** | Viral upper respiratory infection (cold-like). |
| **Influenza** | Flu. |
| **Pneumonia** | Lung infection; higher admit base; CXR infiltrate. |
| **UTI** | Urinary tract infection; urine nitrite. |
| **Gastroenteritis** | Vomiting / diarrhea / dehydration labs. |
| **Migraine** | Headache syndrome; labs mostly normal. |
| **Cellulitis** | Skin infection. |
| **Cardiac_rule_out** | Chest-pain workup; troponin, not a confirmed MI label. |

### Vitals, labs, comorbidities

| Term | Meaning in this lab |
|---|---|
| **Biomarker** | One column in the lab vector (14 of them). |
| **Reference range** | `ref_low` / `ref_high`. Outside the band → NumPy mask `abnormal`. |
| **Z vs band** | `(x - low) / (high - low)` after broadcasting. |
| **WBC** | White blood cell count. |
| **CRP** | C-reactive protein (inflammation). |
| **Lactate** | Perfusion / severity marker; often **missing** on purpose (Pandas imputation lesson). |
| **SpO2** | Oxygen saturation (%). |
| **HR** | Heart rate. |
| **Temp_c** | Temperature °C. |
| **SBP** | Systolic blood pressure. |
| **Glucose** | Blood sugar. |
| **Creatinine** | Kidney marker. |
| **Sodium / potassium** | Electrolytes. |
| **Troponin** | Cardiac injury marker; also planted missing. |
| **UA nitrite** | Urinalysis nitrite (UTI signal). |
| **CXR infiltrate** | Chest X-ray infiltrate score 0–1 (synthetic). |
| **HTN / DM / COPD / CAD** | Hypertension, diabetes, chronic lung disease, coronary disease. Comorbidity flags on the patient. |
| **Comorbid** | Count of those four flags. Used in clustering. |
| **Red flag** | Hard stop for the agent: e.g. SpO2 &lt; 92, SBP &lt; 90, lactate &gt; 2, troponin above URL. |

### Protocol and agent

| Term | Meaning in this lab |
|---|---|
| **Protocol matrix** | Shape `(8 conditions × 14 biomarkers)`. Entries in `[0, 1]` = how indicated that assay is. |
| **Protocol (retrieved)** | Next tests, red flags, disposition hint for a guessed condition. |
| **Tool** | A function with a name + description the planner may call. |
| **Planner / `plan(state)`** | Policy that picks the next tool. Here it is deterministic (curriculum). An LLM would emit `{tool, args}`. |
| **Observation** | Tool return value written into **state**. |
| **Trace** | Table of thought → tool → observation (the audit log). |
| **Stop** | End the loop and write a **suggested** plan for a clinician. |
| **`get_chart`** | Load demographics, ESI, arrival, symptoms, comorbidities. Strips gold labels. |
| **`flag_labs`** | Compare the lab vector to reference ranges (NumPy). |
| **`score_admit`** | `P(admit)` from the sklearn Pipeline. |
| **`nearest_condition`** | L2 nearest row in `lab_means` (prototype retrieval, not the gold label). |
| **`retrieve_protocol`** | Lookup next tests / red flags for that guess. |
| **Gold label** | True synthetic `condition` / `admit`. Hidden from the planner; used only in eval. |
| **Leakage dressed as RAG** | Letting the planner see the gold condition. Same sin as target encoding on the full table. |

### ML / stats (as used in the pages)

| Term | Meaning in this lab |
|---|---|
| **Feature** | A column the model may see at **prediction time**. |
| **Target** | What you predict: census (`encounters`), or `admit` (0/1). |
| **Time split** | Train on earlier dates, test on later. Default for census and admit. |
| **Random split** | Shuffled rows. Inflates R² on time series (leak tomorrow into today). |
| **Leakage** | Test information in train: same-day `admit_rate`, full-frame imputers, ID one-hots, gold labels in the agent. |
| **Lag-7 (`enc_lag7`)** | Same weekday last week, per site. The **naive** census forecast. |
| **Roll-7 (`enc_roll7`)** | Mean of the **past** 7 days (`shift(1).rolling(7)`). No future. |
| **Naive baseline** | Predict lag-7. If Ridge loses to this, you do not have a census model. |
| **One-hot** | Categorical → binary columns (`OneHotEncoder`). Same explosion as `itertools.product`. |
| **Target encoding** | Mean of the label by category. Illegal if fit on the full table including the row. |
| **Pipeline** | `prep` + `model` in one object. Scaler/imputer fit on **train only**. What you `joblib.dump`. |
| **ColumnTransformer** | Numeric vs categorical branches inside the Pipeline. |
| **StandardScaler** | Zero-mean unit-variance. Required for k-means / linear models; trees don’t need it. |
| **Imputer** | Fill missing lactate/troponin from **train** medians (`SimpleImputer`). |
| **Class weight** | `balanced` vs none. Admit is the class you care about. |
| **Threshold** | Cut on `P(admit)`. Product decision (beds), not a magical 0.5. |
| **Accuracy** | Share correct. Misleading when most visits discharge. |
| **Precision** | Of predicted admits, how many truly admit. |
| **Recall** | Of true admits, how many you caught. |
| **F1** | Harmonic mean of precision and recall. |
| **ROC AUC** | Ranking quality across thresholds. |
| **PR curve** | Precision vs recall — more honest for the minority admit class. |
| **Confusion matrix** | Counts of discharge/admit × predicted. |
| **MAE / RMSE / R²** | Regression errors for census. Always vs lag-7. |
| **Residual** | Actual − predicted. Plot vs predicted and **by site**. |
| **TimeSeriesSplit** | Walk-forward CV. Do not shuffle Fridays into train and test. |
| **KMeans** | Euclidean clusters on scaled patient features. |
| **Silhouette** | Cluster tightness hint, not a clinical KPI. |
| **PCA** | 2D view of scaled patients. Visualization, not a diagnosis. |
| **Phenotype / segment** | Unsupervised group (e.g. older high-util). Name it in English or `k` is wrong. |
| **High cardinality ID** | `patient_id` as one-hot. Overfits identity; same bug as stuffing MRN into a prompt. |

### Data files

| Path | What it is |
|---|---|
| `clinic/data.py` | Generator (seed 42). |
| `clinic/ml.py` | Feature frames + transformers. |
| `clinic/agent.py` | Tools, planner, trace. |
| `data/*.parquet`, `data/matrices.npz` | Pre-baked clinic so Cloud boots without synthesizing rows. |

## Deploy

Hosted at [https://agenticdoctorv2.streamlit.app/](https://agenticdoctorv2.streamlit.app/) from `main` / `app.py`. Updates on every `git push`.

If Community Cloud is on Python 3.14:

- Set **Python 3.12** in the app’s **Advanced settings**, then **Reboot**. `runtime.txt` is ignored.
- Do not pin old NumPy/PyArrow (no 3.14 wheels → oven hang compiling from source).
- `requirements.txt` needs `streamlit>=1.61.1` (1.61.0 + Starlette on 3.14 crashes before the app starts).
- Logs that stop at “Spinning up manager process…” are still installing the environment; wait for pip, or reboot after the Python version change.
