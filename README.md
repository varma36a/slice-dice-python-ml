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

The full catalog lives **on the site**: home (tabs), sidebar **All Topics** (example + interview card), and **Interview questions** (live demos, flashcards, MCQ). Source: `clinic/topics.py`, `clinic/interview.py`.

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

If Community Cloud is on Python 3.14, do not pin old NumPy/PyArrow — those have no 3.14 wheels and the app stays in the oven compiling from source. Optionally set **Python 3.12** in the app’s Advanced settings.
