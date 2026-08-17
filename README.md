# Agentic Doctor — Python for ML

End-to-end **Python for machine learning + agent workflow**, taught as a five-site urgent care.

**Live app:** [https://agenticdoctorv2.streamlit.app/](https://agenticdoctorv2.streamlit.app/)  
**Repo:** [github.com/varma36a/slice-dice-python-ml](https://github.com/varma36a/slice-dice-python-ml)

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
| Home | See the clinic universe (encounters, sites, atlas) |
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

All pages share one seeded generator (`clinic/data.py`). The admit Pipeline the classifier page fits is the same shape of object the agent calls as `score_admit`.

## Agent workflow (the point of the rewrite)

```
plan(state) → tool → observation → update(state) → stop
tools: get_chart | flag_labs | score_admit | nearest_condition | retrieve_protocol
```

No LLM is required to learn the architecture. Swap `plan()` for a model that emits `{tool, args}` JSON later if you want.

## Deploy

Hosted at [https://agenticdoctor.streamlit.app/](https://agenticdoctor.streamlit.app/) from `main` / `app.py`. Updates on every `git push`.
