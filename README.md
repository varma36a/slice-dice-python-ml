# Slice & Dice — Python for ML

End-to-end **Python for machine learning** lab, taught as a five-store pizzeria.

**Live app:** [https://agenticdoctor.streamlit.app/](https://agenticdoctor.streamlit.app/)  
**Repo:** [github.com/varma36a/slice-dice-python-ml](https://github.com/varma36a/slice-dice-python-ml)

Aimed at engineers who already write production code. No `hello world`. NumPy, Pandas, EDA, leakage-safe features, and sklearn — then one capstone that actually runs the shop.

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
| Home | See the shop universe (orders, stores, menu) |
| Python for ML | Records, counters, generators, itertools crosses |
| NumPy kitchen | Recipe matrix, broadcasting, masks, views vs copies |
| Pandas ledger | groupby/transform, merge, missing delivery times, rolling |
| EDA & charts | Find rush hour, rain, Campus-weekend — then encode them |
| Feature engineering | Time split vs random split vs illegal same-day revenue |
| Regression | Beat lag-7 naive demand |
| Classification | Late SLA, threshold as an ops budget |
| Clusters & pipelines | RFM segments, `Pipeline` + `TimeSeriesSplit` |
| Capstone | Friday-night board: forecast, restock, late risk, guests |

All pages share one seeded generator (`pizza/data.py`). Change a recipe, and the commissary math and the models still agree.

## Deploy (Streamlit Community Cloud)

Hosted at [https://agenticdoctor.streamlit.app/](https://agenticdoctor.streamlit.app/) from `main` / `app.py`. The app stays up independently of this laptop and updates on every `git push`.

## Design notes

- **Time-based splits** for demand and late-delivery (no shuffled Fridays).
- **Pipelines** so scalers never see the test night.
- **Naive lag-7** is the baseline a GM already has in their head.
- Inventory decisions are `counts @ recipes` — if you can shape that, sklearn is easy.
