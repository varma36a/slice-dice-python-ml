import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from pizza.ml import daily_model_frame, daily_transformer, time_split
from pizza.quiz import ask
from pizza.ui import header, inject, load_shop, ok, pitfall, why

inject()
shop = load_shop()
feat = daily_model_frame(shop)

header(
    "Module 06 · Regression",
    "How many tickets will each store print tomorrow?",
    "Linear as the baseline. Ridge when columns collude. Forests when interactions (rain × suburb) are real.",
)

why(
    "Demand is a regression problem with a time axis. You ship the model that beats a **lag-7 naive** forecast, "
    "not the one with the prettiest training curve."
)

y_col = "tickets"
num_cat = ["store", "weather", "dow", "is_weekend", "is_rain", "is_heat", "tickets_lag7", "tickets_roll7"]
tr, te = time_split(feat, "date", 0.75)
X_train, X_test = feat.loc[tr, num_cat], feat.loc[te, num_cat]
y_train, y_test = feat.loc[tr, y_col], feat.loc[te, y_col]

naive = X_test["tickets_lag7"].to_numpy()

model_name = st.selectbox("Estimator", ["LinearRegression", "Ridge", "RandomForest"])
if model_name == "LinearRegression":
    est = LinearRegression()
elif model_name == "Ridge":
    est = Ridge(alpha=st.slider("Ridge α", 0.1, 20.0, 2.0))
else:
    est = RandomForestRegressor(
        n_estimators=st.slider("Trees", 50, 400, 180, step=10),
        min_samples_leaf=st.slider("min_samples_leaf", 1, 15, 3),
        random_state=0,
        n_jobs=-1,
    )

pipe = Pipeline([("prep", daily_transformer()), ("model", est)])
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

def metrics(y, p, label):
    return {
        "model": label,
        "MAE": mean_absolute_error(y, p),
        "RMSE": root_mean_squared_error(y, p),
        "R²": r2_score(y, p),
    }

table = pd.DataFrame(
    [
        metrics(y_test, naive, "naive lag-7"),
        metrics(y_test, pred, model_name),
    ]
)
st.dataframe(table.round(3), hide_index=True)

plot_df = pd.DataFrame(
    {
        "date": feat.loc[te, "date"].to_numpy(),
        "store": feat.loc[te, "store"].astype(str).to_numpy(),
        "actual": y_test.to_numpy(),
        "pred": pred,
        "naive": naive,
    }
)
store = st.selectbox("Chart store", shop.stores)
one = plot_df[plot_df.store == store].sort_values("date")
st.line_chart(one.set_index("date")[["actual", "pred", "naive"]])

st.markdown("### Residuals — where the model is lying")
resid = plot_df.copy()
resid["residual"] = resid["actual"] - resid["pred"]
fig = px.scatter(resid, x="pred", y="residual", color="store", title="Residual vs predicted")
fig.add_hline(y=0, line_dash="dash")
fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, width="stretch")

if model_name == "RandomForest":
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    top = pd.Series(imp, index=names).sort_values(ascending=False).head(12)
    st.markdown("### Feature importance (impurity)")
    st.bar_chart(top, color="#E85D04")
    pitfall(
        "Impurity importance inflates high-cardinality and correlated lags. Use it to **debug**, not to tell finance a causal story."
    )
else:
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["model"].coef_
    s = pd.Series(coef, index=names).sort_values()
    st.markdown("### Coefficients (after scaling)")
    st.bar_chart(s, color="#FAA307")

ok(
    "Always print the naive baseline on the **same** time-split test window. "
    "If Ridge loses to lag-7, you don't have a demand model — you have a dashboard."
)

ask(
    "q6_reg",
    "Test MAE is 9 tickets. Downtown averages ~90 tickets/day, Suburb ~40. What's missing?",
    [
        "Nothing — MAE is scale-free",
        "A relative metric (MAPE / MAE by store) and a plot of residuals by store",
        "Switch to classification",
    ],
    "A relative metric (MAPE / MAE by store) and a plot of residuals by store",
    "Absolute MAE hides that Suburb is 2× worse in relative terms. Stratify. That's the shop conversation.",
)
