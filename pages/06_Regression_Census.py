import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from clinic.ml import daily_model_frame, daily_transformer, time_split
from clinic.quiz import ask
from clinic.ui import header, inject, load_clinic, ok, pitfall, warn, why

inject()
clinic = load_clinic()
feat = daily_model_frame(clinic)

header(
    "Module 06 · Regression",
    "How many encounters will each site print tomorrow?",
    "Linear baseline. Ridge when columns collude. Forests when flu × suburb is real. Beat lag-7 naive.",
)
warn("Synthetic census. Not a staffing product.")
why("Census is a regression with a time axis. You ship the model that beats last week's Tuesday, not the prettiest training curve.")

y_col = "encounters"
cols = ["site", "season", "dow", "is_weekend", "flu_wave", "is_heat", "enc_lag7", "enc_roll7"]
tr, te = time_split(feat, "date", 0.75)
X_train, X_test = feat.loc[tr, cols], feat.loc[te, cols]
y_train, y_test = feat.loc[tr, y_col], feat.loc[te, y_col]
naive = X_test["enc_lag7"].to_numpy()

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
    return {"model": label, "MAE": mean_absolute_error(y, p), "RMSE": root_mean_squared_error(y, p), "R²": r2_score(y, p)}


st.dataframe(pd.DataFrame([metrics(y_test, naive, "naive lag-7"), metrics(y_test, pred, model_name)]).round(3), hide_index=True)

plot_df = pd.DataFrame(
    {
        "date": feat.loc[te, "date"].to_numpy(),
        "site": feat.loc[te, "site"].astype(str).to_numpy(),
        "actual": y_test.to_numpy(),
        "pred": pred,
        "naive": naive,
    }
)
site = st.selectbox("Chart site", clinic.sites)
one = plot_df[plot_df.site == site].sort_values("date")
st.line_chart(one.set_index("date")[["actual", "pred", "naive"]])

resid = plot_df.copy()
resid["residual"] = resid["actual"] - resid["pred"]
fig = px.scatter(resid, x="pred", y="residual", color="site", title="Residual vs predicted")
fig.add_hline(y=0, line_dash="dash")
fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, width="stretch")

names = pipe.named_steps["prep"].get_feature_names_out()
if model_name == "RandomForest":
    st.bar_chart(pd.Series(pipe.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False).head(12), color="#2A9D8F")
    pitfall("Impurity importance inflates correlated lags. Debug with it; don't tell the COO a causal story.")
else:
    st.bar_chart(pd.Series(pipe.named_steps["model"].coef_, index=names).sort_values(), color="#7DDECF")

ok("Always print the naive baseline on the same time-split window. If Ridge loses to lag-7, you don't have a census model.")
ask(
    "q6_reg",
    "Test MAE is 9 encounters. Downtown averages ~90/day, Suburb ~40. What's missing?",
    [
        "Nothing — MAE is scale-free",
        "A relative metric and residuals by site",
        "Switch to classification",
    ],
    "A relative metric and residuals by site",
    "Absolute MAE hides that Suburb is worse in relative terms. Stratify. That's the operations conversation.",
)
