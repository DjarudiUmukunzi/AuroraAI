"""
app.py

MVP version of Phase 5. One Streamlit page: Kp history chart, the model's
next-window forecast, and the agent's plain-language explanation. No
React/Next.js, no aurora webcam CV module - those come later.

Run:
    streamlit run dashboard/app.py
"""

import os
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from agent.explain_agent import explain_forecast  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "kp_forecast_model.pkl")

st.set_page_config(page_title="AuroraAI - MVP", page_icon="🌌", layout="centered")
st.title("🌌 AuroraAI — Kp Forecast (MVP)")
st.caption("Live NOAA data -> XGBoost forecast -> LLM explanation")


@st.cache_data(ttl=300)
def load_history() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "kp_index.csv")
    df = pd.read_csv(path)
    kp_col = next(c for c in df.columns if c.lower() in ("kp", "kp_index"))
    time_col = next(c for c in df.columns if "time" in c.lower())
    df = df[[time_col, kp_col]].rename(columns={time_col: "time_tag", kp_col: "kp"})
    df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce")
    df["kp"] = pd.to_numeric(df["kp"], errors="coerce")
    return df.dropna().sort_values("time_tag")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_cols"], bundle["n_lags"]


try:
    history = load_history()
except FileNotFoundError:
    st.error("No data found. Run `python data/fetch_noaa_data.py` first.")
    st.stop()

st.subheader("Recent Kp Index")
st.line_chart(history.set_index("time_tag")["kp"].tail(80))

try:
    model, feature_cols, n_lags = load_model()
    recent = history["kp"].tail(n_lags + 3).tolist()
    lags = {f"kp_lag{i}": recent[-i] for i in range(1, n_lags + 1)}
    lags["kp_roll_mean3"] = sum(recent[-3:]) / 3
    X_pred = pd.DataFrame([{c: lags[c] for c in feature_cols}])
    predicted_kp = float(model.predict(X_pred)[0])
except FileNotFoundError:
    st.warning("No trained model found. Run `python models/train_model.py` first.")
    predicted_kp = None

if predicted_kp is not None:
    st.subheader("Next 3-Hour Forecast")
    st.metric("Predicted Kp Index", f"{predicted_kp:.1f}")

    with st.spinner("Generating explanation..."):
        explanation = explain_forecast(predicted_kp)
    st.subheader("AI Explanation")
    st.write(explanation)
