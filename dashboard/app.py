"""
app.py

Phase 5 dashboard: the full AuroraAI pipeline in one page.
- Kp history chart (from live NOAA data)
- Forecast pulled from the Azure ML model registry (Phase 2)
- Full multi-agent explanation via the LangGraph Supervisor - Research +
  Forecasting + Reasoning + Guardrail (Phase 3/4)
- Live aurora forecast image + Vision Agent analysis (Phase 5 CV)

Run:
    streamlit run dashboard/app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agent"))
from supervisor import run as run_supervisor  # noqa: E402
from vision_agent import analyze_aurora_image, fetch_ovation_image, OVATION_IMAGE_URL  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

st.set_page_config(page_title="AuroraAI", page_icon="🌌", layout="centered")
st.title("🌌 AuroraAI")
st.caption("Live NOAA data → Azure ML forecast → multi-agent reasoning → aurora image analysis")


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


@st.cache_data(ttl=600)
def get_vision_analysis():
    image_bytes = fetch_ovation_image()
    analysis = analyze_aurora_image(image_bytes)
    return image_bytes, analysis


# --- Section 1: Kp history ---
try:
    history = load_history()
    st.subheader("Recent Kp Index")
    st.line_chart(history.set_index("time_tag")["kp"].tail(80))
except FileNotFoundError:
    st.error("No data found. Run `python data/fetch_noaa_data.py` first.")
    st.stop()

# --- Section 2: Multi-agent forecast + explanation ---
st.subheader("Forecast & Multi-Agent Analysis")
with st.spinner("Running Research → Forecast → Reasoning → Guardrail pipeline..."):
    result = run_supervisor("geomagnetic storm aurora forecast")

col1, col2 = st.columns(2)
with col1:
    st.metric("Predicted Kp Index (next 3h)", f"{result['predicted_kp']:.2f}")
with col2:
    flags = result.get("guardrail_notes", [])
    st.metric("Guardrail Status", "⚠️ Flagged" if flags else "✅ Pass")

st.write(result["explanation"])

if flags:
    with st.expander("Guardrail flags"):
        for f in flags:
            st.warning(f)

with st.expander(f"Research sources ({len(result['research_results'])} bulletins retrieved)"):
    for r in result["research_results"]:
        st.text(f"[{r['published_at']}]\n{r['content'][:300]}")

# --- Section 3: Computer vision - live aurora forecast image ---
st.subheader("Live Aurora Forecast Image (NOAA OVATION Model)")
try:
    with st.spinner("Fetching live aurora image and running vision analysis..."):
        image_bytes, vision_analysis = get_vision_analysis()
    st.image(image_bytes, caption=f"Source: {OVATION_IMAGE_URL}", use_container_width=True)
    st.write("**Vision Agent analysis:**")
    st.write(vision_analysis)
except Exception as e:
    st.warning(f"Couldn't load aurora image analysis: {e}")