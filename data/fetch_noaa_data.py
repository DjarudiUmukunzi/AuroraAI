"""
fetch_noaa_data.py

Pulls space weather data from NOAA SWPC's public JSON endpoints and saves
it locally as CSV. No auth required - these are public, unauthenticated
feeds. This intentionally skips Azure Data Factory / Databricks / Delta
Lake - it's a plain script so you have a working data source on day one.
You can swap the `save_local()` calls for blob uploads later once the
core pipeline works.

Run:
    python fetch_noaa_data.py
"""

import json
import os
from datetime import datetime, timezone

import requests
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(OUT_DIR, exist_ok=True)

ENDPOINTS = {
    # Observed planetary Kp index, 3-hour resolution, last ~30 days.
    # This is your primary label source for training the forecasting model.
    "kp_index": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    # 3-day Kp forecast (useful later for comparing your model vs NOAA's own forecast)
    "kp_forecast": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json",
    # Real-time solar wind magnetic field. NOAA deprecated the old
    # solar-wind/mag-1-day.json endpoint on 2026-04-30; this replaces it.
    "solar_wind_mag": "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json",
    # Real-time solar wind plasma (density/speed/temperature). Replaces the
    # deprecated solar-wind/plasma-1-day.json. Field names changed: use
    # proton_density / proton_speed / proton_temperature, not density/speed/temperature.
    "solar_wind_plasma": "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json",
    # Sunspot report
    "sunspot_report": "https://services.swpc.noaa.gov/json/sunspot_report.json",
    # Human-readable text alerts/warnings - this is your "bulletin" text for
    # the Research/Reasoning agent to summarize (RAG-lite: no indexing needed
    # yet, just pass the latest message straight into the prompt)
    "alerts": "https://services.swpc.noaa.gov/products/alerts.json",
}


def fetch_json(url: str) -> list:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def to_dataframe(name: str, raw: list) -> pd.DataFrame:
    """NOAA's *-1-day.json feeds return the header row as the first list
    element (e.g. ["time_tag","bx","by","bz","lon_gsm","lat_gsm","bt"]).
    Everything else is [time_tag, value, value, ...]."""
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        header, rows = raw[0], raw[1:]
        df = pd.DataFrame(rows, columns=header)
    else:
        df = pd.DataFrame(raw)
    df["_source"] = name
    df["_fetched_at"] = datetime.now(timezone.utc).isoformat()
    return df


def save_local(name: str, df: pd.DataFrame) -> str:
    path = os.path.join(OUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def main():
    for name, url in ENDPOINTS.items():
        try:
            raw = fetch_json(url)
            df = to_dataframe(name, raw)
            path = save_local(name, df)
            print(f"[ok]   {name:<18} -> {path} ({len(df)} rows)")
        except Exception as e:
            print(f"[fail] {name:<18} -> {e}")


if __name__ == "__main__":
    main()