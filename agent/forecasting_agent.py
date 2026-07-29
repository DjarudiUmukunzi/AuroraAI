"""
forecasting_agent.py

Forecasting Agent: pulls the registered model from the Azure ML model
registry (falls back to the local .pkl if that's unavailable) and
predicts the next Kp-index reading from recent history.

Run directly for a quick test:
    python forecasting_agent.py
"""

import os

import joblib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "kp_forecast_model.pkl")

SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "c93cca8a-a32b-4063-be80-5637cea65027")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "rg-auroraai-dev")
WORKSPACE_NAME = os.getenv("AZURE_ML_WORKSPACE", "mlw-auroraai-djarudi")
REGISTERED_MODEL_NAME = "kp-forecast-xgboost"


def _load_model_bundle() -> dict:
    """Try the Azure ML model registry first (the 'real' path for Phase 3);
    fall back to the local .pkl if that's unavailable so this agent still
    works offline or if Azure ML auth fails."""
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        ml_client = MLClient(
            DefaultAzureCredential(), SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME
        )
        latest = ml_client.models.get(name=REGISTERED_MODEL_NAME, label="latest")
        download_dir = os.path.join(os.path.dirname(__file__), "_downloaded_models")
        ml_client.models.download(
            name=REGISTERED_MODEL_NAME,
            version=latest.version,
            download_path=download_dir,
        )
        # Newer azure-ai-ml versions don't return the path from .download(),
        # so walk the directory we told it to download into instead.
        for root, _, files in os.walk(download_dir):
            for f in files:
                if f.endswith(".pkl"):
                    print(f"Loaded model '{REGISTERED_MODEL_NAME}' v{latest.version} from Azure ML registry.")
                    return joblib.load(os.path.join(root, f))
        raise FileNotFoundError("No .pkl found in downloaded model artifact.")
    except Exception as e:
        print(f"[info] Falling back to local model ({e})")
        return joblib.load(LOCAL_MODEL_PATH)


def forecast_next_kp() -> float:
    bundle = _load_model_bundle()
    model, feature_cols, n_lags = bundle["model"], bundle["feature_cols"], bundle["n_lags"]

    df = pd.read_csv(os.path.join(DATA_DIR, "kp_index.csv"))
    kp_col = next(c for c in df.columns if c.lower() in ("kp", "kp_index"))
    time_col = next(c for c in df.columns if "time" in c.lower())
    df = df[[time_col, kp_col]].rename(columns={time_col: "time_tag", kp_col: "kp"})
    df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce")
    df["kp"] = pd.to_numeric(df["kp"], errors="coerce")
    df = df.dropna().sort_values("time_tag")

    recent = df["kp"].tail(n_lags + 3).tolist()
    lags = {f"kp_lag{i}": recent[-i] for i in range(1, n_lags + 1)}
    lags["kp_roll_mean3"] = sum(recent[-3:]) / 3

    X_pred = pd.DataFrame([{c: lags[c] for c in feature_cols}])
    return float(model.predict(X_pred)[0])


if __name__ == "__main__":
    predicted = forecast_next_kp()
    print(f"Predicted next Kp index: {predicted:.2f}")