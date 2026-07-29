"""
train_model.py

Trains a Kp-index forecasting model on the NOAA observed Kp history pulled
by data/fetch_noaa_data.py. Phase 2: logs the run to Azure ML via MLflow
(experiment tracking) and registers the model in the Azure ML model
registry, in addition to saving a local .pkl for the dashboard to use.

Requires: pip install azure-ai-ml azure-identity mlflow azureml-mlflow
Requires: az login (uses your CLI session for auth via DefaultAzureCredential)

Run (after fetch_noaa_data.py has been run at least once):
    python train_model.py
"""

import os

import joblib
import mlflow
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "kp_index.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "kp_forecast_model.pkl")

N_LAGS = 4  # how many past 3-hour Kp readings to use as features

# Azure ML workspace details (from Phase 2 Terraform output)
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "c93cca8a-a32b-4063-be80-5637cea65027")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "rg-auroraai-dev")
WORKSPACE_NAME = os.getenv("AZURE_ML_WORKSPACE", "mlw-auroraai-djarudi")

EXPERIMENT_NAME = "aurora-kp-forecast"
REGISTERED_MODEL_NAME = "kp-forecast-xgboost"


def load_data() -> pd.DataFrame:
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"{RAW_PATH} not found. Run `python data/fetch_noaa_data.py` first."
        )
    df = pd.read_csv(RAW_PATH)

    # The NOAA feed's header row becomes the CSV header via fetch_noaa_data.py's
    # generic parser, but column names vary slightly across NOAA feed versions
    # (e.g. "Kp" vs "kp_index"). Normalize defensively.
    kp_col = next((c for c in df.columns if c.lower() in ("kp", "kp_index")), None)
    time_col = next((c for c in df.columns if "time" in c.lower()), None)
    if kp_col is None or time_col is None:
        raise ValueError(
            f"Couldn't find Kp/time columns in {RAW_PATH}. Columns found: {list(df.columns)}"
        )

    df = df[[time_col, kp_col]].rename(columns={time_col: "time_tag", kp_col: "kp"})
    df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce")
    df["kp"] = pd.to_numeric(df["kp"], errors="coerce")
    df = df.dropna().sort_values("time_tag").reset_index(drop=True)
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    for lag in range(1, N_LAGS + 1):
        feat[f"kp_lag{lag}"] = feat["kp"].shift(lag)
    feat["kp_roll_mean3"] = feat["kp"].shift(1).rolling(3).mean()
    feat["target"] = feat["kp"].shift(-1)  # predict the *next* reading
    feat = feat.dropna().reset_index(drop=True)
    return feat


def connect_mlflow_to_azure_ml():
    """Point mlflow at the Azure ML workspace's tracking server. Falls back
    to local-only tracking (no Azure logging) if the workspace can't be
    reached, so training still works even if you're offline or auth fails."""
    try:
        ml_client = MLClient(
            DefaultAzureCredential(), SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME
        )
        workspace = ml_client.workspaces.get(WORKSPACE_NAME)
        mlflow.set_tracking_uri(workspace.mlflow_tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        print(f"Connected to Azure ML workspace '{WORKSPACE_NAME}' for tracking.")
        return True
    except Exception as e:
        print(f"[warning] Couldn't connect to Azure ML ({e}). Training locally only.")
        return False


def main():
    connected = connect_mlflow_to_azure_ml()

    df = load_data()
    print(f"Loaded {len(df)} Kp readings from {df['time_tag'].min()} to {df['time_tag'].max()}")

    feat = make_features(df)
    feature_cols = [c for c in feat.columns if c.startswith("kp_lag") or c == "kp_roll_mean3"]
    X, y = feat[feature_cols], feat["target"]

    # Time series: don't shuffle, keep the split chronological
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    params = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05}

    run_ctx = mlflow.start_run() if connected else None
    try:
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"Test MAE: {mae:.3f} Kp units (naive persistence baseline is usually ~0.5-0.8)")

        if connected:
            mlflow.log_params(params)
            mlflow.log_param("n_lags", N_LAGS)
            mlflow.log_param("training_rows", len(X_train))
            mlflow.log_metric("mae", mae)
            # Note: we deliberately don't use mlflow.xgboost.log_model() or
            # mlflow.register_model() here - as of mid-2026, MLflow's client
            # (2.8+) routes all model logging through a newer "Logged Models"
            # API that Azure ML's tracking server doesn't support yet
            # (returns 404), and older MLflow versions won't build on
            # Python 3.13 (pyarrow has no compatible wheel). Registering the
            # model via the Azure ML SDK directly sidesteps this entirely -
            # it talks to a different, unaffected API.
    finally:
        if run_ctx is not None:
            mlflow.end_run()

    # Always keep the local .pkl too - the dashboard reads this directly,
    # no need to hit Azure ML on every dashboard refresh.
    joblib.dump({"model": model, "feature_cols": feature_cols, "n_lags": N_LAGS}, MODEL_PATH)
    print(f"Saved local model -> {MODEL_PATH}")

    if connected:
        register_model_in_azure_ml(mae)


def register_model_in_azure_ml(mae: float):
    """Register the local .pkl as a versioned Model asset in the Azure ML
    model registry, via the SDK directly (bypasses MLflow's model-logging
    path, which has a known compatibility gap with Azure ML - see note above)."""
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model

    ml_client = MLClient(
        DefaultAzureCredential(), SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME
    )
    model_entity = Model(
        path=MODEL_PATH,
        name=REGISTERED_MODEL_NAME,
        type=AssetTypes.CUSTOM_MODEL,
        description="XGBoost Kp-index forecaster, lag-feature based.",
        properties={"mae": str(mae), "n_lags": str(N_LAGS)},
    )
    registered = ml_client.models.create_or_update(model_entity)
    print(f"Registered model '{registered.name}' version {registered.version} in Azure ML.")


if __name__ == "__main__":
    main()