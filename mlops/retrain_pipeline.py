"""
retrain_pipeline.py

Phase 4: a real Azure ML cloud training Job (not just the local CI job
from Phase 4's first pass) that fetches fresh NOAA data and retrains +
registers the Kp forecasting model. Runs on Azure ML's serverless
compute, which auto-selects an available VM size based on your
subscription's actual quota - this sidesteps the regional compute-quota
walls we hit earlier in this project (Databricks classic clusters, GPU
family quota, etc.) rather than requiring us to manually guess a VM type
that might be unavailable.

KNOWN ISSUE (as of azure-ai-ml 1.34.1, mid-2026): job submission via this
script currently fails during code-asset upload with:
    "Uri https://<account>.blob.core.windows.net:443/ does not have any
    container reference."
Root cause is confirmed to be inside the SDK's internal code-asset
resolution path, not this project's configuration - verified directly:
    ml_client.datastores.get_default() correctly returns the real
    container name ('azureml-blobstore-<id>'), proving the workspace,
    storage account, and datastore are all configured correctly. The
    bug is in how a *different* internal code path (_get_code_asset_arm_id)
    builds the upload URI for anonymous code assets during job submission.
Also ruled out during debugging: the ML workspace's Key Vault was found
to be stuck in legacy "Access Policy" mode instead of RBAC mode (fixed
via `az keyvault update --enable-rbac-authorization true`) - a genuine
bug, but unrelated to this specific error, which persisted after the fix.

Workaround options not yet attempted: submitting via the Azure CLI
(`az ml job create`) instead of the Python SDK, which uses a different
code path; or filing an Azure SDK GitHub issue. This script is left in
place as-is since the approach and environment are otherwise correct -
only the job submission step is currently blocked by this SDK bug.

Run:
    python mlops/retrain_pipeline.py
"""

import os

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "c93cca8a-a32b-4063-be80-5637cea65027")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "rg-auroraai-dev")
WORKSPACE_NAME = os.getenv("AZURE_ML_WORKSPACE", "mlw-auroraai-djarudi")

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CONDA_FILE = os.path.join(os.path.dirname(__file__), "conda_env.yml")


def main():
    ml_client = MLClient(DefaultAzureCredential(), SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)

    env = Environment(
        name="aurora-retrain-env",
        description="Environment for the AuroraAI retraining job",
        conda_file=CONDA_FILE,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
    )

    fetch_and_train_job = command(
        display_name="Fetch NOAA data and retrain Kp forecaster",
        description="Pulls live NOAA Kp data and retrains/registers the XGBoost forecaster.",
        code=PROJECT_ROOT,  # uploads the project (minus .amlignore exclusions) to the job
        command="python data/fetch_noaa_data.py && python models/train_model.py",
        environment=env,
        # Omitting 'compute' runs this on serverless compute - auto-selects
        # an available VM size from actual subscription quota.
        experiment_name="aurora-retrain",
    )

    submitted = ml_client.jobs.create_or_update(fetch_and_train_job)
    print(f"Job submitted: {submitted.name}")
    print(f"Status: {submitted.status}")
    print(f"View in Azure ML Studio: {submitted.studio_url}")


if __name__ == "__main__":
    main()