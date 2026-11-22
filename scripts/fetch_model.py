"""Download the latest trained model from MLflow (DagsHub) at image build time.

Reads MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
from the environment (the DagsHub token is used for both username and password).
"""
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient

experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "RandomForest_V1.0.1")
dst = os.environ.get("MODEL_DIR", "/opt/artifacts")

client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    sys.exit(f"Experiment '{experiment_name}' not found on {mlflow.get_tracking_uri()}")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
if not runs:
    sys.exit(f"No finished runs in experiment '{experiment_name}'")

run = runs[0]
print(f"Latest run: {run.info.run_id} | metrics={run.data.metrics} | params={run.data.params}")
path = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="model", dst_path=dst)
print(f"Model downloaded to {path}")
