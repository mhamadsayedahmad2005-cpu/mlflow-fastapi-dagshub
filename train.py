"""Train two RandomForest models and log them to MLflow on DagsHub.

Run locally (NOT in CI):
    pip install -r requirements-train.txt
    export DAGSHUB_REPO_OWNER=<owner> DAGSHUB_REPO_NAME=<repo>
    python train.py
"""
import os

import dagshub
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "RandomForest_V1.0.1")

dagshub.init(
    repo_owner=os.environ["DAGSHUB_REPO_OWNER"],
    repo_name=os.environ["DAGSHUB_REPO_NAME"],
    mlflow=True,
)
mlflow.set_experiment(EXPERIMENT_NAME)

data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)


def train_and_log_model(n_estimators: int, max_depth: int, run_name: str) -> None:
    with mlflow.start_run(run_name=run_name):
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        # Fixed artifact path so the Docker build always knows where to find it.
        mlflow.sklearn.log_model(model, artifact_path="model")
        print(f"{run_name}: n_estimators={n_estimators} max_depth={max_depth} accuracy={accuracy:.4f}")


train_and_log_model(20, 5, "random_forest_model_run_1")
train_and_log_model(100, 10, "random_forest_model_run_2")
