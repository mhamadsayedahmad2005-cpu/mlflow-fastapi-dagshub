# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/opt/artifacts \
    MODEL_PATH=/opt/artifacts/model

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Fetch the latest trained model from DagsHub (via MLflow) at build time ---
ARG MLFLOW_TRACKING_URI
ARG MLFLOW_EXPERIMENT_NAME=RandomForest_V1.0.1
# Changing this value (CI passes the run id) invalidates the cache so that
# every build re-downloads the *latest* model instead of reusing an old layer.
ARG CACHE_BUST=0

COPY scripts/fetch_model.py scripts/fetch_model.py
# The DagsHub token is a BuildKit secret: it is never written to an image layer.
RUN --mount=type=secret,id=dagshub_token \
    echo "cache bust: ${CACHE_BUST}" && \
    MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
    MLFLOW_TRACKING_USERNAME="$(cat /run/secrets/dagshub_token)" \
    MLFLOW_TRACKING_PASSWORD="$(cat /run/secrets/dagshub_token)" \
    MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME}" \
    python scripts/fetch_model.py

COPY app app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
