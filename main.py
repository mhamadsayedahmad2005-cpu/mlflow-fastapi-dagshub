import os
from contextlib import asynccontextmanager

import mlflow.sklearn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "/opt/artifacts/model")
N_FEATURES = 64  # sklearn "digits": 8x8 pixel images, flattened
_state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    _state["model"] = mlflow.sklearn.load_model(MODEL_PATH)
    yield


# Docs/OpenAPI routes are disabled so /predict is the only endpoint.
app = FastAPI(title="Digits classifier", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=N_FEATURES, max_length=N_FEATURES)


class PredictResponse(BaseModel):
    prediction: int


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    x = np.asarray(req.features, dtype=float).reshape(1, -1)
    return PredictResponse(prediction=int(_state["model"].predict(x)[0]))
