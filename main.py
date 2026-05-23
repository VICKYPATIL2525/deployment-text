"""
Mindspace Text Sentiment API — FastAPI inference service.

Loads a pre-trained classifier and its full preprocessing pipeline
from disk at startup, then serves single-sample mental-health predictions.

The request schema (PredictRequest) is built dynamically at startup from
feature_names.json, so it always matches whatever model is currently loaded.
To swap models: update ARTIFACTS_DIR (line 83) and restart.

Endpoints
---------
GET  /           — service info           (requires X-API-Key)
GET  /health     — liveness/readiness probe (no auth required)
POST /predict    — single-sample prediction (requires X-API-Key)
GET  /model/info — model metadata          (requires X-API-Key)

Authentication
--------------
All endpoints except ``/health`` require an ``X-API-Key`` header whose value
must match the ``MINDSPACE_TEXT_API_KEY`` environment variable.

Environment variables
---------------------
MINDSPACE_TEXT_API_KEY : (required) shared secret key for API access.
                         Set in a ``.env`` file or as a real env var.
"""

import json
import logging
import os
import sys
import traceback
import uuid
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, create_model, field_validator

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ─── API Key auth ─────────────────────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_api_key() -> str:
    return os.environ.get("MINDSPACE_TEXT_API_KEY", "")


def verify_api_key(key: str = Security(_api_key_header)) -> None:
    api_key = _get_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail={"error": "server_misconfiguration", "message": "MINDSPACE_TEXT_API_KEY is not set."})
    if key != api_key:
        raise HTTPException(status_code=403, detail={"error": "invalid_api_key", "message": "Invalid or missing API key. Pass it as X-API-Key header."})


# ─── Artifact paths ───────────────────────────────────────────────────────────
# To switch models after retraining: update the folder name on the line below.
ARTIFACTS_DIR = Path(__file__).parent / "pipeline_output" / "Extra_Trees_18-May-2026_12-11-08"

artifacts: dict = {}


def build_predict_request_model(feature_names: list[str]) -> type[BaseModel]:
    """
    Dynamically build a Pydantic request model from the loaded feature list.

    All features are typed as ``float`` with a finite-value validator.
    This means swapping models (and their feature sets) requires no schema changes.
    """
    def _validate_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError(f"Expected a finite number, got {v}")
        return v

    field_definitions: dict[str, Any] = {name: (float, ...) for name in feature_names}

    validators = {"validate_all_finite": field_validator(*feature_names, mode="before")(classmethod(_validate_finite))}
 
    return create_model("PredictRequest", **field_definitions, __validators__=validators)


# Build the request model at module level so Swagger UI shows the schema
_feature_names_file = ARTIFACTS_DIR / "feature_names.json"
PredictRequest = build_predict_request_model(json.loads(_feature_names_file.read_text()))


def load_artifacts() -> None:
    """
    Load all model artifacts from ``ARTIFACTS_DIR`` into the global ``artifacts`` dict.

    Raises
    ------
    Any exception raised by ``joblib.load`` or file I/O is intentionally
    propagated so the caller (``lifespan``) can abort startup.
    """
    logger.info("Loading model artifacts from %s", ARTIFACTS_DIR)
    artifacts["model"]                = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    artifacts["scaler"]               = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    artifacts["label_encoder"]        = joblib.load(ARTIFACTS_DIR / "label_encoder.joblib")
    artifacts["encoding"]             = joblib.load(ARTIFACTS_DIR / "encoding_artifacts.joblib")
    artifacts["outlier_transformers"] = joblib.load(ARTIFACTS_DIR / "outlier_transformers.joblib")
    artifacts["feature_names"]        = json.loads((ARTIFACTS_DIR / "feature_names.json").read_text())
    artifacts["metadata"]             = json.loads((ARTIFACTS_DIR / "model_metadata.json").read_text())

    logger.info("All artifacts loaded — model: %s, features: %d, classes: %s",
                artifacts["metadata"].get("best_model_name"),
                artifacts["metadata"].get("n_features"),
                artifacts["metadata"].get("class_names"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_artifacts()
    except Exception as e:
        logger.critical("Failed to load model artifacts — %s\n%s", e, traceback.format_exc())
        sys.exit(1)

    if not _get_api_key():
        logger.critical("MINDSPACE_TEXT_API_KEY is not set in the environment.")
        sys.exit(1)

    logger.info("Startup complete — API is ready to serve requests.")
    yield
    logger.info("Shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mindspace Mental Health Classifier",
    description="Predicts mental health profile from speech/text features.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Response schema ──────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    prediction_id: str
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    model_name: str


# ─── Preprocessing ────────────────────────────────────────────────────────────

def apply_outlier_transforms_numpy(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """
    Apply per-feature outlier-handling strategies to a single-sample numpy array.

    Supported strategies: yeo-johnson, sqrt, log1p, winsorize.
    """
    transformers = artifacts["outlier_transformers"]
    X = X.copy()
    col_index = {name: i for i, name in enumerate(feature_names)}

    for col, info in transformers.items():
        if col not in col_index:
            continue
        i = col_index[col]
        strategy = info["strategy"]

        if strategy == "yeo-johnson":
            X[0, i] = info["fitted_pt"].transform(X[:, i].reshape(-1, 1)).ravel()[0]
        elif strategy == "sqrt":
            X[0, i] = np.sqrt(max(X[0, i], 0.0))
        elif strategy == "log1p":
            X[0, i] = np.log1p(max(X[0, i], 0.0))
        elif strategy == "winsorize":
            X[0, i] = min(max(X[0, i], info["lower"]), info["upper"])

    return X


def preprocess(raw: dict) -> pd.DataFrame:
    """Build and preprocess the model input from the raw request field dictionary."""
    feature_names = artifacts["feature_names"]
    X = np.array([[raw[f] for f in feature_names]], dtype=np.float64)
    X = apply_outlier_transforms_numpy(X, feature_names)
    scaled = artifacts["scaler"].transform(X)
    return pd.DataFrame(scaled, columns=feature_names)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root(_: None = Security(verify_api_key)):
    return {
        "service": "Mindspace Mental Health Classifier",
        "status": "running",
        "classes": artifacts.get("metadata", {}).get("class_names"),
        "n_features": artifacts.get("metadata", {}).get("n_features"),
    }


@app.get("/health")
def health():
    expected_keys = {"model", "scaler", "label_encoder", "encoding", "outlier_transformers", "feature_names", "metadata"}
    ready = expected_keys.issubset(artifacts.keys())
    if not ready:
        return JSONResponse(status_code=503, content={"status": "unavailable", "artifacts_loaded": len(artifacts)})
    return {
        "status": "ok",
        "model": artifacts.get("metadata", {}).get("best_model_name"),
        "artifacts_loaded": len(artifacts),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, _: None = Security(verify_api_key)):  # type: ignore[valid-type]
    """
    Predict the mental health profile for a single text / speech sample.

    Accepts a JSON body whose fields match the feature names of the currently
    loaded model (auto-discovered from feature_names.json at startup).
    """
    try:
        raw = body.model_dump()
        X = preprocess(raw)
    except Exception as e:
        logger.error("Preprocessing failed — %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=422, detail={"error": "preprocessing_failed", "message": str(e)})

    try:
        model = artifacts["model"]
        le    = artifacts["label_encoder"]

        proba      = model.predict_proba(X)[0]
        pred_idx   = int(np.argmax(proba))
        pred_label = le.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])

        class_names   = le.classes_.tolist()
        probabilities = {cls: round(float(p), 4) for cls, p in zip(class_names, proba)}

        prediction_id = str(uuid.uuid4())

        logger.info("prediction_id=%s prediction=%s confidence=%.4f",
                    prediction_id, pred_label, confidence)

        return PredictResponse(
            prediction_id=prediction_id,
            prediction=pred_label,
            confidence=round(confidence, 4),
            probabilities=probabilities,
            model_name=artifacts.get("metadata", {}).get("best_model_name", "unknown"),
        )
    except Exception as e:
        logger.error("Prediction failed — %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail={"error": "prediction_failed", "message": str(e)})


@app.get("/model/info")
def model_info(_: None = Security(verify_api_key)):
    meta = artifacts.get("metadata", {})
    return {
        "model": meta.get("best_model_name"),
        "n_features": meta.get("n_features"),
        "feature_names": meta.get("feature_names"),
        "classes": meta.get("class_names"),
        "scaler": meta.get("scaler"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5500)
