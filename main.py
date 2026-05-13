"""
Mindspace Text Sentiment API — FastAPI inference service.

Loads a pre-trained LightGBM classifier and its full preprocessing pipeline
from disk at startup, then serves single-sample mental-health predictions.

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

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, field_validator

# ─── Logging setup ────────────────────────────────────────────────────────────
# Writes structured lines to stdout so Docker / gunicorn capture them naturally.
# Format: timestamp | level | message
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
    """Return the expected API key from the ``MINDSPACE_TEXT_API_KEY`` environment variable."""
    return os.environ.get("MINDSPACE_TEXT_API_KEY", "")


def verify_api_key(key: str = Security(_api_key_header)) -> None:
    """
    FastAPI dependency that validates the ``X-API-Key`` request header.

    Raises
    ------
    HTTP 500 : if ``MINDSPACE_TEXT_API_KEY`` is not set on the server.
    HTTP 403 : if the supplied key is absent or does not match the server key.
    """
    api_key = _get_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail={"error": "server_misconfiguration", "message": "MINDSPACE_TEXT_API_KEY is not set."})
    if key != api_key:
        raise HTTPException(status_code=403, detail={"error": "invalid_api_key", "message": "Invalid or missing API key. Pass it as X-API-Key header."})


# ─── Artifact paths ───────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path(__file__).parent / "pipeline_output" / "LightGBM_06-May-2026_18-04-07"

artifacts: dict = {}


def load_artifacts() -> None:
    """
    Load all model artifacts from ``ARTIFACTS_DIR`` into the global ``artifacts`` dict.

    Artifacts loaded
    ----------------
    model                : trained LightGBM classifier
    scaler               : fitted RobustScaler
    label_encoder        : LabelEncoder mapping class indices → class-name strings
    encoding             : categorical encoding artifacts (reserved for future use)
    outlier_transformers : per-feature outlier-handling strategies and fitted objects
    feature_names        : ordered list of the 28 model input feature names
    metadata             : training-time metadata (accuracy, class names, scaler type, …)

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
    """
    Application lifespan handler (startup / shutdown).

    On startup, loads all model artifacts and verifies that the API key
    environment variable is configured.  Calls ``sys.exit(1)`` on either
    failure so the container is marked unhealthy immediately rather than
    accepting requests it cannot serve.
    """
    # Fail hard at startup so the container is marked unhealthy immediately.
    try:
        load_artifacts()
    except Exception as e:
        # Log the full traceback so the root cause is visible in container logs.
        logger.critical("Failed to load model artifacts — %s\n%s", e, traceback.format_exc())
        sys.exit(1)

    # Refuse to start if the API key is not configured.
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


# ─── Request / Response schemas ───────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Input schema for ``POST /predict``.

    Contains the 28 numeric features the LightGBM model was trained on.
    All fields are required.

    Validation rules
    ----------------
    * All ratio / proportion fields are validated to ``[0, 1]``.
    * ``overall_sentiment_score`` is validated to ``[-1, 1]``.
    * ``sentiment_trajectory_slope`` accepts any finite real number.
    * ``sentence_count`` and structural depth fields must be strictly positive (> 0).
    * ``negative_emotion_spike_count`` must be non-negative (>= 0).
    * All fields are checked to be finite (no ``inf`` / ``nan``).
    """
    # ── Lexical / surface features ────────────────────────────────────────────
    hapax_legoman_ratio: float                          # once-occurring word ratio [0, 1]

    # ── Syntactic features ────────────────────────────────────────────────────
    sentence_count: float                               # number of sentences (> 0)
    average_sentence_length: float                      # avg words per sentence (> 0)
    noun_ratio: float                                   # noun proportion [0, 1]
    verb_ratio: float                                   # verb proportion [0, 1]
    adverb_ratio: float                                 # adverb proportion [0, 1]
    first_person_singular_pronoun_frequency: float      # I/me/my ratio [0, 1]
    modal_verb_frequency: float                         # modal verb ratio [0, 1]
    negative_frequency: float                           # negation word ratio [0, 1]

    # ── Deep syntactic features ───────────────────────────────────────────────
    parse_tree_depth: float                             # avg parse tree depth (> 0)
    subordinate_clause_ratio: float                     # subordinate clause ratio [0, 1]

    # ── Sentiment / emotion features ──────────────────────────────────────────
    overall_sentiment_score: float                      # tanh(pos - neg) [-1, 1]
    fear_word_frequency: float                          # fear word ratio [0, 1]
    sadness_word_frequency: float                       # sadness word ratio [0, 1]
    anger_word_frequency: float                         # anger word ratio [0, 1]
    max_negative_emotion: float                         # peak negative emotion score [0, 1]
    negative_emotion_spike_count: float                 # count of negative spikes (>= 0)
    sentiment_trajectory_slope: float                   # slope of sentiment over time (any real)

    # ── Emotional dynamics ────────────────────────────────────────────────────
    emotional_volatility_score: float                   # volatility of emotion [0, 1]

    # ── Psychological / cognitive features ────────────────────────────────────
    absolutist_word_frequency: float                    # absolutist word ratio [0, 1]
    catastrophizing_indicators: float                   # catastrophizing score [0, 1]
    external_locus_of_control_score: float              # external LoC score [0, 1]
    uncertainty_word_frequency: float                   # uncertainty word ratio [0, 1]
    avoidance_language_frequency: float                 # avoidance language ratio [0, 1]

    # ── Temporal focus features ───────────────────────────────────────────────
    past_focused_word_ratio: float                      # past-tense ratio [0, 1]
    present_focused_word_ratio: float                   # present-tense ratio [0, 1]
    future_focused_word_ratio: float                    # future-tense ratio [0, 1]

    # ── Miscellaneous ─────────────────────────────────────────────────────────
    cognitive_load_score: float                         # cognitive load index (> 0)

    # ── Validators ───────────────────────────────────────────────────────────

    @field_validator("overall_sentiment_score", "sentiment_trajectory_slope")
    @classmethod
    def validate_unbounded_finite(cls, v: float) -> float:
        """Reject ``inf`` / ``nan``; no range restriction (applied before ``validate_sentiment``)."""
        if not np.isfinite(v):
            raise ValueError(f"Expected a finite number, got {v}")
        return v

    @field_validator("overall_sentiment_score")
    @classmethod
    def validate_sentiment(cls, v: float) -> float:
        """Enforce the tanh-derived range ``[-1, 1]`` for the overall sentiment score."""
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"overall_sentiment_score must be in [-1, 1], got {v}")
        return v

    @field_validator(
        "hapax_legoman_ratio", "noun_ratio", "verb_ratio", "adverb_ratio",
        "first_person_singular_pronoun_frequency", "modal_verb_frequency", "negative_frequency",
        "subordinate_clause_ratio",
        "fear_word_frequency", "sadness_word_frequency", "anger_word_frequency",
        "max_negative_emotion", "emotional_volatility_score",
        "absolutist_word_frequency", "catastrophizing_indicators",
        "external_locus_of_control_score", "uncertainty_word_frequency",
        "avoidance_language_frequency",
        "past_focused_word_ratio", "present_focused_word_ratio", "future_focused_word_ratio",
    )
    @classmethod
    def validate_unit_ratios(cls, v: float) -> float:
        """Ensure ratio / proportion fields are finite and lie in ``[0, 1]``."""
        if not np.isfinite(v):
            raise ValueError(f"Expected a finite number, got {v}")
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Expected value in [0, 1], got {v}")
        return v

    @field_validator("sentence_count")
    @classmethod
    def validate_counts(cls, v: float) -> float:
        """Ensure ``sentence_count`` is finite and strictly positive."""
        if not np.isfinite(v):
            raise ValueError(f"Expected a finite number, got {v}")
        if v <= 0:
            raise ValueError(f"Count must be positive, got {v}")
        return v

    @field_validator("negative_emotion_spike_count")
    @classmethod
    def validate_non_negative(cls, v: float) -> float:
        """Ensure ``negative_emotion_spike_count`` is finite and non-negative."""
        if not np.isfinite(v):
            raise ValueError(f"Expected a finite number, got {v}")
        if v < 0:
            raise ValueError(f"Expected non-negative value, got {v}")
        return v

    @field_validator("average_sentence_length", "parse_tree_depth")
    @classmethod
    def validate_positive_continuous(cls, v: float) -> float:
        """Ensure continuous structural features are finite and strictly positive."""
        if not np.isfinite(v):
            raise ValueError(f"Expected a finite number, got {v}")
        if v <= 0:
            raise ValueError(f"Expected positive value, got {v}")
        return v

    @field_validator("cognitive_load_score")
    @classmethod
    def validate_cognitive_load(cls, v: float) -> float:
        """Ensure ``cognitive_load_score`` is finite and strictly positive."""
        if not np.isfinite(v):
            raise ValueError(f"Expected a finite number, got {v}")
        if v <= 0:
            raise ValueError(f"cognitive_load_score must be positive, got {v}")
        return v


class PredictResponse(BaseModel):
    """
    Response schema for ``POST /predict``.

    Fields
    ------
    prediction_id  : UUID v4 string for request tracing / deduplication.
    prediction     : predicted class label (e.g. ``"ANXIETY"``).
    confidence     : probability of the predicted class, rounded to 4 d.p.
    probabilities  : full probability distribution over all classes, rounded
                     to 4 d.p.
    model_name     : name of the underlying estimator as recorded in metadata.
    """
    prediction_id: str
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    model_name: str


# ─── Preprocessing ────────────────────────────────────────────────────────────

def apply_outlier_transforms_numpy(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """
    Apply per-feature outlier-handling strategies to a single-sample numpy array.

    Each feature may have a different strategy recorded in the
    ``outlier_transformers`` artifact.  Supported strategies:

    * **yeo-johnson** — apply a fitted ``PowerTransformer`` (sklearn).
    * **sqrt**        — ``sqrt(max(x, 0))``, guarded against negative inputs.
    * **log1p**       — ``log(1 + max(x, 0))``, guarded against negative inputs.
    * **winsorize**   — clip to ``[lower, upper]`` bounds stored in the artifact.

    Parameters
    ----------
    X : np.ndarray, shape (1, n_features)
        Single-sample feature array.  A copy is made internally; the
        original array is not modified.
    feature_names : list[str]
        Column names corresponding to each column of *X*, in order.

    Returns
    -------
    np.ndarray
        Transformed array with the same shape ``(1, n_features)`` as *X*.
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
    """
    Build and preprocess the model input from the raw request field dictionary.

    Steps
    -----
    1. Extract the 28 model features in canonical order from *raw*.
    2. Apply per-feature outlier transforms via
       ``apply_outlier_transforms_numpy``.
    3. Apply the fitted ``RobustScaler``.
    4. Return a named ``pd.DataFrame`` — required because LightGBM was
       fitted with feature names and emits a warning on plain arrays.

    Parameters
    ----------
    raw : dict
        Full field dictionary from ``PredictRequest.model_dump()``.

    Returns
    -------
    pd.DataFrame, shape (1, 28)
        Scaled, outlier-treated feature matrix ready for
        ``model.predict_proba``.
    """
    feature_names = artifacts["feature_names"]
    X = np.array([[raw[f] for f in feature_names]], dtype=np.float64)
    X = apply_outlier_transforms_numpy(X, feature_names)
    scaled = artifacts["scaler"].transform(X)
    return pd.DataFrame(scaled, columns=feature_names)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root(_: None = Security(verify_api_key)):
    """Return basic service information: service name, status, class list, and feature count."""
    return {
        "service": "Mindspace Mental Health Classifier",
        "status": "running",
        "classes": artifacts.get("metadata", {}).get("class_names"),
        "n_features": artifacts.get("metadata", {}).get("n_features"),
    }


@app.get("/health")
def health():
    """
    Liveness / readiness probe — no authentication required.

    Returns HTTP 200 with ``{"status": "ok"}`` when all model artifacts are
    loaded, or HTTP 503 with ``{"status": "unavailable"}`` otherwise.
    Suitable for use as a Docker / Kubernetes health-check target.
    """
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
def predict(request: PredictRequest, _: None = Security(verify_api_key)):
    """
    Predict the mental health profile for a single text / speech sample.

    Accepts the 28-field ``PredictRequest`` body (all fields required).
    Returns a ``PredictResponse`` containing:

    * ``prediction``    — the top predicted class label.
    * ``confidence``    — probability of that class (0–1, 4 d.p.).
    * ``probabilities`` — full distribution over all classes.
    * ``prediction_id`` — UUID v4 for request tracing.
    """
    try:
        raw = request.model_dump()
        X = preprocess(raw)
    except Exception as e:
        # Log server-side so preprocessing bugs are visible in container logs,
        # not just as a 422 response the caller sees.
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

        # Audit log — one line per prediction for tracing and monitoring.
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
        # Log the full traceback — a 500 here means a bug in inference code,
        # not a bad request, so we need the stack trace to debug it.
        logger.error("Prediction failed — %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail={"error": "prediction_failed", "message": str(e)})


@app.get("/model/info")
def model_info(_: None = Security(verify_api_key)):
    """Return model metadata: estimator name, feature count, feature names, class names, and scaler type."""
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
    uvicorn.run("main:app", host="0.0.0.0", port=9000)
