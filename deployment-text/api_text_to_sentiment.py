# =============================================================================
# Mindspace Mental Health Classifier — FastAPI Inference Server
# =============================================================================
# This file is the entire backend API for the Mindspace project.
# It loads the trained LightGBM model and all preprocessing artifacts at
# startup, then serves predictions via HTTP endpoints.
# The API expects clients to send 43 pre-extracted features (from speech/text)
# and returns the predicted mental health profile along with confidence and
# probabilities for all classes.
# Flow for every prediction request:
#   1. Client sends 43 float features (extracted from speech/text)
#   2. API applies outlier smoothing (same transforms used during training)
#   3. API scales the features with RobustScaler (same scaler from training)
#   4. LightGBM model predicts one of 7 mental health profiles
#   5. API returns the label + confidence + all 7 class probabilities
# =============================================================================

import json
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from contextlib import asynccontextmanager # for FastAPI lifespan management

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security # for API key auth dependency
from fastapi.middleware.cors import CORSMiddleware # to allow cross-origin requests from browser-based frontends
from fastapi.security.api_key import APIKeyHeader # to read API key from request headers
from pydantic import BaseModel, field_validator # for defining request/response schemas and validating input features

# ─── Load environment variables ───────────────────────────────────────────────
# Reads API_KEY (and any other vars) from deployment/.env at startup.
# On a real server, set the environment variable directly instead of using .env.
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ─── API Key auth ─────────────────────────────────────────────────────────────
# Clients must send their key in the X-API-Key request header.
# Raises 403 immediately if the key is missing or wrong.
_API_KEY = os.environ.get("API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# FastAPI dependency function to verify the API key on protected routes.
def verify_api_key(key: str = Security(_api_key_header)) -> None:
    """FastAPI dependency — guards any route it is added to."""
    if not _API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfiguration: API_KEY not set.")
    if key != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key. Pass it as X-API-Key header.")

# ─── Artifact paths ──────────────────────────────────────────────────────────
# All model artifacts were saved by the ML pipeline (full-final-pipeline.ipynb)
# into a timestamped folder. This path points to that folder.
# __file__ is deployment-text/api_text_to_sentiment.py → .parent is this folder.

ARTIFACTS_DIR = Path(__file__).parent / "pipeline_output" / "LightGBM_13032026_110356"

# ─── Global state (loaded once at startup) ────────────────────────────────────
# We use a plain dict to hold all artifacts in memory so every request can
# reuse them without re-loading from disk each time (which would be very slow).

artifacts = {}


def load_artifacts():
    """
    Load all 7 ML artifacts from disk into the global `artifacts` dict.
    Called once when the server starts up (see lifespan below).

    Artifacts loaded:
      - best_model.joblib          → trained LightGBM classifier
      - scaler.joblib              → RobustScaler (fit on 40K training rows)
      - label_encoder.joblib       → maps integer predictions → class name strings
      - encoding_artifacts.joblib  → categorical encoding maps (not used at inference
                                     since language is already one-hot in input)
      - outlier_transformers.joblib → per-column smoothing params (winsorize bounds,
                                      yeo-johnson fitted transformer, sqrt shift)
      - feature_names.json         → ordered list of 43 feature names the model expects
      - model_metadata.json        → hyperparams, CV score, test metrics, class names
    """
    artifacts["model"]               = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    artifacts["scaler"]              = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    artifacts["label_encoder"]       = joblib.load(ARTIFACTS_DIR / "label_encoder.joblib")
    artifacts["encoding"]            = joblib.load(ARTIFACTS_DIR / "encoding_artifacts.joblib")
    artifacts["outlier_transformers"] = joblib.load(ARTIFACTS_DIR / "outlier_transformers.joblib")
    artifacts["feature_names"]       = json.loads((ARTIFACTS_DIR / "feature_names.json").read_text())
    artifacts["metadata"]            = json.loads((ARTIFACTS_DIR / "model_metadata.json").read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager — runs code at startup and shutdown.
    Everything before `yield` runs when the server starts.
    Everything after `yield` would run on shutdown (nothing needed here).
    This is the modern FastAPI replacement for @app.on_event("startup").
    """
    load_artifacts()
    yield


# ─── App ──────────────────────────────────────────────────────────────────────
# Create the FastAPI application instance.
# `lifespan=lifespan` wires up the startup artifact loading defined above.
# The title/description appear in the auto-generated Swagger UI at /docs.

app = FastAPI(
    title="Mindspace Mental Health Classifier",
    description="Predicts mental health profile from speech/text features.",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
# Allow any origin so browser-based frontends (voice agent UI, demo pages) can
# call the API. Tighten allow_origins to specific domains in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],# In production, replace "*" with specific frontend domains for security if needed (but in our case this api will work as service).
    allow_methods=["*"], # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers (including X-API-Key for authentication)
)

# ─── Request / Response schemas ───────────────────────────────────────────────
# Pydantic models define the exact shape of JSON that the API accepts and returns. 
# FastAPI uses these to automatically validate incoming requests and
# generate the interactive docs at /docs — no extra work needed.

class PredictRequest(BaseModel):
    """
    The 43 features the model was trained on, after feature selection in the pipeline.
    The caller is responsible for extracting these from raw text before calling /predict.
    
    All values are floats. The only exception is language_hindi / language_marathi
    which must be exactly 0.0 or 1.0 (one-hot encoded language flags).

    Note: not all embedding dimensions (emb_0 to emb_31) are included — only the
    17 that survived the pipeline's feature selection step are required.
    """

    # ── Linguistic / Semantic features (19) ──────────────────────────────────
    # These capture the emotional and semantic content of the text.
    overall_sentiment_score: float      # tanh of (positive - negative); typical range [-1, 1]
    semantic_coherence_score: float     # sentence coherence [0, 1]; typically [0.19, 0.99]
    self_reference_density: float       # first-person pronoun ratio; typically [0.00, 0.45]
    future_focus_ratio: float           # future-tense word ratio; typically [0.00, 0.38]
    positive_emotion_ratio: float       # positive words; typically [0.00, 0.15]
    fear_word_frequency: float          # fear-related words; typically [0.00, 0.38]
    sadness_word_frequency: float       # sadness-related words; typically [0.00, 0.53]
    negative_emotion_ratio: float       # negative words; typically [0.00, 0.65]
    uncertainty_word_frequency: float   # uncertainty words; typically [0.00, 0.45]
    anger_word_frequency: float         # anger-related words; typically [0.00, 0.20]
    rumination_phrase_frequency: float  # rumination patterns; typically [0.00, 0.40]
    filler_word_frequency: float        # filler words (um, uh, like); typically [0.00, 0.28]
    topic_shift_frequency: float        # topic entropy; typical range [0, 1]
    total_word_count: float             # total words spoken/written
    avg_sentence_length: float          # average words per sentence
    language_model_perplexity: float    # how "surprising"/unpredictable the text is; higher = more chaotic
    past_focus_ratio: float             # proportion of past-tense words; range [0, 0.4]
    repetition_rate: float              # how often words/phrases repeat; range [0, 0.2]
    adjective_ratio: float              # proportion of adjectives in the text; range [0, 0.25]

    # ── Topic model outputs (5) ───────────────────────────────────────────────
    # Weights from a topic model (e.g., LDA) trained on the corpus.
    # Each row sums to 1.0 (a probability distribution over 5 topics).
    topic_0: float
    topic_1: float
    topic_2: float
    topic_3: float
    topic_4: float

    # ── Sentence embedding dimensions (17 of 32) ──────────────────────────────
    # 32-dimensional sentence embeddings were generated for the text.
    # After feature selection, only these 17 dimensions were retained.
    # Non-contiguous indices (e.g., emb_2, emb_6 were dropped as low-importance).
    emb_1: float
    emb_3: float
    emb_4: float
    emb_5: float
    emb_7: float
    emb_8: float
    emb_10: float
    emb_11: float
    emb_12: float
    emb_14: float
    emb_15: float
    emb_21: float
    emb_22: float
    emb_25: float
    emb_28: float
    emb_29: float
    emb_30: float

    # ── Language flags (2) ────────────────────────────────────────────────────
    # One-hot encoding of the detected language of the input text.
    # language_hindi=1, language_marathi=0 → Hindi
    # language_hindi=0, language_marathi=1 → Marathi
    # language_hindi=0, language_marathi=0 → English (the baseline/dropped category)
    language_hindi: float    # 1.0 if Hindi, else 0.0
    language_marathi: float  # 1.0 if Marathi, else 0.0

    @field_validator("language_hindi", "language_marathi")
    @classmethod
    def language_must_be_binary(cls, v: float) -> float:
        # Enforce that language flags are strictly 0 or 1 — any other value is invalid.
        if v not in (0.0, 1.0):
            raise ValueError("language flags must be 0 or 1")
        return v

    @field_validator(
        "overall_sentiment_score", "semantic_coherence_score", "self_reference_density",
        "future_focus_ratio", "positive_emotion_ratio", "fear_word_frequency",
        "sadness_word_frequency", "negative_emotion_ratio", "uncertainty_word_frequency",
        "anger_word_frequency", "rumination_phrase_frequency", "filler_word_frequency",
        "past_focus_ratio", "repetition_rate", "adjective_ratio"
    )
    @classmethod
    def validate_ratios(cls, v: float) -> float:
        """Ratios should be non-negative (typically [0, 1] but may exceed slightly due to outliers)."""
        if v < -0.5:  # Allow small negative values from preprocessing, reject large ones
            raise ValueError(f"Expected non-negative value (ratio), got {v}")
        return v

    @field_validator("topic_shift_frequency")
    @classmethod
    def validate_entropy(cls, v: float) -> float:
        """Entropy (topic shift) should be in [0, 1] approximately."""
        if not 0 <= v <= 1.5:  # Allow slight overshoot
            raise ValueError(f"topic_shift_frequency should be roughly [0, 1], got {v}")
        return v

    @field_validator("total_word_count")
    @classmethod
    def validate_word_count(cls, v: float) -> float:
        """Word count must be positive."""
        if v <= 0:
            raise ValueError(f"total_word_count must be positive, got {v}")
        if v > 10000:
            raise ValueError(f"total_word_count suspiciously high ({v}), expected < 10000")
        return v

    @field_validator("avg_sentence_length")
    @classmethod
    def validate_avg_sentence(cls, v: float) -> float:
        """Average sentence length must be positive."""
        if v <= 0:
            raise ValueError(f"avg_sentence_length must be positive, got {v}")
        if v > 100:
            raise ValueError(f"avg_sentence_length suspiciously high ({v}), expected < 100")
        return v

    @field_validator("language_model_perplexity")
    @classmethod
    def validate_perplexity(cls, v: float) -> float:
        """Perplexity must be positive (typically > 1)."""
        if v < 0:
            raise ValueError(f"language_model_perplexity must be positive, got {v}")
        if v > 1000:
            raise ValueError(f"language_model_perplexity suspiciously high ({v}), expected < 1000")
        return v

    @field_validator("topic_0", "topic_1", "topic_2", "topic_3", "topic_4")
    @classmethod
    def validate_topic_weights(cls, v: float) -> float:
        """Topic weights should be probabilities in [0, 1]."""
        if not 0 <= v <= 1:
            raise ValueError(f"topic weight must be in [0, 1], got {v}")
        return v

    @field_validator(
        "emb_1", "emb_3", "emb_4", "emb_5", "emb_7", "emb_8", "emb_10", "emb_11", "emb_12",
        "emb_14", "emb_15", "emb_21", "emb_22", "emb_25", "emb_28", "emb_29", "emb_30"
    )
    @classmethod
    def validate_embeddings(cls, v: float) -> float:
        """Embeddings are typically in [-3, 3] (normalized vectors)."""
        if not -5 <= v <= 5:
            raise ValueError(f"embedding value out of typical range [-5, 5], got {v}")
        return v


class PredictResponse(BaseModel):
    """
    What the API returns after a successful prediction.

      prediction   — the predicted mental health profile (e.g. "Depression")
      confidence   — probability assigned to the predicted class (0.0 to 1.0)
      probabilities — full probability distribution across all 7 classes
      model        — name of the model that made the prediction ("LightGBM")
      accuracy     — the model's test-set accuracy from training (0.92)
    """
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    model: str
    accuracy: float


# ─── Preprocessing ────────────────────────────────────────────────────────────
# These two functions replicate exactly what the training pipeline did to the
# data before fitting the model. Applying the SAME transforms at inference is
# critical — if we skip them, the feature distributions won't match what the
# model was trained on and predictions will be wrong.

def apply_outlier_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply per-column outlier smoothing using the transformers saved during training.

    During training (pipeline Step 8), each column was tested with 4 strategies:
      - winsorize    → clamp values to [lower, upper] percentile bounds
      - sqrt         → square-root compress large values (shift to non-negative first)
      - yeo-johnson  → power transform that handles both positive and negative values
      - log1p        → log(1+x) compression (not used in this run, but supported)

    The strategy that produced the lowest skew was chosen per column and saved
    in outlier_transformers.joblib. We replay the same strategy here.
    """
    transformers = artifacts["outlier_transformers"]
    df = df.copy()

    for col, info in transformers.items():
        # Skip columns that aren't in the input (e.g., columns dropped by feature selection)
        if col not in df.columns:
            continue
        strategy = info["strategy"]

        if strategy == "yeo-johnson":
            # Use the sklearn PowerTransformer that was fit on training data
            pt = info["fitted_pt"]
            df[col] = pt.transform(df[[col]].values).ravel()

        elif strategy == "sqrt":
            # Clip negatives to 0 then sqrt — matches training pipeline exactly
            df[col] = np.sqrt(df[col].clip(lower=0))

        elif strategy == "log1p":
            # Clip negatives to 0 then log1p — matches training pipeline exactly
            df[col] = np.log1p(df[col].clip(lower=0))

        elif strategy == "winsorize":
            # Clip values to the bounds calculated from training data percentiles
            lower = info["lower"]
            upper = info["upper"]
            df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def preprocess(raw: dict) -> np.ndarray:
    """
    Full preprocessing pipeline for a single inference sample.
    Mirrors training pipeline Steps 8 (outlier) and 12 (scaling) exactly.

    Steps:
      1. Wrap the raw feature dict in a single-row DataFrame
      2. Apply per-column outlier smoothing (same strategies as training)
      3. Scale with RobustScaler (fit on training data — robust to outliers)
      4. Select and reorder to exactly the 43 features the model expects

    Returns a (1, 43) numpy array ready to pass to model.predict_proba().
    """
    feature_names = artifacts["feature_names"]  # ordered list of 43 feature names
    df = pd.DataFrame([raw])                     # single-row DataFrame

    # Step 1: Smooth outliers using the saved per-column transformers
    df = apply_outlier_transforms(df)

    # Step 2 & 3: Scale and select features in the exact order the model expects.
    # df[feature_names] reorders columns to match training order before scaling.
    scaler = artifacts["scaler"]
    return scaler.transform(df[feature_names].values)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """
    Service info endpoint — returns a summary of the running model.
    Useful as a quick sanity check that the right model is loaded.
    No authentication or input required.
    """
    meta = artifacts.get("metadata", {})
    return {
        "service": "Mindspace Mental Health Classifier",
        "model": meta.get("best_model_name"),
        "accuracy": meta.get("test_metrics", {}).get("accuracy"),
        "classes": meta.get("class_names"),
        "n_features": meta.get("n_features"),
    }


@app.get("/health")
def health():
    """
    Health check endpoint — confirms all artifacts are loaded and the server is ready.
    Returns { "status": "ok", "artifacts_loaded": true } when healthy.
    Returns artifacts_loaded: false if startup failed to load any artifact.
    Typically polled by load balancers or monitoring systems.
    """
    expected_keys = {"model", "scaler", "label_encoder", "encoding", "outlier_transformers", "feature_names", "metadata"}
    return {"status": "ok", "artifacts_loaded": expected_keys.issubset(artifacts.keys())}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, _: None = Security(verify_api_key)):
    """
    Main prediction endpoint — the core of the API.

    Accepts 43 pre-extracted speech/text features as JSON and returns:
      - prediction    → the most likely mental health profile
      - confidence    → probability of the predicted class (0–1)
      - probabilities → full softmax distribution across all 7 classes
      - model / accuracy → metadata about the model that ran inference

    Two try/except blocks handle errors at different stages:
      - 422 Unprocessable Entity → something went wrong during preprocessing
        (bad feature values, unexpected column, etc.)
      - 500 Internal Server Error → model inference itself failed
        (should be very rare if preprocessing succeeded)
    """
    # ── Preprocessing ────────────────────────────────────────────────────────
    try:
        raw = request.model_dump()   # convert Pydantic model → plain Python dict
        X = preprocess(raw)          # outlier smooth → scale → (1, 43) numpy array
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    # ── Inference ────────────────────────────────────────────────────────────
    try:
        model = artifacts["model"]           # LightGBM classifier
        le    = artifacts["label_encoder"]   # LabelEncoder: int index → class name
        meta  = artifacts["metadata"]

        # predict_proba returns shape (1, 7) — one probability per class
        proba     = model.predict_proba(X)[0]
        pred_idx  = int(np.argmax(proba))                    # index of highest probability
        pred_label = le.inverse_transform([pred_idx])[0]    # e.g. 2 → "Depression"
        confidence = float(proba[pred_idx])

        # Build a readable dict: {"Anxiety": 0.02, "Depression": 0.94, ...}
        class_names   = le.classes_.tolist()
        probabilities = {cls: round(float(p), 4) for cls, p in zip(class_names, proba)}

        return PredictResponse(
            prediction=pred_label,
            confidence=round(confidence, 4),
            probabilities=probabilities,
            model=meta.get("best_model_name", "LightGBM"),
            accuracy=meta.get("test_metrics", {}).get("accuracy", 0.0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/model/info")
def model_info(_: None = Security(verify_api_key)):
    """
    Full model metadata endpoint — returns everything saved about the trained model.
    Includes hyperparameters, cross-validation score, and all test-set metrics
    (accuracy, F1 macro/weighted, precision, recall).
    Useful for auditing what model is running and verifying its performance.
    """
    meta = artifacts.get("metadata", {})
    return meta


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_text_to_sentiment:app", host="0.0.0.0", port=9000, reload=True)

# To run locally (from inside deployment-text/):
# uvicorn api_text_to_sentiment:app --reload --port 9000
# Then open http://localhost:9000/docs for the interactive Swagger UI.