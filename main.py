import json
import os
import sys
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
from pydantic import BaseModel, field_validator, model_validator

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
ARTIFACTS_DIR = Path(__file__).parent / "pipeline_output" / "LightGBM_13032026_110356"

artifacts: dict = {}


def load_artifacts() -> None:
    artifacts["model"]                = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    artifacts["scaler"]               = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    artifacts["label_encoder"]        = joblib.load(ARTIFACTS_DIR / "label_encoder.joblib")
    artifacts["encoding"]             = joblib.load(ARTIFACTS_DIR / "encoding_artifacts.joblib")
    artifacts["outlier_transformers"] = joblib.load(ARTIFACTS_DIR / "outlier_transformers.joblib")
    artifacts["feature_names"]        = json.loads((ARTIFACTS_DIR / "feature_names.json").read_text())
    artifacts["metadata"]             = json.loads((ARTIFACTS_DIR / "model_metadata.json").read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail hard at startup so the container is marked unhealthy immediately.
    try:
        load_artifacts()
    except Exception as e:
        print(f"FATAL: Failed to load model artifacts — {e}", file=sys.stderr)
        sys.exit(1)

    # Refuse to start if the API key is not configured.
    if not _get_api_key():
        print("FATAL: MINDSPACE_TEXT_API_KEY is not set in the environment.", file=sys.stderr)
        sys.exit(1)

    yield


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
    # ── Linguistic / Semantic features (19) ──────────────────────────────────
    overall_sentiment_score: float      # tanh(positive - negative); range [-1, 1]
    semantic_coherence_score: float     # sentence coherence [0, 1]
    self_reference_density: float       # first-person pronoun ratio [0, 0.45]
    future_focus_ratio: float           # future-tense word ratio [0, 0.38]
    positive_emotion_ratio: float       # positive word ratio [0, 0.15]
    fear_word_frequency: float          # fear-related word ratio [0, 0.38]
    sadness_word_frequency: float       # sadness-related word ratio [0, 0.53]
    negative_emotion_ratio: float       # negative word ratio [0, 0.65]
    uncertainty_word_frequency: float   # uncertainty word ratio [0, 0.45]
    anger_word_frequency: float         # anger-related word ratio [0, 0.20]
    rumination_phrase_frequency: float  # rumination pattern ratio [0, 0.40]
    filler_word_frequency: float        # filler word ratio [0, 0.28]
    topic_shift_frequency: float        # topic entropy [0, 1]
    total_word_count: float             # total words spoken/written
    avg_sentence_length: float          # average words per sentence
    language_model_perplexity: float    # text unpredictability score (> 1)
    past_focus_ratio: float             # past-tense word ratio [0, 0.4]
    repetition_rate: float              # word/phrase repetition ratio [0, 0.2]
    adjective_ratio: float              # adjective proportion [0, 0.25]

    # ── Topic model outputs (5) ───────────────────────────────────────────────
    topic_0: float
    topic_1: float
    topic_2: float
    topic_3: float
    topic_4: float

    # ── Sentence embedding dimensions (17 of 32) ──────────────────────────────
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
    # language_hindi=1, language_marathi=0 → Hindi
    # language_hindi=0, language_marathi=1 → Marathi
    # language_hindi=0, language_marathi=0 → English
    language_hindi: float
    language_marathi: float

    @field_validator("overall_sentiment_score")
    @classmethod
    def validate_sentiment(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"overall_sentiment_score must be in [-1, 1], got {v}")
        return v

    @field_validator(
        "semantic_coherence_score", "self_reference_density",
        "future_focus_ratio", "positive_emotion_ratio", "fear_word_frequency",
        "sadness_word_frequency", "negative_emotion_ratio", "uncertainty_word_frequency",
        "anger_word_frequency", "rumination_phrase_frequency", "filler_word_frequency",
        "past_focus_ratio", "repetition_rate", "adjective_ratio"
    )
    @classmethod
    def validate_ratios(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError(f"Expected a finite number, got {v}")
        if v < 0:
            raise ValueError(f"Expected non-negative ratio, got {v}")
        return v

    @field_validator("topic_shift_frequency")
    @classmethod
    def validate_entropy(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError(f"Expected a finite number, got {v}")
        if not 0 <= v <= 1.5:
            raise ValueError(f"topic_shift_frequency should be in [0, 1.5], got {v}")
        return v

    @field_validator("total_word_count")
    @classmethod
    def validate_word_count(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError(f"Expected a finite number, got {v}")
        if v <= 0:
            raise ValueError(f"total_word_count must be positive, got {v}")
        if v > 10000:
            raise ValueError(f"total_word_count suspiciously high ({v}), expected < 10000")
        return v

    @field_validator("avg_sentence_length")
    @classmethod
    def validate_avg_sentence(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError(f"Expected a finite number, got {v}")
        if v <= 0:
            raise ValueError(f"avg_sentence_length must be positive, got {v}")
        if v > 100:
            raise ValueError(f"avg_sentence_length suspiciously high ({v}), expected < 100")
        return v

    @field_validator("language_model_perplexity")
    @classmethod
    def validate_perplexity(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError(f"Expected a finite number, got {v}")
        if v <= 0:
            raise ValueError(f"language_model_perplexity must be positive, got {v}")
        if v > 1000:
            raise ValueError(f"language_model_perplexity suspiciously high ({v}), expected < 1000")
        return v

    @field_validator("topic_0", "topic_1", "topic_2", "topic_3", "topic_4")
    @classmethod
    def validate_topic_weights(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError(f"Expected a finite number, got {v}")
        if not 0 <= v <= 1:
            raise ValueError(f"topic weight must be in [0, 1], got {v}")
        return v

    @field_validator(
        "emb_1", "emb_3", "emb_4", "emb_5", "emb_7", "emb_8", "emb_10", "emb_11", "emb_12",
        "emb_14", "emb_15", "emb_21", "emb_22", "emb_25", "emb_28", "emb_29", "emb_30"
    )
    @classmethod
    def validate_embeddings(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError(f"Expected a finite number, got {v}")
        if not -5 <= v <= 5:
            raise ValueError(f"embedding value out of typical range [-5, 5], got {v}")
        return v

    @model_validator(mode="after")
    def validate_language_flags(self) -> "PredictRequest":
        hi = self.language_hindi
        mr = self.language_marathi
        if hi not in (0.0, 1.0):
            raise ValueError("language_hindi must be 0 or 1")
        if mr not in (0.0, 1.0):
            raise ValueError("language_marathi must be 0 or 1")
        if hi == 1.0 and mr == 1.0:
            raise ValueError("language_hindi and language_marathi cannot both be 1 — only one language at a time.")
        return self


class PredictResponse(BaseModel):
    prediction_id: str
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    model_name: str


# ─── Preprocessing ────────────────────────────────────────────────────────────

def apply_outlier_transforms(df: pd.DataFrame) -> pd.DataFrame:
    transformers = artifacts["outlier_transformers"]
    df = df.copy()

    for col, info in transformers.items():
        if col not in df.columns:
            continue
        strategy = info["strategy"]

        if strategy == "yeo-johnson":
            pt = info["fitted_pt"]
            df[col] = pt.transform(df[[col]].values).ravel()
        elif strategy == "sqrt":
            df[col] = np.sqrt(df[col].clip(lower=0))
        elif strategy == "log1p":
            df[col] = np.log1p(df[col].clip(lower=0))
        elif strategy == "winsorize":
            df[col] = df[col].clip(lower=info["lower"], upper=info["upper"])

    return df


def preprocess(raw: dict) -> np.ndarray:
    feature_names = artifacts["feature_names"]
    df = pd.DataFrame([raw])
    df = apply_outlier_transforms(df)
    scaler = artifacts["scaler"]
    return scaler.transform(df[feature_names].values)


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
def predict(request: PredictRequest, _: None = Security(verify_api_key)):
    try:
        raw = request.model_dump()
        X = preprocess(raw)
    except Exception as e:
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

        return PredictResponse(
            prediction_id=str(uuid.uuid4()),
            prediction=pred_label,
            confidence=round(confidence, 4),
            probabilities=probabilities,
            model_name=artifacts.get("metadata", {}).get("best_model_name", "unknown"),
        )
    except Exception as e:
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
    uvicorn.run("main:app", host="0.0.0.0", port=9000)
