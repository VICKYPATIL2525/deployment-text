# Mindspace — Text Sentiment API

A FastAPI inference server that predicts a mental health profile from 28 pre-extracted text/speech features using a trained LightGBM model.

## What it does

Accepts 28 numerical features extracted from a speech or text sample and returns the most likely mental health profile out of 5 classes: `ANXIETY`, `BIPOLAR`, `DEPRESSION`, `PHOBIA`, `SUICIDAL_TENDENCY`.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | Yes | Service info — name, supported classes, feature count |
| GET | `/health` | No | Health check — returns `{"status": "ok"}` when ready |
| POST | `/predict` | Yes | Run prediction on 28 input features |
| GET | `/model/info` | Yes | Model structure info (feature names, classes, scaler) |

Authentication is via the `X-API-Key` request header. All endpoints except `/health` require it.

## Input features

The `/predict` endpoint expects exactly **28 required float fields**:

| Group | Count | Fields |
|---|---|---|
| Lexical / surface | 1 | `hapax_legoman_ratio` |
| Syntactic | 8 | `sentence_count`, `average_sentence_length`, `noun_ratio`, `verb_ratio`, `adverb_ratio`, `first_person_singular_pronoun_frequency`, `modal_verb_frequency`, `negative_frequency` |
| Deep syntactic | 2 | `parse_tree_depth`, `subordinate_clause_ratio` |
| Sentiment / emotion | 7 | `overall_sentiment_score`, `fear_word_frequency`, `sadness_word_frequency`, `anger_word_frequency`, `max_negative_emotion`, `negative_emotion_spike_count`, `sentiment_trajectory_slope` |
| Emotional dynamics | 1 | `emotional_volatility_score` |
| Psychological / cognitive | 5 | `absolutist_word_frequency`, `catastrophizing_indicators`, `external_locus_of_control_score`, `uncertainty_word_frequency`, `avoidance_language_frequency` |
| Temporal focus | 3 | `past_focused_word_ratio`, `present_focused_word_ratio`, `future_focused_word_ratio` |
| Miscellaneous | 1 | `cognitive_load_score` |

### Validation rules

| Field(s) | Rule |
|---|---|
| All ratio / proportion fields | Must be in `[0, 1]` |
| `overall_sentiment_score` | Must be in `[-1, 1]` |
| `sentiment_trajectory_slope` | Any finite real number |
| `sentence_count`, `average_sentence_length`, `parse_tree_depth`, `cognitive_load_score` | Must be `> 0` |
| `negative_emotion_spike_count` | Must be `>= 0` |
| All fields | Must be finite — no `inf` or `nan` |

See `demo-api-input-data-sample/` for ready-to-use sample payloads per class.

## Sample request

```python
import requests

payload = {
    "hapax_legoman_ratio": 0.2967731849304067,
    "sentence_count": 52.0,
    "average_sentence_length": 2.3686352126669856,
    "noun_ratio": 0.119075921429951,
    "verb_ratio": 0.1053030456986042,
    "adverb_ratio": 0.0369357405664167,
    "first_person_singular_pronoun_frequency": 0.0360912175890274,
    "modal_verb_frequency": 0.0105439882897891,
    "negative_frequency": 0.0302877317434468,
    "parse_tree_depth": 3.139033393984063,
    "subordinate_clause_ratio": 0.4092288771764105,
    "overall_sentiment_score": -0.440641413695465,
    "fear_word_frequency": 0.8018379677415697,
    "sadness_word_frequency": 0.1156283330711261,
    "anger_word_frequency": 0.2595347173804833,
    "max_negative_emotion": 0.8018379677415697,
    "negative_emotion_spike_count": 4.0,
    "sentiment_trajectory_slope": -0.0027067640342667,
    "emotional_volatility_score": 0.7623467828488708,
    "absolutist_word_frequency": 0.1692641149848399,
    "catastrophizing_indicators": 0.4902486769242626,
    "external_locus_of_control_score": 0.0,
    "uncertainty_word_frequency": 0.6252863650474074,
    "avoidance_language_frequency": 0.4899296465048904,
    "past_focused_word_ratio": 0.0,
    "present_focused_word_ratio": 0.4327344615168387,
    "future_focused_word_ratio": 0.4004957450050325,
    "cognitive_load_score": 31.3483649508134
}

response = requests.post(
    "http://localhost:9000/predict",
    json=payload,
    headers={"X-API-Key": "your_api_key"}
)
print(response.json())
```

### Sample response

```json
{
  "prediction_id": "a3f1c2d4-...",
  "prediction": "ANXIETY",
  "confidence": 0.8731,
  "probabilities": {
    "ANXIETY": 0.8731,
    "BIPOLAR": 0.0412,
    "DEPRESSION": 0.0521,
    "PHOBIA": 0.0198,
    "SUICIDAL_TENDENCY": 0.0138
  },
  "model_name": "LightGBM"
}
```

### Ready-to-run test script

```bash
python test_predict_api.py
```

Reads `payload.json`, calls all 4 endpoints, prints results to the console, and saves a timestamped JSON file to `test_api_output/`. Requires `.env` with `MINDSPACE_TEXT_API_KEY` set.

## Setup

### 1. Configure environment

```bash
cp example.env .env
# Edit .env and set MINDSPACE_TEXT_API_KEY to your key
```

### 2. Run with Docker (recommended)

```bash
docker compose up --build
```

### 3. Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 9000
```

Interactive API docs available at `http://localhost:9000/docs`.

## Logging

The API writes structured log lines to stdout in the format:

```
YYYY-MM-DD HH:MM:SS | LEVEL | message
```

Key log events:

| Event | Level | When |
|---|---|---|
| Artifact loading started | `INFO` | Startup |
| All artifacts loaded (model name, feature count, classes) | `INFO` | Startup |
| API ready to serve | `INFO` | Startup complete |
| Per-prediction audit line (`prediction_id`, `prediction`, `confidence`) | `INFO` | Every `/predict` call |
| Preprocessing error with full traceback | `ERROR` | Bad input that passes Pydantic but fails preprocessing |
| Inference error with full traceback | `ERROR` | Unexpected model/artifact failure |
| Artifact load failure with full traceback | `CRITICAL` | Startup — causes `sys.exit(1)` |
| Missing API key | `CRITICAL` | Startup — causes `sys.exit(1)` |
| Shutdown | `INFO` | On stop |

With Docker, logs are accessible via:

```bash
docker logs mindspace-text-api
docker logs mindspace-text-api --follow
```

## File structure

```
deployment-text/
├── main.py                        # FastAPI application
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker image (Gunicorn + Uvicorn workers)
├── docker-compose.yml             # Single-service compose file
├── payload.json                   # Sample input for /predict (28 model features)
├── test_predict_api.py            # Ready-to-run test script
├── example.env                    # Environment variable template
├── .env                           # Your actual keys (never commit this)
├── test_api_output/               # Timestamped test run outputs (auto-created)
├── demo-api-input-data-sample/    # 2 sample payloads per class (10 files total)
│   ├── anxiety_sample_1.json
│   ├── anxiety_sample_2.json
│   ├── bipolar_mania_sample_1.json
│   ├── bipolar_mania_sample_2.json
│   ├── depression_sample_1.json
│   ├── depression_sample_2.json
│   ├── phobia_sample_1.json
│   ├── phobia_sample_2.json
│   ├── suicidal_tendency_sample_1.json
│   └── suicidal_tendency_sample_2.json
└── pipeline_output/               # Trained model artifacts
    └── LightGBM_06-May-2026_18-04-07/
        ├── best_model.joblib
        ├── scaler.joblib
        ├── label_encoder.joblib
        ├── encoding_artifacts.joblib
        ├── outlier_transformers.joblib
        ├── feature_names.json
        └── model_metadata.json
```

## Model performance

Trained on 15,000 samples (12,000 train / 3,000 test), selected via 5-fold cross-validation.

| Metric | Score |
|---|---|
| Accuracy | 94.7% |
| F1 macro | 94.68% |
| F1 weighted | 94.68% |
| CV F1 macro | 94.62% |

Model: **LightGBM** — chosen over Random Forest, XGBoost, Extra Trees, HistGradientBoosting, Logistic Regression, SVM, and KNN.
Scaler: **RobustScaler**. Feature selection: 28 features chosen from 52 via VIF, correlation filtering, and mutual information ranking.
