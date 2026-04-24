# Mindspace — Text Sentiment API

A FastAPI inference server that predicts a mental health profile from 43 pre-extracted text/speech features using a trained LightGBM model.

## What it does

Accepts 43 numerical features extracted from a speech or text sample and returns the most likely mental health profile out of 7 classes: `Anxiety`, `Bipolar_Mania`, `Depression`, `Normal`, `Phobia`, `Stress`, `Suicidal_Tendency`.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | No | Service info — name, supported classes, feature count |
| GET | `/health` | No | Health check — returns `{"status": "ok"}` when ready |
| POST | `/predict` | Yes | Run prediction on 43 input features |
| GET | `/model/info` | Yes | Model structure info (feature names, classes, scaler) |

## Input features

The `/predict` endpoint expects 43 float fields grouped as:

- **Linguistic/Semantic (19):** `overall_sentiment_score`, `semantic_coherence_score`, `self_reference_density`, `future_focus_ratio`, `positive_emotion_ratio`, `fear_word_frequency`, `sadness_word_frequency`, `negative_emotion_ratio`, `uncertainty_word_frequency`, `anger_word_frequency`, `rumination_phrase_frequency`, `filler_word_frequency`, `topic_shift_frequency`, `total_word_count`, `avg_sentence_length`, `language_model_perplexity`, `past_focus_ratio`, `repetition_rate`, `adjective_ratio`
- **Topic model (5):** `topic_0` to `topic_4` (probability distribution, each in [0, 1])
- **Sentence embeddings (17):** `emb_1`, `emb_3`, `emb_4`, `emb_5`, `emb_7`, `emb_8`, `emb_10`, `emb_11`, `emb_12`, `emb_14`, `emb_15`, `emb_21`, `emb_22`, `emb_25`, `emb_28`, `emb_29`, `emb_30`
- **Language flags (2):** `language_hindi`, `language_marathi` — one-hot (0.0 or 1.0, not both 1)

See `payload.json` for a complete sample input.

## How to use

### Prerequisites

```
pip install requests python-dotenv
```

### Call with Python `requests`

```python
import requests

url = "http://localhost:9000/predict"

payload = {
    "overall_sentiment_score": -0.277664,
    "semantic_coherence_score": 0.811103,
    "self_reference_density": 0.075388,
    "future_focus_ratio": 0.152573,
    "positive_emotion_ratio": 0.05661,
    "fear_word_frequency": 0.020899,
    "sadness_word_frequency": 0.018766,
    "negative_emotion_ratio": 0.11364,
    "uncertainty_word_frequency": 0.015442,
    "anger_word_frequency": 0.011136,
    "rumination_phrase_frequency": 0.036132,
    "filler_word_frequency": 0.17825,
    "topic_shift_frequency": 0.742109,
    "total_word_count": 381.0,
    "avg_sentence_length": 11.134937,
    "language_model_perplexity": 129.701744,
    "past_focus_ratio": 0.305773,
    "repetition_rate": 0.100095,
    "adjective_ratio": 0.066824,
    "topic_0": 0.298104,
    "topic_1": 0.513306,
    "topic_2": 0.024847,
    "topic_3": 0.110643,
    "topic_4": 0.053101,
    "emb_1": -0.102718,
    "emb_3": -0.945314,
    "emb_4": -0.880676,
    "emb_5": -1.080255,
    "emb_7": -1.284114,
    "emb_8": 1.158825,
    "emb_10": 1.00407,
    "emb_11": -0.827109,
    "emb_12": 1.572154,
    "emb_14": 0.567563,
    "emb_15": 0.01696,
    "emb_21": -0.615925,
    "emb_22": -1.951899,
    "emb_25": 0.637446,
    "emb_28": 0.57704,
    "emb_29": 0.01438,
    "emb_30": -2.41189,
    "language_hindi": 0.0,
    "language_marathi": 0.0
}

headers = {
    "X-API-Key": "your_api_key"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

### Ready-to-run script

```bash
python test_predict_api.py
```

`test_predict_api.py` reads `payload.json`, calls `/predict`, and prints the result. Requires `.env` with `MINDSPACE_TEXT_API_KEY` set.

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

Interactive docs available at `http://localhost:9000/docs`.

## File structure

```
deployment-text/
├── main.py               # FastAPI application
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Single-service compose file
├── payload.json          # Sample input for /predict
├── output.json           # Sample output from /predict
├── test_predict_api.py   # Ready-to-run test script
├── example.env           # Environment variable template
├── .env                  # Your actual keys (never commit this)
└── pipeline_output/      # Trained model artifacts
```
