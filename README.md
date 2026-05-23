# Mindspace Text API

FastAPI inference service for mental-health text feature classification.

It loads trained artifacts from `pipeline_output/Extra_Trees_18-May-2026_12-11-08` and serves single-sample predictions through REST endpoints.

## Current model snapshot

- Model: `Extra Trees`
- Task: multi-class classification
- Classes: `ANXIETY`, `BIPOLAR`, `DEPRESSION`, `NORMAL`, `STRESS`, `SUICIDAL`
- Number of input features: `20`
- Scaler: `RobustScaler`
- API port: `5500`

## API endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/` | Yes (`X-API-Key`) | Service info |
| GET | `/health` | No | Health/readiness check |
| POST | `/predict` | Yes (`X-API-Key`) | Predict one sample |
| GET | `/model/info` | Yes (`X-API-Key`) | Model metadata |

## Authentication

All endpoints except `/health` require header:

`X-API-Key: <your-key>`

The key is read from environment variable `MINDSPACE_TEXT_API_KEY`.

## Request schema (`POST /predict`)

The request body schema is generated from `feature_names.json` and exposed in Swagger.

Current required float fields:

1. `unique_word_count`
2. `moving_average_ttr`
3. `adverb_ratio`
4. `modal_verb_frequency`
5. `negative_frequency`
6. `overall_sentiment_score`
7. `fear_word_frequency`
8. `sadness_word_frequency`
9. `anger_word_frequency`
10. `surprise_frequency`
11. `max_negative_emotion`
12. `negative_emotion_spike_count`
13. `sentiment_trajectory_slope`
14. `emotional_volatility_score`
15. `catastrophizing_indicators`
16. `external_locus_of_control_score`
17. `self_reference_density`
18. `past_focused_word_ratio`
19. `future_focused_word_ratio`
20. `filler_word_frequency`

## Quick setup

### 1) Create `.env`

Copy `example.env` to `.env` and set your API key:

```env
MINDSPACE_TEXT_API_KEY=your-real-key
```

Optional variable for client scripts:

```env
MINDSPACE_TEXT_HOST=http://127.0.0.1:5500
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run API locally

```bash
python main.py
```

API docs: `http://127.0.0.1:5500/docs`
OpenAPI JSON: `http://127.0.0.1:5500/openapi.json`

## Docker run

```bash
docker compose up --build
```

Container exposes `5500` and runs Gunicorn + Uvicorn workers.

## Test scripts

### A) Single request script (latency + response)

```bash
python script.py
```

What it does:
- Loads `payload.json`
- Calls `POST /predict`
- Prints HTTP status
- Prints request latency (ms)
- Prints full JSON response

### B) Full endpoint tester

```bash
python test_predict_api.py
```

What it does:
- Calls `/`, `/health`, `/predict`, `/model/info`
- Runs `/predict` multiple times (`N_RUNS` in script)
- Prints summary stats (min/max/avg latency)
- Saves full output to `test_api_output/test_run_<timestamp>.json`

## Minimal cURL example

```bash
curl -X POST "http://127.0.0.1:5500/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-real-key" \
  --data @payload.json
```

## Project structure

```text
.
├── main.py
├── script.py
├── test_predict_api.py
├── payload.json
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── example.env
├── demo-api-input-data-sample/
├── pipeline_output/
│   └── Extra_Trees_18-May-2026_12-11-08/
└── test_api_output/
```

## Notes

- Keep `.env` private; never commit secrets.
- If you switch artifact folder in `main.py`, restart the API.
- Swagger request schema updates automatically from the active feature list.
