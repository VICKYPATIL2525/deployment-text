# Mindspace Text API

FastAPI service for mental health profile classification using a trained LightGBM model.

- **Model**: LightGBM
- **Input**: 43 pre-extracted text/speech features
- **Output**: predicted class + confidence + class probabilities
- **Classes**: Anxiety, Bipolar_Mania, Depression, Normal, Phobia, Stress, Suicidal_Tendency
- **Default port**: 9000

## Project Files

| File / Folder | Purpose |
|---|---|
| `api_text_to_sentiment.py` | FastAPI app — all endpoints + inference logic |
| `call_text_api.py` | Python client script — calls all 4 endpoints for testing |
| `pipeline_output/LightGBM_13032026_110356/` | Trained model + preprocessing artifacts |
| `demo-api-input-data-sample/` | Sample JSON payloads (5 per class, 7 classes) |
| `Dockerfile` | Container image build |
| `docker-compose.yml` | Local container orchestration |
| `requirements.txt` | Python dependencies (pinned versions) |
| `.env` | API key (not committed — listed in `.gitignore`) |
| `.gitignore` | Keeps `.env` and other local files out of version control |

## Prerequisites

- Python 3.11+
- pip
- Optional: Docker + Docker Compose

## Environment Variable

Create a `.env` file in the project root:

```env
API_KEY=your-secret-api-key
```

The `/predict` and `/model/info` endpoints require this key in the `X-API-Key` header.
The client script (`call_text_api.py`) reads the same `.env` file automatically.

## Run Locally (Python)

```bash
pip install -r requirements.txt
uvicorn api_text_to_sentiment:app --host 0.0.0.0 --port 9000 --reload
```

Open docs:

- Swagger UI: http://localhost:9000/docs
- ReDoc: http://localhost:9000/redoc

## Run with Docker Compose

```bash
docker compose up --build
```

Stop:

```bash
docker compose down
```

## Endpoints

### 1) GET /

Public service info endpoint.

- **Auth**: Not required
- **Purpose**: quick check of loaded model summary

Example response:

```json
{
  "service": "Mindspace Mental Health Classifier",
  "model": "LightGBM",
  "accuracy": 0.92,
  "classes": ["Anxiety", "Bipolar_Mania", "Depression", "Normal", "Phobia", "Stress", "Suicidal_Tendency"],
  "n_features": 43
}
```

### 2) GET /health

Public healthcheck endpoint.

- **Auth**: Not required
- **Purpose**: readiness/liveness probe

Example response:

```json
{
  "status": "ok",
  "artifacts_loaded": true
}
```

### 3) POST /predict

Main inference endpoint.

- **Auth**: Required (`X-API-Key` header)
- **Body**: JSON object with all 43 required numeric features
- **Returns**: prediction, confidence, probabilities, model, accuracy

Example response:

```json
{
  "prediction": "Depression",
  "confidence": 0.9999,
  "probabilities": {
    "Anxiety": 0.0,
    "Bipolar_Mania": 0.0,
    "Depression": 0.9999,
    "Normal": 0.0,
    "Phobia": 0.0,
    "Stress": 0.0001,
    "Suicidal_Tendency": 0.0
  },
  "model": "LightGBM",
  "accuracy": 0.92
}
```

Common errors:

- `403`: missing/invalid API key
- `422`: validation/preprocessing error in request body
- `500`: inference failure

### 4) GET /model/info

Returns full model metadata.

- **Auth**: Required (`X-API-Key` header)
- **Purpose**: inspect model hyperparams, CV score, and test metrics

## Testing the API

### Option 1: Python client script

The easiest way to test all endpoints at once:

```bash
# Terminal 1 — start the server
uvicorn api_text_to_sentiment:app --host 127.0.0.1 --port 9000

# Terminal 2 — run the client
python call_text_api.py
```

The script reads the API key from `.env`, calls all 4 endpoints, and prints formatted results.

### Option 2: cURL

```bash
curl -X POST "http://localhost:9000/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  --data-binary "@demo-api-input-data-sample/depression_sample_1.json"
```

### Option 3: PowerShell

```powershell
$headers = @{ "X-API-Key" = "your-secret-api-key" }
$body = Get-Content .\demo-api-input-data-sample\depression_sample_1.json -Raw
Invoke-RestMethod -Uri "http://localhost:9000/predict" -Method Post -Headers $headers -ContentType "application/json" -Body $body
```

## Required Input Notes

- All 43 fields are required.
- `language_hindi` and `language_marathi` must be `0.0` or `1.0`.
- Topic weights (`topic_0` to `topic_4`) must be in `[0, 1]`.
- Typical shape is easiest to follow from sample JSON files in `demo-api-input-data-sample/`.

## Quick Test Flow

1. Create `.env` with `API_KEY`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Start API: `uvicorn api_text_to_sentiment:app --host 127.0.0.1 --port 9000`.
4. Run `python call_text_api.py` — should print all 4 endpoint results.
5. Or open `http://localhost:9000/docs` for interactive Swagger UI.
