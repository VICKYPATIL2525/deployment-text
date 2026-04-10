# Mindspace Text API (Quick README)

FastAPI service for mental health profile classification using a trained LightGBM model.

- Model: LightGBM
- Input: 43 pre-extracted text/speech features
- Output: predicted class + confidence + class probabilities
- Classes: Anxiety, Bipolar_Mania, Depression, Normal, Phobia, Stress, Suicidal_Tendency
- Default port: 9000

## Project Files

- `api_text_to_sentiment.py`: FastAPI app and inference logic
- `pipeline_output/LightGBM_13032026_110356/`: model + preprocessing artifacts
- `demo-api-input-data-sample/`: sample JSON payloads
- `Dockerfile`: container build
- `docker-compose.yml`: local container orchestration
- `requirements.txt`: Python dependencies

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

- Auth: Not required
- Purpose: quick check of loaded model summary

Example response (shape):

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

- Auth: Not required
- Purpose: readiness/liveness

Example response:

```json
{
  "status": "ok",
  "artifacts_loaded": true
}
```

### 3) POST /predict

Main inference endpoint.

- Auth: Required (`X-API-Key`)
- Body: JSON object with all 43 required numeric features
- Returns: prediction, confidence, probabilities, model, accuracy

Success response (shape):

```json
{
  "prediction": "Depression",
  "confidence": 0.9412,
  "probabilities": {
    "Anxiety": 0.0101,
    "Bipolar_Mania": 0.0021,
    "Depression": 0.9412,
    "Normal": 0.015,
    "Phobia": 0.0061,
    "Stress": 0.022,
    "Suicidal_Tendency": 0.0035
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

- Auth: Required (`X-API-Key`)
- Purpose: inspect model params and training/test metrics

## Predict Request Example

Use any sample file from `demo-api-input-data-sample/` such as `depression_sample_1.json`.

### cURL

```bash
curl -X POST "http://localhost:9000/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  --data-binary "@demo-api-input-data-sample/depression_sample_1.json"
```

### PowerShell

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
2. Start API.
3. Open `http://localhost:9000/health` and confirm `status: ok`.
4. Send one sample JSON to `/predict` with `X-API-Key`.
5. Verify prediction payload is returned.
