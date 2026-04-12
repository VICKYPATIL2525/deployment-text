# =============================================================================
# Mindspace Mental Health Classifier — API Client Script
# =============================================================================
# This script is a standalone client that calls the Mindspace Text API
# to verify that all 4 endpoints are working correctly.
#
# It reads the API key from the root .env file (same file the server uses),
# so no hardcoded secrets are needed.
#
# Endpoints tested:
#   1. GET  /            — public service info (model name, accuracy, classes)
#   2. GET  /health      — public healthcheck (artifacts loaded?)
#   3. GET  /model/info  — protected model metadata (requires X-API-Key)
#   4. POST /predict     — protected prediction (requires X-API-Key + JSON body)
#
# Usage:
#   1. Start the API server first:
#        uvicorn api_text_to_sentiment:app --host 127.0.0.1 --port 9000
#   2. Then run this script:
#        python call_text_api.py
#
# Requirements:
#   - requests       (HTTP client library)
#   - python-dotenv  (loads API_KEY from .env)
#   Both are listed in requirements.txt.
# =============================================================================

import json                  # for pretty-printing JSON responses
import os                    # for reading environment variables
from pathlib import Path     # for building the .env file path

import requests              # third-party HTTP client (much simpler than urllib)
from dotenv import load_dotenv  # loads key=value pairs from .env into os.environ

# ─── Configuration ────────────────────────────────────────────────────────────
# Load the .env file located next to this script (project root).
# This sets os.environ["API_KEY"] so we don't hardcode secrets.
load_dotenv(Path(__file__).parent / ".env")

# Base URL of the locally running API server (default port 9000).
BASE_URL = "http://127.0.0.1:9000"

# Read the API key that was just loaded from .env.
# Protected endpoints (/predict, /model/info) require this in the X-API-Key header.
API_KEY = os.getenv("API_KEY", "")

# Common headers sent with every request.
# X-API-Key authenticates against the server's verify_api_key dependency.
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}


# ─── Endpoint 1: GET / ───────────────────────────────────────────────────────
def call_root():
    """
    Call the root endpoint to get basic service info.
    This is a public endpoint — no API key required, but we send it anyway.
    Returns: service name, model name, accuracy, class list, feature count.
    """
    print("=" * 60)
    print("1. GET /  (Service Info)")
    print("=" * 60)
    resp = requests.get(f"{BASE_URL}/", headers=HEADERS)
    print(f"Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))


# ─── Endpoint 2: GET /health ─────────────────────────────────────────────────
def call_health():
    """
    Call the health check endpoint to confirm the server is ready.
    This is a public endpoint — no API key required.
    Returns: status ("ok") and whether all ML artifacts are loaded.
    """
    print("\n" + "=" * 60)
    print("2. GET /health  (Health Check)")
    print("=" * 60)
    resp = requests.get(f"{BASE_URL}/health", headers=HEADERS)
    print(f"Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))


# ─── Endpoint 3: GET /model/info ─────────────────────────────────────────────
def call_model_info():
    """
    Call the model metadata endpoint (protected — requires X-API-Key).
    Returns full training details: hyperparams, CV score, test metrics, etc.
    We print only a summary here for readability.
    """
    print("\n" + "=" * 60)
    print("3. GET /model/info  (Model Metadata)")
    print("=" * 60)
    resp = requests.get(f"{BASE_URL}/model/info", headers=HEADERS)
    print(f"Status: {resp.status_code}")
    data = resp.json()
    # Print a concise summary instead of dumping the entire metadata JSON.
    print(f"  Model        : {data.get('best_model_name')}")
    print(f"  Accuracy     : {data.get('test_metrics', {}).get('accuracy')}")
    print(f"  Classes      : {data.get('class_names')}")
    print(f"  Num Features : {data.get('n_features')}")


# ─── Endpoint 4: POST /predict ───────────────────────────────────────────────
def call_predict():
    """
    Call the prediction endpoint (protected — requires X-API-Key).
    Sends a sample JSON payload with 43 pre-extracted text features.
    Returns: predicted class, confidence, per-class probabilities.
    """
    print("\n" + "=" * 60)
    print("4. POST /predict  (Prediction)")
    print("=" * 60)

    # Load one of the demo sample files shipped with the project.
    # Each file contains all 43 features the model expects.
    with open("demo-api-input-data-sample/depression_sample_1.json") as f:
        payload = json.load(f)

    # Send the features as JSON body to the /predict endpoint.
    resp = requests.post(f"{BASE_URL}/predict", headers=HEADERS, json=payload)
    print(f"Status: {resp.status_code}")

    # Parse and display the prediction result.
    result = resp.json()
    print(f"  Prediction   : {result.get('prediction')}")
    print(f"  Confidence   : {result.get('confidence')}")
    print(f"  Model        : {result.get('model')}")
    print(f"  Accuracy     : {result.get('accuracy')}")
    print(f"  Probabilities:")
    for cls, prob in result.get("probabilities", {}).items():
        print(f"    {cls:20s} : {prob}")


# ─── Main ─────────────────────────────────────────────────────────────────────
# Call all 4 endpoints in sequence and print results.
if __name__ == "__main__":
    call_root()
    call_health()
    call_model_info()
    call_predict()
    print("\n" + "=" * 60)
    print("All endpoints called successfully.")
    print("=" * 60)
