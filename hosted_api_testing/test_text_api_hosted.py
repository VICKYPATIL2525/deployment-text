# =============================================================================
# Mindspace Text Classifier - Hosted API Test Client
# =============================================================================
# Purpose:
#   Test the deployed Text Classifier API (port 9000 on VPS) without touching
#   existing local-test files.
#
# References used:
#   - API Repos.pdf
#   - Mindspace_API_Integration_Guide.md (API 4 section)
#
# Hosted endpoint from integration guide:
#   http://88.222.212.15:9000
#
# Auth header for API 4:
#   X-API-Key
#
# Usage:
#   1) Ensure root .env contains API_KEY for deployed API.
#   2) Optional override: set TEXT_API_BASE_URL to another URL.
#   3) Run:
#        python hosted_api_testing/test_text_api_hosted.py
# =============================================================================

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load API_KEY from project root .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Default to hosted URL; can be overridden for local or staging checks.
BASE_URL = os.getenv("TEXT_API_BASE_URL", "http://88.222.212.15:9000")
API_KEY = os.getenv("API_KEY", "")

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}


def call_root() -> None:
    print("=" * 60)
    print("1. GET /  (Service Info)")
    print("=" * 60)
    resp = requests.get(f"{BASE_URL}/", headers=HEADERS, timeout=20)
    print(f"Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))


def call_health() -> None:
    print("\n" + "=" * 60)
    print("2. GET /health  (Health Check)")
    print("=" * 60)
    resp = requests.get(f"{BASE_URL}/health", headers=HEADERS, timeout=20)
    print(f"Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))


def call_model_info() -> None:
    print("\n" + "=" * 60)
    print("3. GET /model/info  (Model Metadata)")
    print("=" * 60)
    resp = requests.get(f"{BASE_URL}/model/info", headers=HEADERS, timeout=20)
    print(f"Status: {resp.status_code}")
    data = resp.json()
    print(f"  Model        : {data.get('best_model_name')}")
    print(f"  Accuracy     : {data.get('test_metrics', {}).get('accuracy')}")
    print(f"  Classes      : {data.get('class_names')}")
    print(f"  Num Features : {data.get('n_features')}")


def call_predict() -> None:
    print("\n" + "=" * 60)
    print("4. POST /predict  (Prediction)")
    print("=" * 60)

    sample_file = Path(__file__).resolve().parent.parent / "demo-api-input-data-sample" / "depression_sample_1.json"
    with sample_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    resp = requests.post(f"{BASE_URL}/predict", headers=HEADERS, json=payload, timeout=30)
    print(f"Status: {resp.status_code}")

    result = resp.json()
    print(f"  Prediction   : {result.get('prediction')}")
    print(f"  Confidence   : {result.get('confidence')}")
    print(f"  Model        : {result.get('model')}")
    print(f"  Accuracy     : {result.get('accuracy')}")
    print("  Probabilities:")
    for cls, prob in result.get("probabilities", {}).items():
        print(f"    {cls:20s} : {prob}")


if __name__ == "__main__":
    print(f"Target API Base URL: {BASE_URL}")
    if not API_KEY:
        print("WARNING: API_KEY is empty. Protected endpoints may return 401/403.")

    call_root()
    call_health()
    call_model_info()
    call_predict()

    print("\n" + "=" * 60)
    print("Hosted API test flow completed.")
    print("=" * 60)
