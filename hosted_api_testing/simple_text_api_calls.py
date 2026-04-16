import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load API key from root .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = os.getenv("TEXT_API_BASE_URL", "http://88.222.212.15:9000")
API_KEY = os.getenv("API_KEY", "")

# Public endpoints
PUBLIC_HEADERS = {"Content-Type": "application/json"}

# Protected endpoints
PROTECTED_HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}


def show_response(name: str, resp: requests.Response) -> None:
    print("\n" + "=" * 60)
    print(name)
    print("Status:", resp.status_code)
    try:
        print(json.dumps(resp.json(), indent=2))
    except ValueError:
        print(resp.text)


if __name__ == "__main__":
    sample_file = Path(__file__).resolve().parent.parent / "demo-api-input-data-sample" / "depression_sample_1.json"
    with sample_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    print("Base URL:", BASE_URL)
    if not API_KEY:
        print("Warning: API_KEY is empty. Protected calls may fail.")

    r1 = requests.get(f"{BASE_URL}/", headers=PUBLIC_HEADERS, timeout=20)
    show_response("GET /", r1)

    r2 = requests.get(f"{BASE_URL}/health", headers=PUBLIC_HEADERS, timeout=20)
    show_response("GET /health", r2)

    r3 = requests.get(f"{BASE_URL}/model/info", headers=PROTECTED_HEADERS, timeout=20)
    show_response("GET /model/info", r3)

    r4 = requests.post(f"{BASE_URL}/predict", headers=PROTECTED_HEADERS, json=payload, timeout=30)
    show_response("POST /predict", r4)
