import json
import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

HOST    = os.environ.get("MINDSPACE_TEXT_HOST", "http://localhost:9000")
API_KEY = os.environ.get("MINDSPACE_TEXT_API_KEY", "")

if not API_KEY:
    print("ERROR: MINDSPACE_TEXT_API_KEY is not set in .env")
    sys.exit(1)

HEADERS = {"X-API-Key": API_KEY}
PAYLOAD = json.loads((Path(__file__).parent / "payload.json").read_text())


def print_result(label, response):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"  HTTP {response.status_code}")
    print(f"{'='*55}")
    try:
        print(json.dumps(response.json(), indent=2))
    except Exception:
        print(response.text)


print(f"\nTarget: {HOST}")
print(f"API Key: {API_KEY[:8]}...")

# ── GET / ─────────────────────────────────────────────────────
r = requests.get(f"{HOST}/", headers=HEADERS, timeout=10)
print_result("GET /  — service info", r)

# ── GET /health ───────────────────────────────────────────────
r = requests.get(f"{HOST}/health", headers=HEADERS, timeout=10)
print_result("GET /health  — health check", r)

# ── POST /predict ─────────────────────────────────────────────
r = requests.post(f"{HOST}/predict", json=PAYLOAD, headers=HEADERS, timeout=30)
print_result("POST /predict  — prediction", r)

# ── GET /model/info ───────────────────────────────────────────
r = requests.get(f"{HOST}/model/info", headers=HEADERS, timeout=10)
data = r.json()
# feature_names list — summarise to keep output readable
if "feature_names" in data:
    data["feature_names"] = f"[... {len(data['feature_names'])} features ...]"
print(f"\n{'='*55}")
print(f"  GET /model/info  — model structure")
print(f"  HTTP {r.status_code}")
print(f"{'='*55}")
print(json.dumps(data, indent=2))
