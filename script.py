import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

HOST = os.environ.get("MINDSPACE_TEXT_HOST", "http://127.0.0.1:5500")
API_KEY = os.environ.get("MINDSPACE_TEXT_API_KEY", "")

if not API_KEY:
    print("ERROR: MINDSPACE_TEXT_API_KEY is not set in .env")
    exit(1)

payload = json.loads((Path(__file__).parent / "payload.json").read_text())

t0 = time.perf_counter()
resp = requests.post(
    f"{HOST}/predict",
    json=payload,
    headers={"X-API-Key": API_KEY},
    timeout=30,
)
latency_ms = (time.perf_counter() - t0) * 1000

print(f"Status: {resp.status_code}")
print(f"Latency: {latency_ms:.1f} ms")
print(json.dumps(resp.json(), indent=2))
