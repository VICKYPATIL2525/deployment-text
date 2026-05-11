import json
import os
import sys
import time
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

HOST    = os.environ.get("MINDSPACE_TEXT_HOST", "http://127.0.0.1:9000")
API_KEY = os.environ.get("MINDSPACE_TEXT_API_KEY", "")

if not API_KEY:
    print("ERROR: MINDSPACE_TEXT_API_KEY is not set in .env")
    sys.exit(1)

HEADERS = {"X-API-Key": API_KEY}
PAYLOAD = json.loads((Path(__file__).parent / "payload.json").read_text())

OUTPUT_DIR = Path(__file__).parent / "test_api_output"
OUTPUT_DIR.mkdir(exist_ok=True)

results = {}


def call_and_record(label, method, url, **kwargs):
    t0 = time.perf_counter()
    r = method(url, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"  HTTP {r.status_code}  |  {elapsed_ms:.0f} ms")
    print(f"{'='*55}")
    try:
        body = r.json()
        print(json.dumps(body, indent=2))
    except Exception:
        body = r.text
        print(body)

    results[label] = {
        "status_code": r.status_code,
        "elapsed_ms": round(elapsed_ms, 2),
        "response": body,
    }


print(f"\nTarget: {HOST}")
print(f"API Key: {API_KEY[:8]}...")

# ── GET / ─────────────────────────────────────────────────────
call_and_record(
    "GET /  — service info",
    requests.get, f"{HOST}/",
    headers=HEADERS, timeout=10,
)

# ── GET /health ───────────────────────────────────────────────
call_and_record(
    "GET /health  — health check (no auth)",
    requests.get, f"{HOST}/health",
    timeout=10,
)

# ── POST /predict ─────────────────────────────────────────────
call_and_record(
    "POST /predict  — prediction",
    requests.post, f"{HOST}/predict",
    json=PAYLOAD, headers=HEADERS, timeout=30,
)

# ── GET /model/info ───────────────────────────────────────────
t0 = time.perf_counter()
r = requests.get(f"{HOST}/model/info", headers=HEADERS, timeout=10)
elapsed_ms = (time.perf_counter() - t0) * 1000
data = r.json()
display_data = dict(data)
if "feature_names" in display_data:
    display_data["feature_names"] = f"[... {len(display_data['feature_names'])} features ...]"
print(f"\n{'='*55}")
print(f"  GET /model/info  — model structure")
print(f"  HTTP {r.status_code}  |  {elapsed_ms:.0f} ms")
print(f"{'='*55}")
print(json.dumps(display_data, indent=2))
results["GET /model/info  — model structure"] = {
    "status_code": r.status_code,
    "elapsed_ms": round(elapsed_ms, 2),
    "response": data,
}

# ── Save output ───────────────────────────────────────────────
timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
output_file = OUTPUT_DIR / f"test_run_{timestamp}.json"
output_file.write_text(json.dumps({
    "run_at": datetime.now().isoformat(),
    "host": HOST,
    "payload": PAYLOAD,
    "results": results,
}, indent=2))

print(f"\n✓ Output saved → {output_file}")

