import json
import os
import sys
import time
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

HOST    = os.environ.get("MINDSPACE_TEXT_HOST", "http://127.0.0.1:5500")
API_KEY = os.environ.get("MINDSPACE_TEXT_API_KEY", "")

# ── Repeat control ────────────────────────────────────────────────────────────
# Set N_RUNS to any positive integer to call POST /predict that many times.
# Default is 1 (single run, same behaviour as before).
#
# What increasing N_RUNS helps you test / understand:
#   • Latency consistency  — see if response times are stable or spike
#   • Warm vs cold cache   — first call loads CPU/memory caches; later calls
#                            are usually faster (shows the "hot path" speed)
#   • Throughput estimate  — avg ms per call × N gives a rough calls/sec ceiling
#   • Model determinism    — every run must return the same prediction and
#                            probabilities (confirms no randomness at inference)
#   • Server stability     — quickly spot memory leaks or slowdowns under load
N_RUNS = 50   # ← change this to e.g. 10 or 50 to run multiple prediction calls

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
predict_runs = []

for run_idx in range(1, N_RUNS + 1):
    run_label = f"POST /predict  — prediction (run {run_idx}/{N_RUNS})"
    call_and_record(
        run_label,
        requests.post, f"{HOST}/predict",
        json=PAYLOAD, headers=HEADERS, timeout=30,
    )
    predict_runs.append(results[run_label])

# ── /predict summary (only printed when N_RUNS > 1) ──────────
if N_RUNS > 1:
    times = [r["elapsed_ms"] for r in predict_runs]
    predictions = [r["response"].get("prediction") for r in predict_runs if isinstance(r["response"], dict)]
    consistent = len(set(predictions)) == 1

    print(f"\n{'='*55}")
    print(f"  POST /predict  — summary over {N_RUNS} runs")
    print(f"{'='*55}")
    print(f"  Min   : {min(times):.0f} ms")
    print(f"  Max   : {max(times):.0f} ms")
    print(f"  Avg   : {sum(times)/len(times):.0f} ms")
    print(f"  First : {times[0]:.0f} ms  (cold)   Last: {times[-1]:.0f} ms  (warm)")
    print(f"  Consistent prediction : {'YES ✓' if consistent else 'NO ✗  — check model!'}")
    if consistent:
        print(f"  Prediction : {predictions[0]}")

    results["POST /predict  — summary"] = {
        "n_runs": N_RUNS,
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "avg_ms": round(sum(times) / len(times), 2),
        "cold_ms": round(times[0], 2),
        "warm_ms": round(times[-1], 2),
        "consistent_prediction": consistent,
        "prediction": predictions[0] if consistent else list(set(predictions)),
        "all_times_ms": [round(t, 2) for t in times],
    }
else:
    # Single run — keep original flat key for backward compatibility
    results["POST /predict  — prediction"] = predict_runs[0]

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

