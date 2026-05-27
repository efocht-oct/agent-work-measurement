#!/usr/bin/env python3
"""Generate deterministic synthetic log data for task3_log_analyzer."""
import json
import random
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "log.jsonl")

random.seed(42)  # deterministic

LEVELS = ["INFO", "WARN", "ERROR"]
STATUS_CODES = [200, 200, 200, 200, 200, 201, 204, 301, 400, 404, 500, 502, 503]

lines = []
for i in range(10000):
    hour = i % 24
    minute = (i * 7) % 60
    second = (i * 13) % 60
    ts = f"2026-01-{(i % 28)+1:02d}T{hour:02d}:{minute:02d}:{second:02d}Z"

    level = random.choices(LEVELS, weights=[70, 20, 10])[0]
    status = random.choice(STATUS_CODES)

    if level == "ERROR":
        response_time = random.uniform(1000, 10000)
    elif level == "WARN":
        response_time = random.uniform(200, 2000)
    else:
        response_time = random.uniform(10, 500)

    record = {
        "timestamp": ts,
        "level": level,
        "response_time_ms": round(response_time, 2),
        "status": status,
        "request_id": f"req-{i:06d}",
    }
    lines.append(json.dumps(record))

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Generated {len(lines)} log entries to {OUTPUT_FILE}")
