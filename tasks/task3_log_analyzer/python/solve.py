#!/usr/bin/env python3
"""JSON-lines log analyzer.
Parses a log.jsonl file, computes statistics on duration_ms,
and produces a summary with per-level breakdown.
"""
import argparse
import json
import sys


def compute_stats(values):
    """Compute count, mean, stddev, p50, p95, p99, min, max."""
    if not values:
        return {}
    n = len(values)
    sv = sorted(values)
    mean = sum(sv) / n
    variance = sum((x - mean) ** 2 for x in sv) / n
    stddev = variance ** 0.5

    def percentile(p):
        k = (p / 100.0) * (n - 1)
        f = int(k)
        c = min(f + 1, n - 1)
        if f == c:
            return sv[f]
        return sv[f] * (c - k) + sv[c] * (k - f)

    return {
        "count": n,
        "mean": mean,
        "stddev": stddev,
        "p50": percentile(50),
        "p95": percentile(95),
        "p99": percentile(99),
        "min": sv[0],
        "max": sv[-1],
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze JSON-lines log files.")
    parser.add_argument("file", help="Path to log.jsonl file")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--level", "-l", help="Filter by level")
    parser.add_argument("--top-n", "-n", type=int, default=10, help="Number of slowest entries")
    parser.add_argument("--format", "-f", choices=["json", "text"], default="json")
    args = parser.parse_args()

    entries = []
    with open(args.file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    total_entries = len(entries)

    if args.level:
        entries = [e for e in entries if e.get("level") == args.level]

    # Level counts
    level_counts = {}
    for e in entries:
        lvl = e.get("level", "UNKNOWN")
        level_counts[lvl] = level_counts.get(lvl, 0) + 1

    # Service counts
    service_counts = {}
    for e in entries:
        svc = e.get("service", "UNKNOWN")
        service_counts[svc] = service_counts.get(svc, 0) + 1

    # Per-service error counts
    per_service_errors = {}
    for e in entries:
        if e.get("level") == "ERROR":
            svc = e.get("service", "UNKNOWN")
            per_service_errors[svc] = per_service_errors.get(svc, 0) + 1

    # Top slowest
    all_times = []
    for e in entries:
        dur = e.get("duration_ms", 0)
        all_times.append((dur, e))
    all_times.sort(key=lambda x: x[0], reverse=True)
    slowest = [{"duration_ms": d, **e} for d, e in all_times[:args.top_n]]

    result = {
        "total_entries": total_entries,
        "level_counts": level_counts,
        "service_counts": service_counts,
        "per_service_error_counts": per_service_errors,
        "slowest_entries": slowest,
    }

    output = json.dumps(result, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
