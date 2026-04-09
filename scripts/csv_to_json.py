#!/usr/bin/env python3
"""
Convert FHE benchmark CSV results → JSON for the React frontend.

Usage:
    python scripts/csv_to_json.py

Reads from:  mnist_training/fhe_test_results_deg{2,3,4}_128bit.csv
Writes to:   frontend/public/benchmark_data.json

The JSON contains per-image results AND pre-computed summaries so the
frontend doesn't need to do any heavy computation.
"""

import csv
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "mnist_training"
OUT_FILE = ROOT / "frontend" / "public" / "benchmark_data.json"

CONFIGS = [
    {"degree": 2, "file": "fhe_test_results_deg2_128bit.csv", "label": "x² (degree 2)"},
    {"degree": 3, "file": "fhe_test_results_deg3_128bit.csv", "label": "x³ (degree 3)"},
    {"degree": 4, "file": "fhe_test_results_deg4_128bit.csv", "label": "x⁴ (degree 4)"},
]

TIMING_KEYS = [
    "encryption_ms", "conv1_ms", "act1_ms", "pool1_ms",
    "conv2_ms", "act2_ms", "pool2_ms", "fc_ms", "decryption_ms",
]

LAYER_LABELS = [
    "Encryption", "Conv1", "Activation1", "Pool1",
    "Conv2", "Activation2", "Pool2", "FC", "Decryption",
]


def load_csv(filepath):
    """Load a benchmark CSV and return list of row dicts with typed values."""
    rows = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            typed = {
                "image_index": int(row["image_index"]),
                "true_label": int(row["true_label"]),
                "predicted": int(row["predicted"]),
                "correct": row["correct"] == "True",
                "confidence": round(float(row["confidence"]), 4),
                "total_ms": round(float(row["total_ms"]), 1),
                "status": row["status"],
            }
            # Add per-layer timings
            for key in TIMING_KEYS:
                typed[key] = round(float(row[key]), 1)
            rows.append(typed)
    return rows


def compute_summary(rows, config):
    """Compute aggregate statistics for a set of benchmark rows."""
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    totals = [r["total_ms"] for r in rows]

    # Per-layer averages
    layer_avgs = {}
    for key in TIMING_KEYS:
        vals = [r[key] for r in rows]
        layer_avgs[key] = round(statistics.mean(vals), 1)

    # Per-digit accuracy
    digit_accuracy = {}
    for digit in range(10):
        digit_rows = [r for r in rows if r["true_label"] == digit]
        if digit_rows:
            digit_correct = sum(1 for r in digit_rows if r["correct"])
            digit_accuracy[str(digit)] = {
                "correct": digit_correct,
                "total": len(digit_rows),
                "pct": round(100 * digit_correct / len(digit_rows), 1),
            }

    return {
        "degree": config["degree"],
        "label": config["label"],
        "security": "128-bit",
        "num_images": n,
        "accuracy": f"{correct}/{n}",
        "accuracy_pct": round(100 * correct / n, 1),
        "avg_total_ms": round(statistics.mean(totals), 1),
        "median_total_ms": round(statistics.median(totals), 1),
        "min_total_ms": round(min(totals), 1),
        "max_total_ms": round(max(totals), 1),
        "std_total_ms": round(statistics.stdev(totals), 1) if n > 1 else 0,
        "layer_averages": layer_avgs,
        "layer_labels": LAYER_LABELS,
        "per_digit_accuracy": digit_accuracy,
    }


def main():
    output = {
        "generated_at": "auto",
        "description": "FHE CNN benchmark results — MNIST digit classification with OpenFHE BFV",
        "scheme": "BFV",
        "library": "OpenFHE v1.2.2",
        "plaintext_modulus": 100073473,
        "scale_factor": 1000,
        "configs": [],
        "raw_results": {},
    }

    for config in CONFIGS:
        csv_path = CSV_DIR / config["file"]
        if not csv_path.exists():
            print(f"⚠ Skipping {config['file']} — file not found")
            continue

        rows = load_csv(csv_path)
        if not rows:
            print(f"⚠ Skipping {config['file']} — file is empty (0 rows)")
            continue

        summary = compute_summary(rows, config)
        output["configs"].append(summary)
        output["raw_results"][f"deg{config['degree']}"] = rows
        print(f"✓ deg{config['degree']}: {len(rows)} images, "
              f"accuracy={summary['accuracy']} ({summary['accuracy_pct']}%), "
              f"avg={summary['avg_total_ms']:.0f}ms")

    # Add total images count
    from datetime import datetime
    output["generated_at"] = datetime.now().isoformat()
    total_images = sum(c["num_images"] for c in output["configs"])
    output["total_images"] = total_images

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Written to {OUT_FILE}")
    print(f"   {len(output['configs'])} configs, {total_images} total images")


if __name__ == "__main__":
    main()
