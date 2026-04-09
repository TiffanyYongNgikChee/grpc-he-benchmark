#!/usr/bin/env python3
"""
Convert NEW experiment benchmark CSVs → JSON for the React frontend.

Generates TWO JSON files:
  1. frontend/public/scale_factor_data.json   — Scale Factor experiment
  2. frontend/public/plaintext_modulus_data.json — Plaintext Modulus experiment

Usage:
    python3 scripts/csv_to_json_experiments.py

Reads from:  mnist_training/fhe_benchmark_{scale100,scale1000,scale10000}_128bit.csv
             mnist_training/fhe_benchmark_{p65537,p100073473,p4294967311}_128bit.csv
"""

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "mnist_training"
OUT_DIR = ROOT / "frontend" / "public"

TIMING_KEYS = [
    "encryption_ms", "conv1_ms", "act1_ms", "pool1_ms",
    "conv2_ms", "act2_ms", "pool2_ms", "fc_ms", "decryption_ms",
]

LAYER_LABELS = [
    "Encryption", "Conv1", "Activation1", "Pool1",
    "Conv2", "Activation2", "Pool2", "FC", "Decryption",
]

# ── Scale Factor configs ──
SCALE_CONFIGS = [
    {
        "scale_factor": 100,
        "file": "fhe_benchmark_scale100_128bit.csv",
        "label": "S = 100 (low precision)",
    },
    {
        "scale_factor": 1000,
        "file": "fhe_benchmark_scale1000_128bit.csv",
        "label": "S = 1,000 (baseline)",
    },
    {
        "scale_factor": 10000,
        "file": "fhe_benchmark_scale10000_128bit.csv",
        "label": "S = 10,000 (high precision)",
    },
]

# ── Plaintext Modulus configs ──
MODULUS_CONFIGS = [
    {
        "plaintext_modulus": 65537,
        "file": "fhe_benchmark_p65537_128bit.csv",
        "label": "p = 65,537 (16-bit)",
    },
    {
        "plaintext_modulus": 100073473,
        "file": "fhe_benchmark_p100073473_128bit.csv",
        "label": "p = 100,073,473 (baseline)",
    },
    {
        "plaintext_modulus": 4294967311,
        "file": "fhe_benchmark_p4294967311_128bit.csv",
        "label": "p = 4,294,967,311 (32-bit)",
    },
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
                "predicted": int(row.get("predicted_digit", row.get("predicted", 0))),
                "correct": row["correct"] in ("True", "1", "true"),
                "total_ms": round(float(row["total_ms"]), 1),
            }
            for key in TIMING_KEYS:
                typed[key] = round(float(row.get(key, 0)), 1)
            rows.append(typed)
    return rows


def compute_summary(rows, config):
    """Compute aggregate statistics for a set of benchmark rows."""
    n = len(rows)
    if n == 0:
        return None

    correct = sum(1 for r in rows if r["correct"])
    totals = [r["total_ms"] for r in rows]

    layer_avgs = {}
    for key in TIMING_KEYS:
        vals = [r[key] for r in rows]
        layer_avgs[key] = round(statistics.mean(vals), 1)

    per_digit = {}
    for digit in range(10):
        digit_rows = [r for r in rows if r["true_label"] == digit]
        if digit_rows:
            dc = sum(1 for r in digit_rows if r["correct"])
            per_digit[str(digit)] = {
                "correct": dc,
                "total": len(digit_rows),
                "pct": round(100 * dc / len(digit_rows), 1),
            }

    return {
        **{k: v for k, v in config.items() if k != "file"},
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
        "per_digit_accuracy": per_digit,
    }


def process_experiment(configs, experiment_name, extra_meta=None):
    """Process a set of configs and write JSON."""
    output = {
        "generated_at": datetime.now().isoformat(),
        "experiment": experiment_name,
        "description": f"FHE CNN benchmark — {experiment_name} comparison",
        "scheme": "BFV",
        "library": "OpenFHE v1.2.2",
        "activation_degree": 2,
        "configs": [],
        "raw_results": {},
    }
    if extra_meta:
        output.update(extra_meta)

    for config in configs:
        csv_path = CSV_DIR / config["file"]
        if not csv_path.exists():
            print(f"  ⚠ Skipping {config['file']} — not found")
            continue

        rows = load_csv(csv_path)
        summary = compute_summary(rows, config)
        if summary:
            output["configs"].append(summary)
            key = config["file"].replace("fhe_benchmark_", "").replace("_128bit.csv", "")
            output["raw_results"][key] = rows
            print(f"  ✓ {config['label']}: {len(rows)} images, "
                  f"accuracy={summary['accuracy']} ({summary['accuracy_pct']}%), "
                  f"avg={summary['avg_total_ms']:.0f}ms")

    total_images = sum(c["num_images"] for c in output["configs"])
    output["total_images"] = total_images

    return output


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Scale Factor ──
    print("\n══ Scale Factor Experiment ══")
    scale_data = process_experiment(
        SCALE_CONFIGS,
        "Scale Factor",
        {"plaintext_modulus": 100073473},
    )
    scale_out = OUT_DIR / "scale_factor_data.json"
    with open(scale_out, "w") as f:
        json.dump(scale_data, f, indent=2)
    print(f"  📄 Written to {scale_out}")

    # ── Plaintext Modulus ──
    print("\n══ Plaintext Modulus Experiment ══")
    modulus_data = process_experiment(
        MODULUS_CONFIGS,
        "Plaintext Modulus",
        {"scale_factor": 1000},
    )
    modulus_out = OUT_DIR / "plaintext_modulus_data.json"
    with open(modulus_out, "w") as f:
        json.dump(modulus_data, f, indent=2)
    print(f"  📄 Written to {modulus_out}")

    print(f"\n✅ Done! Copy JSON files to frontend EC2:")
    print(f"   scp {scale_out} {modulus_out} <frontend-ec2>:~/frontend/build/")


if __name__ == "__main__":
    main()
