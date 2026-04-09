#!/usr/bin/env python3
"""
Re-export quantized weights with different scale factors and plaintext moduli.

This script loads the TRAINED model (.pth file) and re-quantizes weights
at different scale factors / plaintext moduli, creating new weight directories
that the Rust benchmark can use directly.

Usage (run on EC2 #2 in the mnist_training/ directory):
    cd mnist_training
    python3 reexport_weights.py

Prerequisites:
    pip install torch torchvision numpy

Creates:
    weights_scale100/       — scale_factor=100
    weights_scale10000/     — scale_factor=10000
    weights_p65537/         — plaintext_modulus=65537
    weights_p4294967311/    — plaintext_modulus=4294967311
"""

import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── CNN architecture (must match train_mnist.py) ──
class PolynomialActivation(nn.Module):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree

    def forward(self, x):
        if self.degree == 2:
            return x ** 2
        elif self.degree == 3:
            return 0.125 * x ** 3 + 0.5 * x
        elif self.degree == 4:
            return 0.0625 * x ** 4 + 0.25 * x ** 2 + 0.1
        return x ** 2


class HE_CNN(nn.Module):
    def __init__(self, activation_degree=2):
        super().__init__()
        self.activation_degree = activation_degree
        self.conv1 = nn.Conv2d(1, 1, 5, bias=True)
        self.act = PolynomialActivation(degree=activation_degree)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(1, 1, 5, bias=True)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def quantize_and_export(model, output_dir, scale_factor, plaintext_modulus, activation_degree=2):
    """
    Quantize model weights and export to CSV + model_config.json.

    Args:
        model: Trained HE_CNN model
        output_dir: Directory to write CSV files + config
        scale_factor: Integer scale factor for quantization
        plaintext_modulus: BFV plaintext modulus
        activation_degree: Polynomial activation degree
    """
    os.makedirs(output_dir, exist_ok=True)
    state_dict = model.state_dict()
    max_half = plaintext_modulus // 2

    print(f"\n  Output: {output_dir}")
    print(f"  scale_factor={scale_factor}, plaintext_modulus={plaintext_modulus}")
    print(f"  Max representable: ±{max_half}")

    layers = [
        ("conv1.weight", "conv1_weights.csv", (5, 5)),
        ("conv1.bias",   "conv1_bias.csv",    (1, -1)),
        ("conv2.weight", "conv2_weights.csv", (5, 5)),
        ("conv2.bias",   "conv2_bias.csv",    (1, -1)),
        ("fc.weight",    "fc_weights.csv",    (10, 16)),
        ("fc.bias",      "fc_bias.csv",       (1, -1)),
    ]

    overflow = False
    for param_name, csv_name, shape in layers:
        float_w = state_dict[param_name].numpy()
        int_w = np.round(float_w * scale_factor).astype(np.int64).reshape(shape)
        max_abs = np.max(np.abs(int_w))

        status = "✓" if max_abs <= max_half else "⚠ OVERFLOW"
        if max_abs > max_half:
            overflow = True

        np.savetxt(os.path.join(output_dir, csv_name), int_w, fmt="%d", delimiter=",")
        print(f"    {csv_name:<22s}  max|w|={max_abs:>10d}  {status}")

    if overflow:
        print(f"  ⚠ WARNING: Some weights exceed ±{max_half} — expect incorrect results!")
    else:
        print(f"  ✓ All weights fit within ±{max_half}")

    # Determine activation formula
    act_formulas = {2: "x^2", 3: "0.125*x^3 + 0.5*x", 4: "0.0625*x^4 + 0.25*x^2 + 0.1"}
    act_types = {2: "PolynomialActivation(degree=2)", 3: "PolynomialActivation(degree=3)", 4: "PolynomialActivation(degree=4)"}

    config = {
        "model_name": "HE_CNN",
        "framework": "PyTorch",
        "activation_degree": activation_degree,
        "architecture": {
            "layers": [
                {"name": "conv1", "type": "Conv2d", "kernel_size": 5, "in_channels": 1, "out_channels": 1, "padding": 0},
                {"name": "act1", "type": act_types.get(activation_degree, "PolynomialActivation(degree=2)"), "formula": act_formulas.get(activation_degree, "x^2")},
                {"name": "pool1", "type": "AvgPool2d", "kernel_size": 2, "stride": 2},
                {"name": "conv2", "type": "Conv2d", "kernel_size": 5, "in_channels": 1, "out_channels": 1, "padding": 0},
                {"name": "act2", "type": act_types.get(activation_degree, "PolynomialActivation(degree=2)"), "formula": act_formulas.get(activation_degree, "x^2")},
                {"name": "pool2", "type": "AvgPool2d", "kernel_size": 2, "stride": 2},
                {"name": "fc", "type": "Linear", "in_features": 16, "out_features": 10},
            ],
            "input_shape": [1, 1, 28, 28],
            "output_shape": [1, 10],
            "total_parameters": 222,
        },
        "quantization": {
            "scale_factor": scale_factor,
            "method": f"round(weight * {scale_factor})",
            "plaintext_modulus": plaintext_modulus,
            "scheme": "BFV",
        },
        "accuracy": {
            "float_model": 88.86,
            "per_digit": {},  # Will be filled by benchmark
        },
        "files": {
            "conv1_weights": "conv1_weights.csv",
            "conv1_bias": "conv1_bias.csv",
            "conv2_weights": "conv2_weights.csv",
            "conv2_bias": "conv2_bias.csv",
            "fc_weights": "fc_weights.csv",
            "fc_bias": "fc_bias.csv",
        },
    }

    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Config saved: {config_path}")


def copy_test_images(src_dir, dst_dir):
    """Copy test image/label CSVs from source weights dir to new dir."""
    for fname in ["test_images.csv", "test_labels.csv", "test_images_100.csv", "test_labels_100.csv"]:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"    Copied: {fname}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load the trained model
    model_path = os.path.join(script_dir, "he_cnn_model.pth")
    if not os.path.exists(model_path):
        print(f"ERROR: Trained model not found at {model_path}")
        print("  Run train_mnist.py first, or check the path.")
        sys.exit(1)

    print("=" * 60)
    print("  Re-exporting weights for new experiments")
    print("=" * 60)

    model = HE_CNN(activation_degree=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"\n  Loaded model from: {model_path}")

    baseline_dir = os.path.join(script_dir, "weights_deg2")

    # ════════════════════════════════════════════════════════════════
    # Experiment 1: Scale Factor variations
    # ════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  EXPERIMENT 1: Scale Factor")
    print("═" * 60)

    scale_configs = [
        (100,   "weights_scale100"),
        (10000, "weights_scale10000"),
    ]

    for sf, dirname in scale_configs:
        out = os.path.join(script_dir, dirname)
        quantize_and_export(model, out, scale_factor=sf, plaintext_modulus=100073473, activation_degree=2)
        copy_test_images(baseline_dir, out)

    print(f"\n  Note: weights_deg2/ (scale=1000) is the baseline — no re-export needed.")

    # ════════════════════════════════════════════════════════════════
    # Experiment 2: Plaintext Modulus variations
    # ════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  EXPERIMENT 2: Plaintext Modulus")
    print("═" * 60)

    modulus_configs = [
        (65537,      "weights_p65537"),
        (4294967311, "weights_p4294967311"),
    ]

    for pm, dirname in modulus_configs:
        out = os.path.join(script_dir, dirname)
        quantize_and_export(model, out, scale_factor=1000, plaintext_modulus=pm, activation_degree=2)
        copy_test_images(baseline_dir, out)

    print(f"\n  Note: weights_deg2/ (p=100073473) is the baseline — no re-export needed.")

    # ════════════════════════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  All weight directories created!")
    print("═" * 60)
    print()

    all_dirs = [d for _, d in scale_configs] + [d for _, d in modulus_configs]
    for dirname in all_dirs:
        full = os.path.join(script_dir, dirname)
        if os.path.exists(full):
            with open(os.path.join(full, "model_config.json")) as f:
                cfg = json.load(f)
            sf = cfg["quantization"]["scale_factor"]
            pm = cfg["quantization"]["plaintext_modulus"]
            print(f"  ✓ {dirname:<30s}  scale={sf:<8d}  modulus={pm}")

    print(f"\n  Baseline (no re-export needed):")
    print(f"    weights_deg2/  scale=1000  modulus=100073473")
    print(f"\n  Next: run  ./scripts/run_new_experiments.sh")


if __name__ == "__main__":
    main()
