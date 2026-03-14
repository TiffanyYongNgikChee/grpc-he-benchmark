#!/usr/bin/env python3
"""
Plaintext integer CNN simulation — reproduces the EXACT same arithmetic
as openfhe_cnn_ops.cpp to find the bug causing 0% accuracy.

This runs the same pipeline as the HE code but without encryption,
using pure Python integer math. If this gets ~87% accuracy, the bug
is in the HE slot layout. If this also gets 0%, the bug is in the
integer arithmetic / scale management.

Pipeline:
  Input 28×28 (scaled pixels) 
  → conv2d(5×5, ÷S) → +bias → x²/S → avgpool(2×2)
  → conv2d(5×5, ÷S) → +bias → x²/S → avgpool(2×2)  
  → matmul(10×16, ÷S) → +bias → argmax
"""

import csv
import os
import json
import sys

def load_weights(weights_dir):
    """Load all weights from CSV files."""
    def load_csv(filename):
        path = os.path.join(weights_dir, filename)
        values = []
        with open(path) as f:
            for line in f:
                for tok in line.strip().split(","):
                    tok = tok.strip()
                    if tok:
                        values.append(int(tok))
        return values
    
    return {
        "conv1_kernel": load_csv("conv1_weights.csv"),   # 25 values (5×5)
        "conv1_bias": load_csv("conv1_bias.csv"),         # 1 value
        "conv2_kernel": load_csv("conv2_weights.csv"),   # 25 values (5×5)
        "conv2_bias": load_csv("conv2_bias.csv"),         # 1 value
        "fc_weights": load_csv("fc_weights.csv"),         # 160 values (10×16)
        "fc_bias": load_csv("fc_bias.csv"),               # 10 values
    }


def load_test_data(weights_dir):
    """Load test images and labels."""
    images = []
    with open(os.path.join(weights_dir, "test_images.csv")) as f:
        reader = csv.reader(f)
        for row in reader:
            images.append([int(x) for x in row])
    
    with open(os.path.join(weights_dir, "test_labels.csv")) as f:
        labels = [int(x) for x in f.readline().strip().split(",")]
    
    return images, labels


def scale_pixels(pixels, scale_factor):
    """Same as Rust: round(pixel / 255.0 * scale_factor)"""
    return [round(p / 255.0 * scale_factor) for p in pixels]


def conv2d(input_flat, input_h, input_w, kernel_flat, kh, kw, divisor):
    """
    Reproduce the EXACT C++ conv2d logic from openfhe_cnn_ops.cpp.
    
    The C++ "RESET" approach (line ~340):
    1. For each kernel position (kh, kw), rotate input by (kh*input_w + kw)
    2. Multiply ALL slots by scalar kernel[kh][kw]  
    3. Accumulate
    4. Decrypt, rearrange from input-width layout to output-width layout, ÷divisor
    
    In plaintext: the "rotate" means shifting the array.
    After accumulation, slot (oh*input_w + ow) contains:
      sum_{kh,kw} input[(oh+kh)*input_w + (ow+kw)] * kernel[kh*kw_dim+kw]
    which is exactly conv2d output for position (oh, ow)!
    """
    out_h = input_h - kh + 1
    out_w = input_w - kw + 1
    
    # Simulate the rotate-multiply-accumulate
    # After rotation by `rot`, slot[i] = input[i + rot]
    # Then multiply by scalar kval, so slot[i] = input[i+rot] * kval
    # Accumulate across all kernel positions
    
    n = len(input_flat)
    acc = [0] * n
    
    for ky in range(kh):
        for kx in range(kw):
            kval = kernel_flat[ky * kw + kx]
            if kval == 0:
                continue
            rot = ky * input_w + kx
            
            # After EvalRotate by `rot`, slot[i] = input[i + rot]
            # (assuming zero for out-of-bounds, though in HE it wraps)
            for i in range(n):
                src = i + rot
                if src < n:
                    acc[i] += input_flat[src] * kval
    
    # Rearrange: extract valid positions from input-width layout to output-width layout
    # C++ code: src_idx = oh * input_w + ow, dst_idx = oh * out_w + ow
    output = [0] * (out_h * out_w)
    for oh in range(out_h):
        for ow in range(out_w):
            src_idx = oh * input_w + ow
            dst_idx = oh * out_w + ow
            if src_idx < n:
                output[dst_idx] = acc[src_idx] // divisor
    
    return output, out_h, out_w


def add_bias(data, bias_val, count):
    """Add bias to first `count` elements."""
    result = list(data)
    for i in range(min(count, len(result))):
        result[i] += bias_val
    return result


def square_activate(data, count, divisor):
    """x² / divisor for first `count` elements."""
    result = [0] * len(data)
    for i in range(min(count, len(data))):
        result[i] = (data[i] * data[i]) // divisor
    return result


def avgpool(input_flat, input_h, input_w, pool_size, stride):
    """
    Reproduce C++ avgpool logic:
    1. For each offset (ph, pw) in pool window, rotate by (ph*input_w + pw) and add
    2. Decrypt, extract valid positions, ÷pool_area, rearrange to output layout
    """
    out_h = (input_h - pool_size) // stride + 1
    out_w = (input_w - pool_size) // stride + 1
    pool_area = pool_size * pool_size
    
    n = len(input_flat)
    pool_sum = [0] * n
    
    for ph in range(pool_size):
        for pw in range(pool_size):
            rot = ph * input_w + pw
            for i in range(n):
                src = i + rot
                if src < n:
                    pool_sum[i] += input_flat[src]
    
    # Extract valid output positions
    output = [0] * (out_h * out_w)
    for oh in range(out_h):
        for ow in range(out_w):
            src_idx = (oh * stride) * input_w + (ow * stride)
            dst_idx = oh * out_w + ow
            if src_idx < n:
                output[dst_idx] = pool_sum[src_idx] // pool_area
    
    return output, out_h, out_w


def matmul(input_flat, weight_flat, rows, cols, divisor):
    """
    Reproduce C++ matmul logic:
    For each row i: 
      1. Create mask with weight_row[i] in slots 0..cols-1
      2. EvalMult(input, mask) → product
      3. Rotate-and-add to sum slots 0..cols-1
      4. Extract slot 0, place in slot i
    Then ÷divisor via decrypt
    """
    output = [0] * rows
    for i in range(rows):
        dot = 0
        for j in range(cols):
            dot += input_flat[j] * weight_flat[i * cols + j]
        output[i] = dot // divisor
    return output


def predict_plaintext(pixels_0_255, weights, scale_factor):
    """Run the full CNN pipeline in plaintext integers."""
    S = scale_factor
    
    # Scale pixels
    scaled = scale_pixels(pixels_0_255, S)
    
    # Conv1: 28×28 → 24×24, ÷S
    x, h, w = conv2d(scaled, 28, 28, weights["conv1_kernel"], 5, 5, S)
    
    # + bias (broadcast to 24×24 = 576 slots)
    x = add_bias(x, weights["conv1_bias"][0], h * w)
    
    # x²/S
    x = square_activate(x, h * w, S)
    
    # AvgPool 2×2 (24×24 → 12×12)
    x, h, w = avgpool(x, h, w, 2, 2)
    
    # Conv2: 12×12 → 8×8, ÷S
    x, h, w = conv2d(x, h, w, weights["conv2_kernel"], 5, 5, S)
    
    # + bias (broadcast to 8×8 = 64 slots)
    x = add_bias(x, weights["conv2_bias"][0], h * w)
    
    # x²/S
    x = square_activate(x, h * w, S)
    
    # AvgPool 2×2 (8×8 → 4×4)
    x, h, w = avgpool(x, h, w, 2, 2)
    
    # FC: 16 → 10, ÷S
    logits = matmul(x, weights["fc_weights"], 10, 16, S)
    
    # + FC bias
    for i in range(10):
        logits[i] += weights["fc_bias"][i]
    
    predicted = logits.index(max(logits))
    return predicted, logits


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, "weights")
    
    # Load config
    with open(os.path.join(weights_dir, "model_config.json")) as f:
        config = json.load(f)
    scale_factor = config["quantization"]["scale_factor"]
    
    print("=" * 70)
    print("  Plaintext Integer CNN Verification")
    print(f"  Scale factor: {scale_factor}")
    print(f"  Expected float accuracy: {config['accuracy']['float_model']}%")
    print("=" * 70)
    
    weights = load_weights(weights_dir)
    images, labels = load_test_data(weights_dir)
    
    print(f"\n  Conv1 kernel: {weights['conv1_kernel']}")
    print(f"  Conv1 bias:   {weights['conv1_bias']}")
    print(f"  Conv2 kernel: {weights['conv2_kernel']}")
    print(f"  Conv2 bias:   {weights['conv2_bias']}")
    
    correct = 0
    total = len(images)
    
    for i, (pixels, true_label) in enumerate(zip(images, labels)):
        predicted, logits = predict_plaintext(pixels, weights, scale_factor)
        is_correct = predicted == true_label
        if is_correct:
            correct += 1
        icon = "✅" if is_correct else "❌"
        print(f"\n  [{i+1}/{total}] Digit {true_label}: {icon} predicted={predicted}")
        print(f"         logits: {logits}")
    
    print(f"\n{'=' * 70}")
    print(f"  PLAINTEXT RESULTS: {correct}/{total} ({correct/total*100:.1f}% accuracy)")
    print(f"{'=' * 70}")
    
    if correct == 0:
        print("\n  ⚠ 0% accuracy in plaintext too!")
        print("    → Bug is in integer arithmetic / scale management, NOT in HE slots.")
        print("    → Need to compare with PyTorch float model output.")
    elif correct < total * 0.5:
        print(f"\n  ⚠ Low accuracy ({correct/total*100:.1f}%) — quantization error too large?")
    else:
        print(f"\n  ✓ Good accuracy — if HE gives 0%, the bug is in the HE slot layout.")


if __name__ == "__main__":
    main()
