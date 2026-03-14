#!/usr/bin/env python3
"""
Debug the HE conv2d by simulating WITH cyclic rotation (like OpenFHE).
Compare with the linear (non-wrapping) version to find the bug.
"""
import csv, os, json

def load_weights(weights_dir):
    def load_csv(filename):
        values = []
        with open(os.path.join(weights_dir, filename)) as f:
            for line in f:
                for tok in line.strip().split(","):
                    tok = tok.strip()
                    if tok: values.append(int(tok))
        return values
    return {
        "conv1_kernel": load_csv("conv1_weights.csv"),
        "conv1_bias": load_csv("conv1_bias.csv"),
        "conv2_kernel": load_csv("conv2_weights.csv"),
        "conv2_bias": load_csv("conv2_bias.csv"),
        "fc_weights": load_csv("fc_weights.csv"),
        "fc_bias": load_csv("fc_bias.csv"),
    }

def load_test_data(weights_dir):
    images = []
    with open(os.path.join(weights_dir, "test_images.csv")) as f:
        reader = csv.reader(f)
        for row in reader:
            images.append([int(x) for x in row])
    with open(os.path.join(weights_dir, "test_labels.csv")) as f:
        labels = [int(x) for x in f.readline().strip().split(",")]
    return images, labels

def scale_pixels(pixels, S):
    return [round(p / 255.0 * S) for p in pixels]

SLOT_COUNT = 8192  # OpenFHE BFV default for these params

def conv2d_cyclic(input_flat, input_h, input_w, kernel_flat, kh, kw, divisor):
    """Conv2d with CYCLIC rotation (simulating OpenFHE EvalRotate)."""
    out_h = input_h - kh + 1
    out_w = input_w - kw + 1
    
    # Pad input to slot_count (zeros beyond pixel data)
    slots = [0] * SLOT_COUNT
    for i in range(len(input_flat)):
        if i < SLOT_COUNT:
            slots[i] = input_flat[i]
    
    acc = [0] * SLOT_COUNT
    
    for ky in range(kh):
        for kx in range(kw):
            kval = kernel_flat[ky * kw + kx]
            if kval == 0:
                continue
            rot = ky * input_w + kx
            
            # CYCLIC rotation: slot[i] = slots[(i + rot) % SLOT_COUNT]
            rotated = [0] * SLOT_COUNT
            for i in range(SLOT_COUNT):
                src = (i + rot) % SLOT_COUNT
                rotated[i] = slots[src]
            
            # Multiply by scalar kval (all slots)
            for i in range(SLOT_COUNT):
                acc[i] += rotated[i] * kval
    
    # Rearrange from input-width to output-width + ÷divisor
    output = [0] * (out_h * out_w)
    for oh in range(out_h):
        for ow in range(out_w):
            src_idx = oh * input_w + ow
            dst_idx = oh * out_w + ow
            if src_idx < SLOT_COUNT:
                output[dst_idx] = acc[src_idx] // divisor
    
    return output, out_h, out_w

def conv2d_linear(input_flat, input_h, input_w, kernel_flat, kh, kw, divisor):
    """Conv2d with LINEAR shift (no wrapping) — the correct behavior."""
    out_h = input_h - kh + 1
    out_w = input_w - kw + 1
    
    n = len(input_flat)
    acc = [0] * n
    
    for ky in range(kh):
        for kx in range(kw):
            kval = kernel_flat[ky * kw + kx]
            if kval == 0:
                continue
            rot = ky * input_w + kx
            for i in range(n):
                src = i + rot
                if src < n:
                    acc[i] += input_flat[src] * kval
    
    output = [0] * (out_h * out_w)
    for oh in range(out_h):
        for ow in range(out_w):
            src_idx = oh * input_w + ow
            dst_idx = oh * out_w + ow
            if src_idx < n:
                output[dst_idx] = acc[src_idx] // divisor
    
    return output, out_h, out_w

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, "weights")
    weights = load_weights(weights_dir)
    images, labels = load_test_data(weights_dir)
    
    S = 1000
    
    # Test first image (digit 0)
    pixels = images[0]
    scaled = scale_pixels(pixels, S)
    
    print("Testing digit 0 — Conv1 comparison")
    print("=" * 60)
    
    out_cyc, h_c, w_c = conv2d_cyclic(scaled, 28, 28, weights["conv1_kernel"], 5, 5, S)
    out_lin, h_l, w_l = conv2d_linear(scaled, 28, 28, weights["conv1_kernel"], 5, 5, S)
    
    print(f"  Cyclic output[0:20]:  {out_cyc[:20]}")
    print(f"  Linear output[0:20]:  {out_lin[:20]}")
    
    # Check if they match
    match = True
    diffs = 0
    for i in range(len(out_lin)):
        if out_cyc[i] != out_lin[i]:
            if diffs < 5:
                print(f"  DIFF at [{i}]: cyclic={out_cyc[i]} linear={out_lin[i]}")
            diffs += 1
            match = False
    
    if match:
        print(f"\n  ✅ Conv1 outputs MATCH — cyclic rotation is NOT the bug for 784→8192 slots")
    else:
        print(f"\n  ❌ Conv1 outputs differ at {diffs} positions — cyclic rotation IS the bug!")
    
    # Now test the full pipeline with cyclic rotation everywhere
    print("\n\nFull pipeline with cyclic conv2d:")
    print("=" * 60)
    
    for idx in range(len(images)):
        pixels = images[idx]
        true_label = labels[idx]
        scaled = scale_pixels(pixels, S)
        
        # Conv1 cyclic
        x, h, w = conv2d_cyclic(scaled, 28, 28, weights["conv1_kernel"], 5, 5, S)
        # + bias
        for i in range(h * w): x[i] += weights["conv1_bias"][0]
        # x²/S
        x = [(v * v) // S for v in x]
        # avgpool (use linear — pool doesn't have the same issue since data fits)
        x2, h, w = avgpool_cyclic(x, h, w, 2, 2)
        
        # Conv2 cyclic
        x2, h, w = conv2d_cyclic(x2, h, w, weights["conv2_kernel"], 5, 5, S)
        for i in range(h * w): x2[i] += weights["conv2_bias"][0]
        x2 = [(v * v) // S for v in x2]
        x2, h, w = avgpool_cyclic(x2, h, w, 2, 2)
        
        # FC (same as before)
        logits = [0] * 10
        for i in range(10):
            dot = 0
            for j in range(16):
                dot += x2[j] * weights["fc_weights"][i * 16 + j]
            logits[i] = dot // S + weights["fc_bias"][i]
        
        predicted = logits.index(max(logits))
        icon = "✅" if predicted == true_label else "❌"
        print(f"  [{idx+1}] digit={true_label} {icon} predicted={predicted}  logits={logits}")


def avgpool_cyclic(input_flat, input_h, input_w, pool_size, stride):
    """Avgpool with cyclic rotation."""
    out_h = (input_h - pool_size) // stride + 1
    out_w = (input_w - pool_size) // stride + 1
    pool_area = pool_size * pool_size
    
    slots = [0] * SLOT_COUNT
    for i in range(min(len(input_flat), SLOT_COUNT)):
        slots[i] = input_flat[i]
    
    pool_sum = [0] * SLOT_COUNT
    for ph in range(pool_size):
        for pw in range(pool_size):
            rot = ph * input_w + pw
            for i in range(SLOT_COUNT):
                src = (i + rot) % SLOT_COUNT
                pool_sum[i] += slots[src]
    
    output = [0] * (out_h * out_w)
    for oh in range(out_h):
        for ow in range(out_w):
            src_idx = (oh * stride) * input_w + (ow * stride)
            dst_idx = oh * out_w + ow
            if src_idx < SLOT_COUNT:
                output[dst_idx] = pool_sum[src_idx] // pool_area
    
    return output, out_h, out_w


if __name__ == "__main__":
    main()
