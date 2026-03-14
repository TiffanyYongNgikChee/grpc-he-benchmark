#!/usr/bin/env python3
"""
Find the optimal scale factor that:
1. Keeps x² values within p/2 (no overflow)
2. Still gives good accuracy

Also test with modular arithmetic to confirm the overflow theory.
"""
import csv, os, json, math

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

def mod_center(val, p):
    """Center value in [-p/2, p/2]"""
    val = val % p
    if val > p // 2:
        val -= p
    return val

def conv2d(input_flat, input_h, input_w, kernel_flat, kh, kw, divisor, p=None):
    out_h = input_h - kh + 1
    out_w = input_w - kw + 1
    n = len(input_flat)
    acc = [0] * n
    for ky in range(kh):
        for kx in range(kw):
            kval = kernel_flat[ky * kw + kx]
            if kval == 0: continue
            rot = ky * input_w + kx
            for i in range(n):
                src = i + rot
                if src < n:
                    acc[i] += input_flat[src] * kval
                    if p: acc[i] = mod_center(acc[i], p)
    
    output = [0] * (out_h * out_w)
    for oh in range(out_h):
        for ow in range(out_w):
            src_idx = oh * input_w + ow
            dst_idx = oh * out_w + ow
            if src_idx < n:
                output[dst_idx] = acc[src_idx] // divisor
    return output, out_h, out_w

def avgpool(input_flat, input_h, input_w, pool_size, stride, p=None):
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
                    if p: pool_sum[i] = mod_center(pool_sum[i], p)
    
    output = [0] * (out_h * out_w)
    for oh in range(out_h):
        for ow in range(out_w):
            src_idx = (oh * stride) * input_w + (ow * stride)
            dst_idx = oh * out_w + ow
            if src_idx < n:
                output[dst_idx] = pool_sum[src_idx] // pool_area
    return output, out_h, out_w

def predict(pixels, weights, S, p=None):
    """Run pipeline. If p is given, apply modular arithmetic (simulating BFV)."""
    scaled = scale_pixels(pixels, S)
    
    x, h, w = conv2d(scaled, 28, 28, weights["conv1_kernel"], 5, 5, S, p)
    # bias
    bias = weights["conv1_bias"][0]
    for i in range(h*w): x[i] += bias
    # x²/S  — THIS IS WHERE OVERFLOW HAPPENS
    for i in range(h*w):
        sq = x[i] * x[i]
        if p: sq = mod_center(sq, p)  # modular wrapping!
        x[i] = sq // S
    
    x, h, w = avgpool(x, h, w, 2, 2, p)
    
    x, h, w = conv2d(x, h, w, weights["conv2_kernel"], 5, 5, S, p)
    bias = weights["conv2_bias"][0]
    for i in range(h*w): x[i] += bias
    for i in range(h*w):
        sq = x[i] * x[i]
        if p: sq = mod_center(sq, p)
        x[i] = sq // S
    
    x, h, w = avgpool(x, h, w, 2, 2, p)
    
    logits = [0] * 10
    for i in range(10):
        dot = 0
        for j in range(16):
            dot += x[j] * weights["fc_weights"][i * 16 + j]
        logits[i] = dot // S + weights["fc_bias"][i]
    
    predicted = logits.index(max(logits))
    return predicted, logits


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, "weights")
    weights = load_weights(weights_dir)
    images, labels = load_test_data(weights_dir)
    
    P = 7340033
    
    # ---- Test 1: Confirm the bug by simulating WITH modular arithmetic ----
    print("=" * 70)
    print("  Test 1: With mod p=7340033 arithmetic (simulating BFV overflow)")
    print("=" * 70)
    correct = 0
    for i in range(len(images)):
        pred, logits = predict(images[i], weights, 1000, p=P)
        is_correct = pred == labels[i]
        if is_correct: correct += 1
        icon = "✅" if is_correct else "❌"
        print(f"  [{i+1}] digit={labels[i]} {icon} pred={pred}  logits={logits}")
    print(f"  Result: {correct}/{len(images)} ({correct/len(images)*100:.0f}%)")
    
    # ---- Test 2: Find safe scale factors ----
    print(f"\n{'=' * 70}")
    print(f"  Test 2: Accuracy vs scale factor (no mod, unlimited precision)")
    print(f"{'=' * 70}")
    
    for S in [10, 50, 100, 200, 500, 1000]:
        correct = 0
        for i in range(len(images)):
            pred, _ = predict(images[i], weights, S)
            if pred == labels[i]: correct += 1
        
        # Check max intermediate x² value
        max_sq = 0
        for i in range(len(images)):
            scaled = scale_pixels(images[i], S)
            x, h, w = conv2d(scaled, 28, 28, weights["conv1_kernel"], 5, 5, S)
            bias = weights["conv1_bias"][0]
            for j in range(h*w): x[j] += bias
            for j in range(h*w):
                if x[j] * x[j] > max_sq:
                    max_sq = x[j] * x[j]
        
        safe = "✅" if max_sq < P // 2 else "❌"
        print(f"  S={S:>5}: accuracy={correct}/{len(images)} ({correct/len(images)*100:5.1f}%)  "
              f"max_x²={max_sq:>12,}  {safe} {'SAFE' if max_sq < P//2 else f'OVERFLOW (>{P//2:,})'}")
    
    # ---- Test 3: Larger plaintext modulus options ----
    print(f"\n{'=' * 70}")
    print(f"  Test 3: With S=1000, what plaintext modulus is needed?")
    print(f"{'=' * 70}")
    
    # Find max x² across ALL layers
    for i in range(len(images)):
        scaled = scale_pixels(images[i], 1000)
        x, h, w = conv2d(scaled, 28, 28, weights["conv1_kernel"], 5, 5, 1000)
        bias = weights["conv1_bias"][0]
        for j in range(h*w): x[j] += bias
        max_after_conv1 = max(abs(v) for v in x[:h*w])
        
        sq_vals = [v*v for v in x[:h*w]]
        max_sq1 = max(sq_vals)
        
        for j in range(h*w): x[j] = sq_vals[j] // 1000
        x, h, w = avgpool(x, h, w, 2, 2)
        x, h, w = conv2d(x, h, w, weights["conv2_kernel"], 5, 5, 1000)
        bias2 = weights["conv2_bias"][0]
        for j in range(h*w): x[j] += bias2
        max_after_conv2 = max(abs(v) for v in x[:h*w])
        sq_vals2 = [v*v for v in x[:h*w]]
        max_sq2 = max(sq_vals2)
        
        if i == 0:
            print(f"  Image 0:")
            print(f"    After conv1+bias: max |val| = {max_after_conv1}")
            print(f"    After x² (layer1): max = {max_sq1:,}")
            print(f"    After conv2+bias: max |val| = {max_after_conv2}")
            print(f"    After x² (layer2): max = {max_sq2:,}")
    
    # Find overall max across all images
    overall_max_sq = 0
    for i in range(len(images)):
        scaled = scale_pixels(images[i], 1000)
        x, h, w = conv2d(scaled, 28, 28, weights["conv1_kernel"], 5, 5, 1000)
        for j in range(h*w): x[j] += weights["conv1_bias"][0]
        for j in range(h*w):
            overall_max_sq = max(overall_max_sq, x[j]*x[j])
        for j in range(h*w): x[j] = (x[j]*x[j]) // 1000
        x, h, w = avgpool(x, h, w, 2, 2)
        x, h, w = conv2d(x, h, w, weights["conv2_kernel"], 5, 5, 1000)
        for j in range(h*w): x[j] += weights["conv2_bias"][0]
        for j in range(h*w):
            overall_max_sq = max(overall_max_sq, x[j]*x[j])
    
    min_p = overall_max_sq * 2 + 1
    print(f"\n  Overall max x² across all images: {overall_max_sq:,}")
    print(f"  Minimum p needed: {min_p:,}")
    print(f"  Current p: {P:,}")
    print(f"  Need p at least: {min_p:,} (current {P:,} is {'sufficient' if P > min_p else 'TOO SMALL'})")


if __name__ == "__main__":
    main()
