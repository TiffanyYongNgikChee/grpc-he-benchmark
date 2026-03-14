#!/usr/bin/env python3
"""
Check if intermediate values overflow the plaintext modulus p=7340033.
"""
import csv, os

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

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, "weights")
    weights = load_weights(weights_dir)
    images, labels = load_test_data(weights_dir)
    S = 1000
    P = 7340033
    HALF_P = P // 2
    
    print(f"Plaintext modulus p = {P}")
    print(f"Half p = {HALF_P} = ±{HALF_P}")
    print()
    
    # Check Conv1 intermediate accumulator values (before ÷S)
    for img_idx in range(1):  # Just first image
        pixels = images[img_idx]
        scaled = scale_pixels(pixels, S)
        
        print(f"Image {img_idx} (digit {labels[img_idx]}):")
        print(f"  Max scaled pixel: {max(scaled)}")
        print(f"  Max |kernel|: {max(abs(v) for v in weights['conv1_kernel'])}")
        
        # Simulate conv2d accumulation WITHOUT division
        # This is what happens inside the HE ciphertext slots
        input_h, input_w = 28, 28
        kernel = weights['conv1_kernel']
        kh, kw = 5, 5
        
        max_acc = 0
        min_acc = 0
        overflow_count = 0
        
        n = len(scaled)
        acc = [0] * 8192
        
        for ky in range(kh):
            for kx in range(kw):
                kval = kernel[ky * kw + kx]
                if kval == 0: continue
                rot = ky * input_w + kx
                for i in range(n):
                    src = i + rot
                    if src < n:
                        acc[i] += scaled[src] * kval
        
        valid_acc = acc[:n]
        max_acc = max(valid_acc)
        min_acc = min(valid_acc)
        
        print(f"\n  Conv1 accumulator (BEFORE ÷{S}):")
        print(f"    Max: {max_acc:>15,}")
        print(f"    Min: {min_acc:>15,}")
        print(f"    ±half_p: {HALF_P:>15,}")
        
        if max_acc > HALF_P or min_acc < -HALF_P:
            print(f"    🚨 OVERFLOW! Values exceed ±{HALF_P}")
            overflow_count = sum(1 for v in valid_acc if abs(v) > HALF_P)
            print(f"    {overflow_count} slots overflow out of {n}")
        else:
            print(f"    ✅ No overflow in conv1 accumulator")
        
        # Conv1 output after ÷S
        conv1_out = [v // S for v in valid_acc]
        out_h, out_w = 24, 24
        conv1_output = [0] * (out_h * out_w)
        for oh in range(out_h):
            for ow in range(out_w):
                src = oh * input_w + ow
                dst = oh * out_w + ow
                conv1_output[dst] = acc[src] // S
        
        # After + bias
        bias = weights['conv1_bias'][0]
        conv1_biased = [v + bias for v in conv1_output]
        
        # x²: each value squared BEFORE division
        # In HE: EvalMult(ct, ct) gives ct² in plaintext space
        # The values in slots are (conv1_out + bias), which are ~±600 to ±3000
        # Squaring gives values up to ~9,000,000 which EXCEEDS p/2 = 3,670,016!
        print(f"\n  After Conv1 + bias:")
        print(f"    Max: {max(conv1_biased):>15,}")
        print(f"    Min: {min(conv1_biased):>15,}")
        
        # x² values BEFORE division
        squared_raw = [v * v for v in conv1_biased]
        max_sq = max(squared_raw)
        print(f"\n  x² (BEFORE ÷{S}):")
        print(f"    Max: {max_sq:>15,}")
        print(f"    ±half_p: {HALF_P:>15,}")
        
        if max_sq > HALF_P:
            print(f"    🚨🚨🚨 OVERFLOW! x² = {max_sq} > {HALF_P}")
            print(f"    This is THE BUG! Square activation overflows the plaintext modulus!")
            overflow_count = sum(1 for v in squared_raw if v > HALF_P)
            print(f"    {overflow_count}/{len(squared_raw)} values overflow")
            
            # What's the max value that can be squared without overflow?
            import math
            max_safe = int(math.sqrt(HALF_P))
            print(f"\n    Max safe value for x²: ±{max_safe}")
            print(f"    But actual max |value|: {max(abs(v) for v in conv1_biased)}")
            count_unsafe = sum(1 for v in conv1_biased if abs(v) > max_safe)
            print(f"    {count_unsafe}/{len(conv1_biased)} values exceed ±{max_safe}")
        else:
            print(f"    ✅ No overflow in square activation")

if __name__ == "__main__":
    main()
