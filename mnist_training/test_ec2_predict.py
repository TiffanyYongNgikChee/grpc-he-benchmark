#!/usr/bin/env python3
"""
Test the /api/predict endpoint with real MNIST test images.

Reads test_images.csv and test_labels.csv from the weights/ folder,
sends each image to the Spring Boot API, and reports accuracy + timing.

Usage (run on EC2):
    python3 test_ec2_predict.py http://localhost:8080

Or from your Mac (if EC2 is accessible):
    python3 test_ec2_predict.py http://54.205.254.22:8080
"""

import csv
import json
import sys
import time
import urllib.request
import urllib.error

def load_test_data(weights_dir="weights"):
    """Load test images and labels from CSV files."""
    # Load images (each row = 784 comma-separated pixel values)
    images = []
    with open(f"{weights_dir}/test_images.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            pixels = [int(x) for x in row]
            assert len(pixels) == 784, f"Expected 784 pixels, got {len(pixels)}"
            images.append(pixels)
    
    # Load labels (single row of comma-separated digits)
    with open(f"{weights_dir}/test_labels.csv", "r") as f:
        labels = [int(x) for x in f.readline().strip().split(",")]
    
    assert len(images) == len(labels), f"Mismatch: {len(images)} images, {len(labels)} labels"
    return images, labels


def predict(api_base, pixels, scale_factor=1000):
    """Send a prediction request to the API."""
    url = f"{api_base}/api/predict"
    payload = json.dumps({
        "pixels": pixels,
        "scaleFactor": scale_factor
    }).encode("utf-8")
    
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    api_base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    
    print("=" * 70)
    print("  MNIST True FHE CNN Inference Test")
    print(f"  API: {api_base}")
    print("=" * 70)
    
    # Load test data
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, "weights")
    images, labels = load_test_data(weights_dir)
    print(f"\n  Loaded {len(images)} test images (digits {labels})")
    
    # Test each image
    correct = 0
    total = 0
    results = []
    
    for i, (pixels, true_label) in enumerate(zip(images, labels)):
        print(f"\n  [{i+1}/{len(images)}] Testing digit {true_label}...", end=" ", flush=True)
        
        start = time.time()
        try:
            result = predict(api_base, pixels)
            elapsed = time.time() - start
            
            predicted = result.get("predictedDigit", -1)
            confidence = result.get("confidence", 0)
            total_ms = result.get("totalMs", 0)
            status = result.get("status", "unknown")
            logits = result.get("logits", [])
            
            is_correct = predicted == true_label
            if is_correct:
                correct += 1
            total += 1
            
            icon = "✅" if is_correct else "❌"
            print(f"{icon} predicted={predicted} (confidence={confidence:.1%}, time={total_ms:.0f}ms)")
            
            # Show logits for wrong predictions
            if not is_correct:
                print(f"         logits: {logits}")
            
            results.append({
                "image_index": i,
                "true_label": true_label,
                "predicted": predicted,
                "correct": is_correct,
                "confidence": confidence,
                "total_ms": total_ms,
                "encryption_ms": result.get("encryptionMs", 0),
                "conv1_ms": result.get("conv1Ms", 0),
                "act1_ms": result.get("act1Ms", 0),
                "pool1_ms": result.get("pool1Ms", 0),
                "conv2_ms": result.get("conv2Ms", 0),
                "act2_ms": result.get("act2Ms", 0),
                "pool2_ms": result.get("pool2Ms", 0),
                "fc_ms": result.get("fcMs", 0),
                "decryption_ms": result.get("decryptionMs", 0),
                "logits": logits,
                "status": status
            })
            
        except Exception as e:
            elapsed = time.time() - start
            print(f"💥 ERROR after {elapsed:.1f}s: {e}")
            total += 1
            results.append({
                "image_index": i,
                "true_label": true_label,
                "predicted": -1,
                "correct": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 70)
    print(f"  RESULTS: {correct}/{total} correct ({correct/total*100:.1f}% accuracy)")
    print("=" * 70)
    
    # Timing summary
    successful = [r for r in results if "total_ms" in r]
    if successful:
        times = [r["total_ms"] for r in successful]
        print(f"\n  Timing (per image):")
        print(f"    Average: {sum(times)/len(times):.0f} ms")
        print(f"    Min:     {min(times):.0f} ms")
        print(f"    Max:     {max(times):.0f} ms")
        
        # Per-layer averages
        print(f"\n  Per-layer averages:")
        for key in ["encryption_ms", "conv1_ms", "act1_ms", "pool1_ms",
                     "conv2_ms", "act2_ms", "pool2_ms", "fc_ms", "decryption_ms"]:
            vals = [r.get(key, 0) for r in successful]
            label = key.replace("_ms", "").replace("_", " ").title()
            print(f"    {label:20s}: {sum(vals)/len(vals):8.1f} ms")
    
    # Save results to CSV
    csv_path = os.path.join(script_dir, "fhe_test_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_index", "true_label", "predicted", "correct",
            "confidence", "total_ms", "encryption_ms", "conv1_ms",
            "act1_ms", "pool1_ms", "conv2_ms", "act2_ms", "pool2_ms",
            "fc_ms", "decryption_ms", "status"
        ])
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in writer.fieldnames}
            writer.writerow(row)
    print(f"\n  Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
