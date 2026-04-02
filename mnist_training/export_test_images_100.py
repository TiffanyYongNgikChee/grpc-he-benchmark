#!/usr/bin/env python3
"""
Export 100 MNIST test images (10 per digit) for automated benchmarking.

Selects 10 correctly-classified images per digit (0-9) = 100 total.
Exports to each weight directory (weights/, weights_deg2/, weights_deg3/, weights_deg4/)
so the Rust benchmark can load them alongside the matching model weights.

Output files (per weight directory):
  test_images_100.csv   — 100 images, each row = 784 pixel values (0-255)
  test_labels_100.csv   — 100 labels (one per row)

Usage:
    cd mnist_training
    python export_test_images_100.py
"""

import os
import sys
import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(__file__))
from train_mnist import HE_CNN, SquareActivation


IMAGES_PER_DIGIT = 10
TOTAL_IMAGES = IMAGES_PER_DIGIT * 10  # 100


def export_test_images_100():
    print("=" * 60)
    print("Export 100 MNIST Test Images (10 per digit)")
    print("=" * 60)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    print(f"\n  Test set: {len(test_dataset)} images")

    # Load trained model (deg2 — the one with highest accuracy)
    model_path = os.path.join(os.path.dirname(__file__), "he_cnn_model.pth")
    if not os.path.exists(model_path):
        print(f"\n  ERROR: Trained model not found at {model_path}")
        print("  Run train_mnist.py first.")
        sys.exit(1)

    model = HE_CNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    print(f"  Model loaded from: {model_path}")

    # Find 10 correctly-classified images per digit
    print(f"\n  Finding {IMAGES_PER_DIGIT} correct predictions per digit...")
    selected = {d: [] for d in range(10)}  # digit → list of (image_tensor, raw_pixels, idx)

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image, label = test_dataset[idx]

            if len(selected[label]) >= IMAGES_PER_DIGIT:
                continue

            output = model(image.unsqueeze(0))
            predicted = output.argmax(dim=1).item()

            if predicted == label:
                raw_pixels = (image.squeeze(0).numpy() * 255).astype(np.uint8)
                selected[label].append((image, raw_pixels, idx))

            # Check if we have enough
            if all(len(v) >= IMAGES_PER_DIGIT for v in selected.values()):
                break

    # Summary
    for d in range(10):
        count = len(selected[d])
        indices = [s[2] for s in selected[d]]
        print(f"    Digit {d}: {count}/{IMAGES_PER_DIGIT} images (indices: {indices[:3]}...)")

    total = sum(len(v) for v in selected.values())
    print(f"\n  Total selected: {total}/{TOTAL_IMAGES}")

    if total < TOTAL_IMAGES:
        missing = {d: IMAGES_PER_DIGIT - len(selected[d]) for d in range(10) if len(selected[d]) < IMAGES_PER_DIGIT}
        print(f"  WARNING: Missing images for digits: {missing}")

    # Build arrays (sorted by digit, then by selection order)
    all_images = []
    all_labels = []
    for d in range(10):
        for img_tensor, raw_pixels, idx in selected[d]:
            all_images.append(raw_pixels.flatten().astype(np.int64))
            all_labels.append(d)

    images_array = np.array(all_images)  # (100, 784)
    labels_array = np.array(all_labels)  # (100,)

    print(f"\n  Images array shape: {images_array.shape}")
    print(f"  Labels array shape: {labels_array.shape}")

    # Export to ALL weight directories
    base_dir = os.path.dirname(__file__)
    weight_dirs = [
        os.path.join(base_dir, "weights"),
        os.path.join(base_dir, "weights_deg2"),
        os.path.join(base_dir, "weights_deg3"),
        os.path.join(base_dir, "weights_deg4"),
    ]

    for wdir in weight_dirs:
        if not os.path.isdir(wdir):
            print(f"\n  SKIP: {wdir} does not exist")
            continue

        images_path = os.path.join(wdir, "test_images_100.csv")
        np.savetxt(images_path, images_array, fmt="%d", delimiter=",")

        labels_path = os.path.join(wdir, "test_labels_100.csv")
        np.savetxt(labels_path, labels_array.reshape(-1, 1), fmt="%d", delimiter=",")

        print(f"\n  Saved to {wdir}/:")
        print(f"    test_images_100.csv  — {images_array.shape[0]} images × {images_array.shape[1]} pixels")
        print(f"    test_labels_100.csv  — {labels_array.shape[0]} labels")

    # Final summary
    print(f"\n{'=' * 60}")
    print("Export COMPLETE")
    print(f"{'=' * 60}")
    print(f"  {total} images exported to {len([d for d in weight_dirs if os.path.isdir(d)])} weight directories")
    print(f"  Format: raw 0-255 pixel values (NOT scaled)")
    print(f"  Rust pipeline scales: round(pixel / 255 * scale_factor)")
    print(f"\n  Next: Run 'cargo run --example mnist_benchmark' on EC2 #2")


if __name__ == "__main__":
    export_test_images_100()
