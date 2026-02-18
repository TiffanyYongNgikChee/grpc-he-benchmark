#!/usr/bin/env python3
"""
Step 6d: Export MNIST test images for Rust encrypted inference verification.

Exports a small set of MNIST test images as CSV files that can be loaded
by the Rust weight loader. Selects one correctly-classified image per digit
(0-9) to verify the full encrypted pipeline.

Output files (in mnist_training/weights/):
  test_images.csv   — 10 images, each row = 784 pixel values (0-255 range)
  test_labels.csv   — 10 labels (one per image, 0-9)

The images are in the SAME format as PyTorch's raw MNIST (0-255 uint8),
NOT the ToTensor-scaled (0.0-1.0) format. The Rust pipeline will apply
the same scaling: round(pixel / 255.0 * scale_factor).

Usage:
    cd mnist_training
    python export_test_images.py
"""

import os
import sys
import numpy as np
import torch
from torchvision import datasets, transforms

# Add parent dir so we can import HE_CNN from train_mnist
sys.path.insert(0, os.path.dirname(__file__))
from train_mnist import HE_CNN, SquareActivation


def export_test_images():
    """Export 10 correctly-classified MNIST test images (one per digit)."""
    print("=" * 60)
    print("Step 6d: Export MNIST Test Images for Rust Verification")
    print("=" * 60)

    # Load MNIST test set (raw 0-255 pixels)
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    # We need two transforms:
    # 1. ToTensor (for model inference, scales to 0-1)
    # 2. Raw pixels (for CSV export, 0-255)
    transform = transforms.Compose([transforms.ToTensor()])
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"\n  Test set: {len(test_dataset)} images")

    # Load trained model
    model_path = os.path.join(os.path.dirname(__file__), "he_cnn_model.pth")
    if not os.path.exists(model_path):
        print(f"\n  ERROR: Trained model not found at {model_path}")
        print("  Run train_mnist.py first to train the model.")
        sys.exit(1)

    model = HE_CNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    print(f"  Model loaded from: {model_path}")

    # Find one correctly-classified image per digit
    print("\n  Finding one correct prediction per digit...")
    selected = {}  # digit → (image_tensor, raw_pixels, index)
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image, label = test_dataset[idx]  # image: [1, 28, 28] float 0-1
            
            if label in selected:
                continue  # Already have this digit
            
            # Run inference
            output = model(image.unsqueeze(0))  # [1, 10]
            predicted = output.argmax(dim=1).item()
            
            if predicted == label:
                # Get raw 0-255 pixels for CSV export
                # image is [1, 28, 28] in range [0, 1], multiply back by 255
                raw_pixels = (image.squeeze(0).numpy() * 255).astype(np.uint8)
                selected[label] = (image, raw_pixels, idx)
                print(f"    Digit {label}: test image #{idx} (confidence: {torch.softmax(output, dim=1)[0, label]:.4f})")
            
            if len(selected) == 10:
                break
    
    if len(selected) < 10:
        missing = [d for d in range(10) if d not in selected]
        print(f"\n  WARNING: Could not find correct predictions for digits: {missing}")
        print("  Exporting what we have.")

    # Sort by digit (0-9)
    digits = sorted(selected.keys())
    
    # Export images as CSV (each row = 784 pixels, 0-255)
    output_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(output_dir, exist_ok=True)
    
    # test_images.csv: 10 rows × 784 columns
    images_array = np.zeros((len(digits), 784), dtype=np.int64)
    labels_array = np.zeros(len(digits), dtype=np.int64)
    
    for i, digit in enumerate(digits):
        _, raw_pixels, _ = selected[digit]
        images_array[i] = raw_pixels.flatten().astype(np.int64)
        labels_array[i] = digit
    
    images_path = os.path.join(output_dir, "test_images.csv")
    np.savetxt(images_path, images_array, fmt="%d", delimiter=",")
    print(f"\n  Saved: {images_path}")
    print(f"    Shape: {images_array.shape} ({len(digits)} images × 784 pixels)")
    
    labels_path = os.path.join(output_dir, "test_labels.csv")
    np.savetxt(labels_path, labels_array.reshape(1, -1), fmt="%d", delimiter=",")
    print(f"  Saved: {labels_path}")
    print(f"    Labels: {labels_array.tolist()}")
    
    # Verify: run quantized inference on selected images
    print("\n  Verification (quantized integer inference):")
    scale_factor = 1000
    
    # Load quantized weights
    state_dict = model.state_dict()
    quantized_model = HE_CNN()
    q_state = quantized_model.state_dict()
    
    layer_map = [
        ("conv1.weight", [1, 1, 5, 5]),
        ("conv1.bias",   [1]),
        ("conv2.weight", [1, 1, 5, 5]),
        ("conv2.bias",   [1]),
        ("fc.weight",    [10, 16]),
        ("fc.bias",      [10]),
    ]
    
    for param_name, shape in layer_map:
        float_w = state_dict[param_name].numpy()
        int_w = np.round(float_w * scale_factor).astype(np.int64)
        q_state[param_name] = torch.from_numpy(
            (int_w.astype(np.float64) / scale_factor).astype(np.float32).reshape(shape)
        )
    
    quantized_model.load_state_dict(q_state)
    quantized_model.eval()
    
    correct = 0
    print(f"\n    {'Digit':>5s}  {'Float':>8s}  {'Quantized':>10s}  {'Match':>6s}")
    print(f"    {'─'*5}  {'─'*8}  {'─'*10}  {'─'*6}")
    
    with torch.no_grad():
        for i, digit in enumerate(digits):
            image_tensor, _, _ = selected[digit]
            
            # Float model prediction
            float_pred = model(image_tensor.unsqueeze(0)).argmax(dim=1).item()
            
            # Quantized model prediction
            quant_pred = quantized_model(image_tensor.unsqueeze(0)).argmax(dim=1).item()
            
            match = "✓" if float_pred == quant_pred else "✗"
            if quant_pred == digit:
                correct += 1
            
            print(f"    {digit:5d}  {float_pred:8d}  {quant_pred:10d}  {match:>6s}")
    
    print(f"\n    Quantized accuracy on selected: {correct}/{len(digits)}")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Step 6d: Test Image Export COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Files exported to: {output_dir}")
    print(f"    test_images.csv  — {len(digits)} images (784 pixels each)")
    print(f"    test_labels.csv  — {len(digits)} ground-truth labels")
    print(f"  Format: raw 0-255 pixel values (NOT scaled)")
    print(f"  Rust pipeline should scale: round(pixel / 255 * {scale_factor})")
    print(f"\n  Next: Run 'cargo run --example mnist_verify' in Docker")


if __name__ == "__main__":
    export_test_images()
