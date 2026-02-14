"""
MNIST CNN Training for Homomorphic Encryption Inference
========================================================

This script trains a simple CNN on MNIST and exports integer-quantized weights
for use with the OpenFHE encrypted inference pipeline.

Architecture (must match HE ops from Week 1):
  Conv1 (5×5, 1→1 channel) → x² activation → AvgPool 2×2
  Conv2 (5×5, 1→1 channel) → x² activation → AvgPool 2×2
  Flatten (4×4=16) → FC (16→10)

Constraints:
  - Single channel (no multi-channel conv support in current HE ops)
  - x² activation (not ReLU — HE can't do comparisons)
  - Average pooling (not max pool — HE can't do comparisons)
  - Integer weights (BFV scheme operates on integers)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

# ============================================================================
# Step 1: Prepare the Dataset
# ============================================================================

def prepare_dataset():
    """
    Download and prepare MNIST dataset.
    
    MNIST details:
      - 28×28 grayscale images of handwritten digits (0-9)
      - 60,000 training images, 10,000 test images
      - Pixel values: 0-255 (uint8)
    
    Preprocessing for HE compatibility:
      - Keep pixel values as integers (0-255 range) since BFV uses integers
      - We use a simple normalization: scale to 0-100 range (integer-friendly)
      - No data augmentation (keep it simple for the benchmark)
    
    Why integer-friendly preprocessing?
      Our OpenFHE pipeline uses BFV scheme which operates on integers.
      Standard PyTorch normalizes to float (mean=0.1307, std=0.3081),
      but we need to think in integers. We train with floats for gradient
      descent to work, then quantize everything at export time.
    
    Returns:
        train_loader: DataLoader for training set (60,000 images)
        test_loader: DataLoader for test set (10,000 images)
    """
    
    # Data directory
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Transform: convert to tensor (scales 0-255 → 0.0-1.0 automatically)
    # We keep standard float training for now — quantization happens at export (Step 5)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image (0-255) to FloatTensor (0.0-1.0)
    ])
    
    # Download and load MNIST
    print("Downloading MNIST dataset...")
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"  Training set: {len(train_dataset)} images")
    print(f"  Test set:     {len(test_dataset)} images")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0  # Keep simple for compatibility
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader


def verify_dataset(train_loader, test_loader):
    """
    Verify dataset is loaded correctly and visualize a few samples.
    
    This prints statistics and saves a sample grid image so you can
    visually confirm everything looks right before training.
    """
    
    # Get one batch
    images, labels = next(iter(train_loader))
    
    print(f"\n  Batch shape:  {images.shape}")  # [64, 1, 28, 28]
    print(f"  Label shape:  {labels.shape}")    # [64]
    print(f"  Pixel range:  [{images.min():.4f}, {images.max():.4f}]")
    print(f"  Pixel dtype:  {images.dtype}")
    print(f"  Label sample: {labels[:10].tolist()}")
    
    # Image dimensions
    print(f"\n  Image size:   {images.shape[2]}×{images.shape[3]} (height×width)")
    print(f"  Channels:     {images.shape[1]} (grayscale)")
    
    # Class distribution in training set
    all_labels = []
    for _, batch_labels in train_loader:
        all_labels.extend(batch_labels.tolist())
    
    print(f"\n  Class distribution (training set):")
    for digit in range(10):
        count = all_labels.count(digit)
        print(f"    Digit {digit}: {count:5d} images ({count/len(all_labels)*100:.1f}%)")
    
    # Save sample images
    output_dir = os.path.dirname(__file__)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("MNIST Sample Images", fontsize=14)
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.set_title(f"Label: {labels[i].item()}", fontsize=10)
        ax.axis("off")
    
    sample_path = os.path.join(output_dir, "sample_images.png")
    plt.tight_layout()
    plt.savefig(sample_path, dpi=100)
    plt.close()
    print(f"\n  Sample images saved to: {sample_path}")
    
    # Verify dimensions match HE pipeline expectations
    print(f"\n  HE Pipeline Compatibility Check:")
    print(f"    Input:  28×28 = 784 pixels  ✓ (fits in BFV slots)")
    print(f"    Conv1:  28×28 → 24×24 (5×5 kernel, valid padding)")
    print(f"    Pool1:  24×24 → 12×12 (2×2 avg pool, stride 2)")
    print(f"    Conv2:  12×12 → 8×8   (5×5 kernel, valid padding)")
    print(f"    Pool2:  8×8   → 4×4   (2×2 avg pool, stride 2)")
    print(f"    FC:     4×4=16 → 10   (10 output classes)")


# ============================================================================
# Step 2a: Custom x² Activation Function
# ============================================================================

class SquareActivation(nn.Module):
    """
    Square activation function: f(x) = x²
    
    Why x² instead of ReLU?
      HE (homomorphic encryption) only supports addition and multiplication
      on encrypted data. ReLU requires a comparison (if x < 0, set to 0),
      which is impossible on encrypted values.
      
      x² is a valid alternative because:
        - It only requires one multiplication (HE-friendly)
        - Outputs are always non-negative (like ReLU)
        - It's used in CryptoNets (Gilad-Bachrach et al., ICML 2016)
      
      The model must be TRAINED with x² so the weights learn to work
      with this activation. You can't train with ReLU and swap to x²
      at inference — the accuracy would collapse.
    
    Maps to: openfhe_poly_relu() in openfhe_cnn_ops.cpp
    """
    
    def forward(self, x):
        return x * x


def verify_activation():
    """Test the square activation on known values."""
    act = SquareActivation()
    
    test_input = torch.tensor([-3.0, -1.0, 0.0, 1.0, 2.0, 5.0])
    expected   = torch.tensor([ 9.0,  1.0, 0.0, 1.0, 4.0, 25.0])
    output = act(test_input)
    
    match = torch.allclose(output, expected)
    print(f"  SquareActivation test:")
    print(f"    Input:    {test_input.tolist()}")
    print(f"    Output:   {output.tolist()}")
    print(f"    Expected: {expected.tolist()}")
    print(f"    Match:    {'✓' if match else '✗'}")
    
    # Verify gradient flows (needed for backprop during training)
    x = torch.tensor([2.0, -3.0], requires_grad=True)
    y = act(x)
    y.sum().backward()
    # d/dx(x²) = 2x, so gradients should be [4.0, -6.0]
    expected_grad = torch.tensor([4.0, -6.0])
    grad_match = torch.allclose(x.grad, expected_grad)
    print(f"    Gradient: {x.grad.tolist()} (expected {expected_grad.tolist()}) {'✓' if grad_match else '✗'}")
    
    return match and grad_match


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MNIST CNN Training for Homomorphic Encryption")
    print("=" * 70)
    
    # Step 1: Prepare dataset
    print("\nStep 1: Preparing MNIST Dataset")
    print("-" * 40)
    train_loader, test_loader = prepare_dataset()
    verify_dataset(train_loader, test_loader)
    
    print("\n" + "=" * 70)
    print("Step 1 Complete: Dataset ready for training")
    print("=" * 70)
    
    # Step 2a: Custom activation
    print("\nStep 2a: Square Activation Function (x²)")
    print("-" * 40)
    activation_ok = verify_activation()
    if not activation_ok:
        print("  ERROR: Activation function test failed!")
        exit(1)
    print("  Step 2a Complete: x² activation verified ✓")
    
    # Step 2b: Build model      (TODO)
    # Step 2c: Verify forward pass (TODO)
    # Step 3: Train model       (TODO)
    # Step 4: Evaluate accuracy  (TODO)
    # Step 5: Export weights     (TODO)
