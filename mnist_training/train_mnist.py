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
# Step 2b: Define the CNN Model
# ============================================================================

class HE_CNN(nn.Module):
    """
    Simple CNN designed for homomorphic encryption inference.
    
    Architecture:
      Conv1 (5×5, 1→1) → x² → AvgPool(2×2)
      Conv2 (5×5, 1→1) → x² → AvgPool(2×2)
      Flatten → FC (16→10)
    
    Dimension flow:
      Input:  [batch, 1, 28, 28]
      Conv1:  [batch, 1, 24, 24]  (28-5+1=24)
      Act1:   [batch, 1, 24, 24]
      Pool1:  [batch, 1, 12, 12]  (24/2=12)
      Conv2:  [batch, 1,  8,  8]  (12-5+1=8)
      Act2:   [batch, 1,  8,  8]
      Pool2:  [batch, 1,  4,  4]  (8/2=4)
      Flat:   [batch, 16]         (1×4×4=16)
      FC:     [batch, 10]
    
    Constraints for HE compatibility:
      - Single channel only (current conv2d op doesn't support multi-channel)
      - x² activation (no ReLU — HE can't do comparisons)
      - Average pooling (no max pool — HE can't do comparisons)
      - No batch normalization (not supported in HE)
      - No dropout (not needed for HE inference)
    
    HE operation mapping:
      Conv1/Conv2 → openfhe_conv2d()     (5×5 kernel, valid padding)
      Act1/Act2   → openfhe_poly_relu()  (x² square activation)
      Pool1/Pool2 → openfhe_avgpool()    (2×2, stride 2)
      FC          → openfhe_matmul()     (16→10 matrix-vector multiply)
    
    Total parameters: Conv1(25+1) + Conv2(25+1) + FC(160+10) = 222
    """
    
    def __init__(self):
        super(HE_CNN, self).__init__()
        
        # Conv1: 1 input channel, 1 output channel, 5×5 kernel
        # No padding (valid convolution) — matches openfhe_conv2d behavior
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=5,
            padding=0,
            bias=True
        )
        
        # Square activation (replaces ReLU)
        self.act1 = SquareActivation()
        
        # Average pooling 2×2, stride 2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Conv2: 1 input channel, 1 output channel, 5×5 kernel
        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=5,
            padding=0,
            bias=True
        )
        
        # Square activation
        self.act2 = SquareActivation()
        
        # Average pooling 2×2, stride 2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Fully connected: 1×4×4 = 16 inputs → 10 outputs (digits 0-9)
        self.fc = nn.Linear(16, 10, bias=True)
    
    def forward(self, x):
        """
        Forward pass with shape tracking.
        
        Args:
            x: Input tensor [batch, 1, 28, 28]
        Returns:
            Output logits [batch, 10]
        """
        # Conv1 block
        x = self.conv1(x)    # [batch, 1, 28, 28] → [batch, 1, 24, 24]
        x = self.act1(x)     # [batch, 1, 24, 24] → [batch, 1, 24, 24]
        x = self.pool1(x)    # [batch, 1, 24, 24] → [batch, 1, 12, 12]
        
        # Conv2 block
        x = self.conv2(x)    # [batch, 1, 12, 12] → [batch, 1, 8, 8]
        x = self.act2(x)     # [batch, 1, 8, 8]   → [batch, 1, 8, 8]
        x = self.pool2(x)    # [batch, 1, 8, 8]   → [batch, 1, 4, 4]
        
        # Flatten and classify
        x = x.view(x.size(0), -1)  # [batch, 1, 4, 4] → [batch, 16]
        x = self.fc(x)             # [batch, 16]       → [batch, 10]
        
        return x


# ============================================================================
# Step 2c: Verify Forward Pass
# ============================================================================

def verify_model(model):
    """
    Verify the model dimensions with a dummy forward pass.
    
    Feeds a random 28×28 image through every layer and prints
    the shape at each stage. Catches dimension mismatches before training.
    """
    
    # Print model architecture
    print(f"  Model Architecture:")
    print(f"  {model}\n")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters:")
    for name, param in model.named_parameters():
        print(f"    {name:20s} shape={str(list(param.shape)):15s} count={param.numel()}")
    print(f"    {'TOTAL':20s} {'':15s} count={total_params}")
    print(f"    Trainable: {trainable_params}")
    
    # Dummy forward pass with shape tracking
    print(f"\n  Forward Pass (dummy 28×28 input):")
    x = torch.randn(1, 1, 28, 28)
    print(f"    Input:       {list(x.shape)}")
    
    x = model.conv1(x)
    print(f"    After Conv1: {list(x.shape)}")
    
    x = model.act1(x)
    print(f"    After Act1:  {list(x.shape)}")
    
    x = model.pool1(x)
    print(f"    After Pool1: {list(x.shape)}")
    
    x = model.conv2(x)
    print(f"    After Conv2: {list(x.shape)}")
    
    x = model.act2(x)
    print(f"    After Act2:  {list(x.shape)}")
    
    x = model.pool2(x)
    print(f"    After Pool2: {list(x.shape)}")
    
    x = x.view(x.size(0), -1)
    print(f"    After Flat:  {list(x.shape)}")
    
    x = model.fc(x)
    print(f"    After FC:    {list(x.shape)}")
    
    # Verify output shape
    expected_shape = [1, 10]
    shape_ok = list(x.shape) == expected_shape
    print(f"\n  Output shape: {list(x.shape)} (expected {expected_shape}) {'✓' if shape_ok else '✗'}")
    
    return shape_ok


# ============================================================================
# Step 3a: Training Function
# ============================================================================

def train_one_epoch(model, train_loader, optimizer, criterion, epoch):
    """
    Train the model for one epoch.
    
    Args:
        model: HE_CNN model
        train_loader: DataLoader for training data
        optimizer: Adam optimizer
        criterion: CrossEntropyLoss
        epoch: Current epoch number (for display)
    
    Returns:
        avg_loss: Average loss over all batches in this epoch
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    total_batches = len(train_loader)
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track loss
        running_loss += loss.item()
        
        # Print progress every 200 batches
        if (batch_idx + 1) % 200 == 0:
            avg_so_far = running_loss / (batch_idx + 1)
            print(f"    Epoch {epoch+1} [{batch_idx+1:4d}/{total_batches}]  loss: {avg_so_far:.4f}")
    
    avg_loss = running_loss / total_batches
    return avg_loss


# ============================================================================
# Step 3b: Evaluation Function
# ============================================================================

def evaluate(model, test_loader):
    """
    Evaluate model accuracy on the test set.
    
    Runs inference on all test images with no gradient computation
    (faster, less memory). Returns overall accuracy and per-digit accuracy.
    
    Args:
        model: HE_CNN model
        test_loader: DataLoader for test data (10,000 images)
    
    Returns:
        accuracy: Overall accuracy as a percentage (0-100)
        per_digit: Dict mapping digit → accuracy percentage
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    
    correct = 0
    total = 0
    per_digit_correct = [0] * 10
    per_digit_total = [0] * 10
    
    with torch.no_grad():  # No gradients needed — saves memory and time
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)  # Get digit with highest score
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-digit tracking
            for digit in range(10):
                mask = (labels == digit)
                per_digit_total[digit] += mask.sum().item()
                per_digit_correct[digit] += (predicted[mask] == digit).sum().item()
    
    accuracy = 100.0 * correct / total
    per_digit = {}
    for digit in range(10):
        if per_digit_total[digit] > 0:
            per_digit[digit] = 100.0 * per_digit_correct[digit] / per_digit_total[digit]
        else:
            per_digit[digit] = 0.0
    
    return accuracy, per_digit


# ============================================================================
# Step 3c: Full Training Loop
# ============================================================================

def train_full(model, train_loader, test_loader, num_epochs=10, lr=0.001):
    """
    Train the model for multiple epochs, tracking loss and accuracy.
    
    Saves the best model (highest test accuracy) and generates a
    training curve plot showing loss and accuracy over epochs.
    
    Args:
        model: HE_CNN model
        train_loader: Training data
        test_loader: Test data
        num_epochs: Number of training epochs (default: 10)
        lr: Learning rate for Adam optimizer (default: 0.001)
    
    Returns:
        model: Trained model (with best weights loaded)
        history: Dict with 'loss', 'accuracy' lists per epoch
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = {"loss": [], "accuracy": []}
    best_accuracy = 0.0
    best_state = None
    output_dir = os.path.dirname(__file__)
    
    print(f"  Config:")
    print(f"    Optimizer:  Adam (lr={lr})")
    print(f"    Loss:       CrossEntropyLoss")
    print(f"    Epochs:     {num_epochs}")
    print(f"    Batch size: {train_loader.batch_size}")
    print()
    
    for epoch in range(num_epochs):
        # Train one epoch
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        
        # Evaluate on test set
        accuracy, per_digit = evaluate(model, test_loader)
        
        # Record history
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)
        
        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " ★ best"
        else:
            marker = ""
        
        print(f"  Epoch {epoch+1:2d}/{num_epochs}  "
              f"loss: {avg_loss:.4f}  "
              f"accuracy: {accuracy:.2f}%{marker}")
    
    # Load best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Save model checkpoint
    model_path = os.path.join(output_dir, "he_cnn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n  Best accuracy: {best_accuracy:.2f}%")
    print(f"  Model saved to: {model_path}")
    
    # Plot training curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs_range = range(1, num_epochs + 1)
    
    ax1.plot(epochs_range, history["loss"], "b-o", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_range, history["accuracy"], "r-o", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Test Accuracy")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curve_path = os.path.join(output_dir, "training_curve.png")
    plt.savefig(curve_path, dpi=100)
    plt.close()
    print(f"  Training curve saved to: {curve_path}")
    
    return model, history


# ============================================================================
# Step 4a: Final Accuracy Report
# ============================================================================

def evaluate_final(model, test_loader):
    """
    Generate a final accuracy report with per-digit breakdown and bar chart.
    
    Loads the best model weights (already loaded after train_full) and runs
    evaluate() to get overall + per-digit accuracy. Prints a formatted
    report and saves a per-digit accuracy bar chart.
    
    Args:
        model: Trained HE_CNN model (best weights already loaded)
        test_loader: DataLoader for test data (10,000 images)
    """
    output_dir = os.path.dirname(__file__)
    
    # Run evaluation
    accuracy, per_digit = evaluate(model, test_loader)
    
    # Print overall accuracy
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  Overall Test Accuracy: {accuracy:6.2f}%      │")
    print(f"  └─────────────────────────────────────┘\n")
    
    # Print per-digit breakdown
    print(f"  Per-Digit Accuracy:")
    print(f"  {'Digit':>5s}  {'Accuracy':>8s}  {'Bar'}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*30}")
    
    for digit in range(10):
        acc = per_digit[digit]
        bar_len = int(acc / 100.0 * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {digit:5d}  {acc:7.2f}%  {bar}")
    
    # Summary statistics
    accs = list(per_digit.values())
    best_digit = max(per_digit, key=per_digit.get)
    worst_digit = min(per_digit, key=per_digit.get)
    spread = max(accs) - min(accs)
    
    print(f"\n  Best digit:   {best_digit} ({per_digit[best_digit]:.2f}%)")
    print(f"  Worst digit:  {worst_digit} ({per_digit[worst_digit]:.2f}%)")
    print(f"  Spread:       {spread:.2f}% (best − worst)")
    print(f"  Mean per-digit: {np.mean(accs):.2f}%")
    
    # Save per-digit accuracy bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    
    digits = list(range(10))
    accuracies = [per_digit[d] for d in digits]
    colors = ["#2ecc71" if a >= 90 else "#f39c12" if a >= 80 else "#e74c3c" for a in accuracies]
    
    bars = ax.bar(digits, accuracies, color=colors, edgecolor="white", linewidth=0.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    # Add overall accuracy line
    ax.axhline(y=accuracy, color="#3498db", linestyle="--", linewidth=1.5, label=f"Overall: {accuracy:.2f}%")
    ax.legend(fontsize=11, loc="lower right")
    
    ax.set_xlabel("Digit", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Digit Test Accuracy (HE-Compatible CNN)", fontsize=14)
    ax.set_xticks(digits)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "per_digit_accuracy.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n  Per-digit chart saved to: {chart_path}")


# ============================================================================
# Step 4b: Confusion Matrix
# ============================================================================

def confusion_matrix_report(model, test_loader):
    """
    Build and visualize a 10×10 confusion matrix.
    
    The confusion matrix shows how predictions map to true labels:
      - Rows    = true digit (what the image actually is)
      - Columns = predicted digit (what the model guessed)
      - Diagonal = correct predictions
      - Off-diagonal = misclassifications
    
    This helps identify which digit pairs the model confuses most,
    which is useful for understanding HE inference errors later.
    
    Args:
        model: Trained HE_CNN model
        test_loader: DataLoader for test data (10,000 images)
    """
    output_dir = os.path.dirname(__file__)
    
    # Build confusion matrix
    cm = np.zeros((10, 10), dtype=int)
    model.eval()
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            
            for true_label, pred_label in zip(labels, predicted):
                cm[true_label.item()][pred_label.item()] += 1
    
    # Print text confusion matrix
    print(f"\n  Confusion Matrix (rows=true, cols=predicted):\n")
    
    # Header
    header = "       " + "".join(f"{d:>6d}" for d in range(10))
    print(f"  {header}")
    print(f"  {'─' * 67}")
    
    for true_digit in range(10):
        row = f"  {true_digit:>4d} │"
        for pred_digit in range(10):
            count = cm[true_digit][pred_digit]
            if true_digit == pred_digit:
                row += f"{count:>6d}"  # Diagonal (correct)
            elif count > 0:
                row += f"{count:>6d}"  # Misclassification
            else:
                row += f"{'·':>6s}"   # Zero
        
        # Row accuracy
        row_total = cm[true_digit].sum()
        row_correct = cm[true_digit][true_digit]
        row_acc = 100.0 * row_correct / row_total if row_total > 0 else 0
        row += f"  │ {row_acc:5.1f}%"
        print(row)
    
    print(f"  {'─' * 67}")
    
    # Find top misclassification pairs
    print(f"\n  Top Misclassifications:")
    
    # Get off-diagonal entries sorted by count
    misclass = []
    for true_d in range(10):
        for pred_d in range(10):
            if true_d != pred_d and cm[true_d][pred_d] > 0:
                misclass.append((cm[true_d][pred_d], true_d, pred_d))
    
    misclass.sort(reverse=True)
    
    for count, true_d, pred_d in misclass[:10]:
        pct = 100.0 * count / cm[true_d].sum()
        print(f"    {true_d} → {pred_d}: {count:4d} times ({pct:.1f}% of digit {true_d})")
    
    # Plot confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Normalize for color (percentage per row)
    cm_pct = cm.astype(float)
    row_sums = cm_pct.sum(axis=1, keepdims=True)
    cm_pct = np.zeros_like(cm_pct)
    np.divide(cm.astype(float), row_sums, out=cm_pct, where=row_sums != 0)
    cm_pct *= 100
    
    im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    
    # Add text annotations
    for i in range(10):
        for j in range(10):
            count = cm[i][j]
            pct = cm_pct[i][j]
            if count > 0:
                color = "white" if pct > 50 else "black"
                ax.text(j, i, f"{count}\n({pct:.0f}%)",
                        ha="center", va="center", fontsize=7, color=color)
    
    ax.set_xlabel("Predicted Digit", fontsize=12)
    ax.set_ylabel("True Digit", fontsize=12)
    ax.set_title("Confusion Matrix (HE-Compatible CNN)", fontsize=14)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Percentage (%)", fontsize=10)
    
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n  Confusion matrix saved to: {cm_path}")


# ============================================================================
# Step 4c: Sample Predictions
# ============================================================================

def sample_predictions(model, test_loader):
    """
    Visualize sample correct and incorrect predictions for each digit.
    
    Creates a 10×2 grid:
      - Left column:  one correctly classified example per digit
      - Right column: one incorrectly classified example per digit (if any)
    
    Each cell shows the image, true label, predicted label, and confidence.
    This gives a quick visual sanity check of what the model gets right
    and wrong, and is useful for comparing with HE inference results later.
    
    Args:
        model: Trained HE_CNN model
        test_loader: DataLoader for test data (10,000 images)
    """
    output_dir = os.path.dirname(__file__)
    model.eval()
    
    # Collect one correct and one incorrect example per digit
    correct_samples = {}   # digit → (image, predicted, confidence)
    incorrect_samples = {} # digit → (image, predicted, confidence)
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probs, dim=1)
            
            for i in range(images.size(0)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                conf = confidences[i].item() * 100
                img = images[i]
                
                if true_label == pred_label and true_label not in correct_samples:
                    correct_samples[true_label] = (img, pred_label, conf)
                
                if true_label != pred_label and true_label not in incorrect_samples:
                    incorrect_samples[true_label] = (img, pred_label, conf)
            
            # Stop early if we have all samples
            if len(correct_samples) == 10 and len(incorrect_samples) >= 8:
                break
    
    # Print summary
    print(f"\n  Correct examples found:   {len(correct_samples)}/10 digits")
    print(f"  Incorrect examples found: {len(incorrect_samples)}/10 digits")
    
    # Print text summary of incorrect predictions
    if incorrect_samples:
        print(f"\n  Misclassified Examples:")
        for digit in range(10):
            if digit in incorrect_samples:
                _, pred, conf = incorrect_samples[digit]
                print(f"    Digit {digit} → predicted {pred} (confidence: {conf:.1f}%)")
            else:
                print(f"    Digit {digit} → (all correct in sample)")
    
    # Create visualization: 10 rows × 2 columns (correct | incorrect)
    fig, axes = plt.subplots(10, 2, figsize=(6, 24))
    fig.suptitle("Sample Predictions: Correct vs Incorrect", fontsize=14, y=0.995)
    
    # Column headers
    axes[0, 0].set_title("✓ Correct", fontsize=12, fontweight="bold", color="#2ecc71")
    axes[0, 1].set_title("✗ Incorrect", fontsize=12, fontweight="bold", color="#e74c3c")
    
    for digit in range(10):
        # Left column: correct prediction
        ax_correct = axes[digit, 0]
        if digit in correct_samples:
            img, pred, conf = correct_samples[digit]
            ax_correct.imshow(img.squeeze(), cmap="gray")
            ax_correct.set_ylabel(f"Digit {digit}", fontsize=10, fontweight="bold")
            ax_correct.set_xlabel(f"pred={pred} ({conf:.0f}%)", fontsize=8, color="#2ecc71")
        else:
            ax_correct.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
            ax_correct.set_ylabel(f"Digit {digit}", fontsize=10, fontweight="bold")
        ax_correct.set_xticks([])
        ax_correct.set_yticks([])
        
        # Right column: incorrect prediction
        ax_incorrect = axes[digit, 1]
        if digit in incorrect_samples:
            img, pred, conf = incorrect_samples[digit]
            ax_incorrect.imshow(img.squeeze(), cmap="gray")
            ax_incorrect.set_xlabel(f"pred={pred} ({conf:.0f}%)", fontsize=8, color="#e74c3c")
        else:
            ax_incorrect.text(0.5, 0.5, "All correct!", ha="center", va="center",
                            fontsize=9, color="#2ecc71", fontweight="bold")
        ax_incorrect.set_xticks([])
        ax_incorrect.set_yticks([])
    
    plt.tight_layout()
    pred_path = os.path.join(output_dir, "sample_predictions.png")
    plt.savefig(pred_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Sample predictions saved to: {pred_path}")


# ============================================================================
# Step 5a: Quantize Weights to Integers
# ============================================================================

def quantize_weights(model, scale_factor=1000):
    """
    Quantize float32 model weights to integers for BFV encryption.
    
    BFV (Brakerski-Fan-Vercauteren) operates on integers modulo a plaintext
    modulus. Our OpenFHE setup uses plaintext_modulus = 7340033, so integer
    weights must fit within that range.
    
    Quantization process:
      1. Take float weight w (e.g., 0.0342)
      2. Multiply by scale_factor (e.g., 1000 → 34.2)
      3. Round to nearest integer (→ 34)
    
    The same scale_factor must be applied to input pixels at inference time.
    After each layer, intermediate values grow by scale_factor, so we need
    to track the cumulative scale through the network.
    
    Scale tracking through layers:
      Input pixels:  scaled by S (scale_factor)
      After Conv1:   S × S = S² (input × kernel, both scaled)
      After x²:     S² × S² = S⁴
      After Pool:   S⁴ (averaging doesn't change scale)
      After Conv2:  S⁴ × S = S⁵
      After x²:    S⁵ × S⁵ = S¹⁰
      After Pool:  S¹⁰
      After FC:    S¹⁰ × S = S¹¹
    
    Note: The scale grows exponentially! With scale=1000, S¹¹ overflows.
    In practice, the HE pipeline applies rescaling after each operation.
    For weight export, we just need each layer's weights × scale_factor.
    
    Args:
        model: Trained HE_CNN model with float32 weights
        scale_factor: Integer multiplier (default: 1000)
    
    Returns:
        quantized: Dict with quantized weights for each layer
                   Keys: 'conv1_weight', 'conv1_bias', 'conv2_weight',
                         'conv2_bias', 'fc_weight', 'fc_bias',
                         'scale_factor'
    """
    print(f"\n  Scale factor: {scale_factor}")
    print(f"  Plaintext modulus: 7340033 (max integer in BFV)")
    print(f"  Max representable value: ±{7340033 // 2}\n")
    
    quantized = {"scale_factor": scale_factor}
    
    # Layer mapping: param_name → (display_name, key)
    layers = [
        ("conv1.weight", "Conv1 kernel", "conv1_weight"),
        ("conv1.bias",   "Conv1 bias",   "conv1_bias"),
        ("conv2.weight", "Conv2 kernel", "conv2_weight"),
        ("conv2.bias",   "Conv2 bias",   "conv2_bias"),
        ("fc.weight",    "FC weights",   "fc_weight"),
        ("fc.bias",      "FC bias",      "fc_bias"),
    ]
    
    state_dict = model.state_dict()
    
    print(f"  {'Layer':<15s} {'Shape':>15s} {'Float range':>25s} {'Int range':>20s} {'Max |int|':>10s}")
    print(f"  {'─'*15}  {'─'*15} {'─'*25} {'─'*20} {'─'*10}")
    
    for param_name, display_name, key in layers:
        float_weights = state_dict[param_name].numpy()
        
        # Quantize: multiply by scale factor and round
        int_weights = np.round(float_weights * scale_factor).astype(np.int64)
        
        quantized[key] = int_weights
        
        # Statistics
        f_min, f_max = float_weights.min(), float_weights.max()
        i_min, i_max = int_weights.min(), int_weights.max()
        max_abs = max(abs(i_min), abs(i_max))
        
        shape_str = str(list(float_weights.shape))
        float_range = f"[{f_min:+.6f}, {f_max:+.6f}]"
        int_range = f"[{i_min:+d}, {i_max:+d}]"
        
        print(f"  {display_name:<15s} {shape_str:>15s} {float_range:>25s} {int_range:>20s} {max_abs:>10d}")
    
    # Check if any values exceed plaintext modulus
    max_half = 7340033 // 2
    overflow = False
    for param_name, display_name, key in layers:
        max_val = max(abs(quantized[key].min()), abs(quantized[key].max()))
        if max_val > max_half:
            print(f"\n  ⚠ WARNING: {display_name} max |value| = {max_val} exceeds ±{max_half}")
            overflow = True
    
    if not overflow:
        print(f"\n  ✓ All quantized values fit within plaintext modulus (±{max_half})")
    
    # Show quantization error
    print(f"\n  Quantization Error (per layer):")
    for param_name, display_name, key in layers:
        float_weights = state_dict[param_name].numpy()
        # Reconstruct: int / scale → approximate float
        reconstructed = quantized[key].astype(np.float64) / scale_factor
        error = np.abs(float_weights - reconstructed)
        
        print(f"    {display_name:<15s}  "
              f"mean_err={error.mean():.6f}  "
              f"max_err={error.max():.6f}  "
              f"(max is {error.max() / (abs(float_weights).max() + 1e-10) * 100:.2f}% of max weight)")
    
    # Total parameter count
    total_ints = sum(quantized[key].size for _, _, key in layers)
    print(f"\n  Total quantized parameters: {total_ints}")
    
    return quantized


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
    
    # Step 2b + 2c: Build model and verify forward pass
    print("\nStep 2b: Building CNN Model")
    print("-" * 40)
    model = HE_CNN()
    
    print("\nStep 2c: Verifying Forward Pass")
    print("-" * 40)
    model_ok = verify_model(model)
    if not model_ok:
        print("  ERROR: Model shape verification failed!")
        exit(1)
    print("  Step 2 Complete: Model built and verified ✓")
    
    # Step 3: Train the model (10 epochs)
    print("\nStep 3: Training Model (10 epochs)")
    print("-" * 40)
    model, history = train_full(model, train_loader, test_loader, num_epochs=10, lr=0.001)
    print(f"  Step 3 Complete: Training finished ✓")
    
    # Final accuracy report
    print("\nFinal Accuracy Report")
    print("-" * 40)
    evaluate_final(model, test_loader)
    print("  Accuracy report generated ✓")

    # Confusion matrix
    print("\nConfusion Matrix")
    print("-" * 40)
    confusion_matrix_report(model, test_loader)
    print("  Confusion matrix generated ✓")

    # Sample predictions
    print("\nSample Predictions")
    print("-" * 40)
    sample_predictions(model, test_loader)
    print("  Sample predictions generated ✓")

    # Export weights
    print("\nStep 5a: Quantize Weights")
    print("-" * 40)
    quantized = quantize_weights(model)
    print("  Step 5a Complete: Weights quantized ✓")

    # Step 5b: Save weights to CSV  (TODO)
    # Step 5c: Verify quantized accuracy (TODO)
