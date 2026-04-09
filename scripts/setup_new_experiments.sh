#!/usr/bin/env bash
#
# Setup new experiment weight directories for Scale Factor & Plaintext Modulus experiments.
#
# IMPORTANT: Changing scale_factor requires re-quantizing the weights (not just
# editing model_config.json), because the CSV files contain round(float × scale).
# This script uses mnist_training/reexport_weights.py to do it correctly.
#
# Run this ONCE on EC2 #2 (backend compute server) before running the benchmarks.
#
# Usage:
#   chmod +x scripts/setup_new_experiments.sh
#   ./scripts/setup_new_experiments.sh
#
# Prerequisites:
#   pip install torch torchvision numpy   (for weight re-export)
#   mnist_training/he_cnn_model.pth       (trained model file)
#

set -euo pipefail

echo "================================================================"
echo "  Setting up new experiment directories"
echo "  Step 1: Re-export weights (re-quantize with new scale/modulus)"
echo "  Step 2: Copy test images into new directories"
echo "================================================================"
echo ""

# Check prerequisites
if [ ! -f "mnist_training/he_cnn_model.pth" ]; then
    echo "ERROR: mnist_training/he_cnn_model.pth not found!"
    echo "  This is the trained PyTorch model needed for re-quantization."
    echo "  Make sure you're in the project root directory."
    exit 1
fi

if [ ! -d "mnist_training/weights_deg2" ]; then
    echo "ERROR: mnist_training/weights_deg2/ not found!"
    echo "  This is the baseline directory with test images."
    exit 1
fi

# ════════════════════════════════════════════════════════════════════
# Step 1: Run the Python re-export script
# ════════════════════════════════════════════════════════════════════
# This creates:
#   weights_scale100/     — scale_factor=100, re-quantized weight CSVs
#   weights_scale10000/   — scale_factor=10000, re-quantized weight CSVs
#   weights_p65537/       — plaintext_modulus=65537, same scale but different modulus
#   weights_p4294967311/  — plaintext_modulus=4294967311, larger modulus

echo "Running weight re-export..."
echo ""
cd mnist_training
python3 reexport_weights.py
cd ..

echo ""

# ════════════════════════════════════════════════════════════════════
# Step 2: Verify all directories have test images
# ════════════════════════════════════════════════════════════════════
echo "── Verifying test images ──"
echo ""

for DIR in mnist_training/weights_scale100 mnist_training/weights_scale10000 \
           mnist_training/weights_p65537 mnist_training/weights_p4294967311; do
    if [ -d "$DIR" ]; then
        IMG_COUNT=0
        [ -f "$DIR/test_images_100.csv" ] && IMG_COUNT=$(wc -l < "$DIR/test_images_100.csv" | tr -d ' ')
        [ -f "$DIR/test_images.csv" ] && [ "$IMG_COUNT" -eq 0 ] && IMG_COUNT=$(wc -l < "$DIR/test_images.csv" | tr -d ' ')

        SF=$(python3 -c "import json; cfg=json.load(open('$DIR/model_config.json')); print(cfg['quantization']['scale_factor'])")
        PM=$(python3 -c "import json; cfg=json.load(open('$DIR/model_config.json')); print(cfg['quantization']['plaintext_modulus'])")

        echo "  ✓ $DIR"
        echo "    scale_factor=$SF  plaintext_modulus=$PM  test_images=$IMG_COUNT"
    else
        echo "  ✗ $DIR — MISSING"
    fi
done

echo ""
echo "================================================================"
echo "  Setup complete!"
echo "  Next: run  ./scripts/run_new_experiments.sh"
echo "================================================================"
