#!/usr/bin/env bash
#
# Run the MNIST encrypted inference benchmark for all activation degrees.
#
# This script runs directly on EC2 #2 (compute server) inside the Docker container.
# It benchmarks 100 images × 3 activation degrees (deg2, deg3, deg4).
#
# Usage:
#   chmod +x scripts/run_all_benchmarks.sh
#   ./scripts/run_all_benchmarks.sh
#
# Or inside Docker:
#   docker exec -it <container> bash -c "./scripts/run_all_benchmarks.sh"
#
# Expected runtime: ~100 images × ~27s/image × 3 configs ≈ 2.25 hours total
#

set -euo pipefail

echo "================================================================"
echo "  MNIST FHE Benchmark Suite"
echo "  100 images × 3 activation degrees (deg2, deg3, deg4)"
echo "================================================================"
echo ""

WEIGHT_DIRS=(
    "mnist_training/weights_deg2"
    "mnist_training/weights_deg3"
    "mnist_training/weights_deg4"
)

# Fallback: weights/ is the same as weights_deg2
if [ ! -d "mnist_training/weights_deg2" ] && [ -d "mnist_training/weights" ]; then
    WEIGHT_DIRS[0]="mnist_training/weights"
fi

TOTAL_START=$(date +%s)

for WDIR in "${WEIGHT_DIRS[@]}"; do
    if [ ! -d "$WDIR" ]; then
        echo "SKIP: $WDIR does not exist"
        echo ""
        continue
    fi

    if [ ! -f "$WDIR/test_images_100.csv" ]; then
        echo "SKIP: $WDIR/test_images_100.csv not found"
        echo "Run: cd mnist_training && python export_test_images_100.py"
        echo ""
        continue
    fi

    DEG=$(cat "$WDIR/model_config.json" | grep -o '"activation_degree": [0-9]*' | grep -o '[0-9]*')
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running: Degree $DEG ($WDIR)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    START=$(date +%s)

    cargo run --release --example mnist_benchmark -- \
        --weights "$WDIR" \
        --output "mnist_training/fhe_benchmark_deg${DEG}_128bit.csv"

    END=$(date +%s)
    ELAPSED=$((END - START))
    echo "  Degree $DEG completed in ${ELAPSED}s"
    echo ""
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo "================================================================"
echo "  All benchmarks complete!"
echo "  Total time: ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s)"
echo "================================================================"
echo ""
echo "  Results:"
ls -la mnist_training/fhe_benchmark_*.csv 2>/dev/null || echo "  (no CSV files found)"
echo ""
echo "  Summaries:"
ls -la mnist_training/fhe_benchmark_summary_*.txt 2>/dev/null || echo "  (no summary files found)"
