#!/bin/bash
# Train FluxNet Q-table for DSJC250.5
# This creates a pretrained Q-table for aggressive world-record attempts

set -e

echo "=== FluxNet Q-Table Training for DSJC250.5 ==="
echo ""

# Configuration
GRAPH="benchmarks/dimacs/DSJC250.5.col"
OUTPUT="profiles/curriculum/qtable_dsjc250.bin"
EPOCHS=1000
LOG_LEVEL="info"

# Check if graph file exists
if [ ! -f "$GRAPH" ]; then
    echo "ERROR: Graph file not found: $GRAPH"
    echo "Please download DSJC250.5.col from DIMACS benchmark suite"
    exit 1
fi

# Create output directory
mkdir -p profiles/curriculum
mkdir -p results

echo "Configuration:"
echo "  Graph: $GRAPH"
echo "  Epochs: $EPOCHS"
echo "  Output: $OUTPUT"
echo ""

# Build the training binary
echo "Building fluxnet_train binary..."
cargo build --release --bin fluxnet_train

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi

echo "Build successful!"
echo ""

# Run training
echo "Starting training (this may take several minutes)..."
echo ""

RUST_LOG=$LOG_LEVEL ./target/release/fluxnet_train \
    "$GRAPH" \
    $EPOCHS \
    "$OUTPUT"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Training failed"
    exit 1
fi

echo ""
echo "=== Training Complete ==="
echo ""
echo "Q-table saved to: $OUTPUT"
echo "JSON version: ${OUTPUT%.bin}.json"
echo ""
echo "Usage with prism-cli:"
echo "  ./target/release/prism-cli \\"
echo "    --input $GRAPH \\"
echo "    --warmstart \\"
echo "    --gpu \\"
echo "    --config configs/dsjc250_aggressive.toml \\"
echo "    --fluxnet-qtable $OUTPUT"
echo ""
