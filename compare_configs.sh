#!/bin/bash

# Compare different configurations on same graph

GRAPH="${1:-benchmarks/dimacs/DSJC125.5.col}"
echo "Testing graph: $GRAPH"
echo ""

export PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6

echo "1. ULTRA_FAST (30 seconds):"
timeout 30 ./target/release/prism-cli -i "$GRAPH" --config configs/ULTRA_FAST_17.toml --attempts 1 2>&1 | grep "Best chromatic"

echo ""
echo "2. FAST_17 (60 seconds):"
timeout 60 ./target/release/prism-cli -i "$GRAPH" --config configs/FAST_17.toml --attempts 1 2>&1 | grep "Best chromatic"

echo ""
echo "3. AGGRESSIVE_17 (90 seconds):"
timeout 90 ./target/release/prism-cli -i "$GRAPH" --config configs/AGGRESSIVE_17.toml --attempts 1 2>&1 | grep "Best chromatic"

echo ""
echo "4. FULL_POWER_17 (120 seconds):"
timeout 120 ./target/release/prism-cli -i "$GRAPH" --config configs/FULL_POWER_17.toml --attempts 1 2>&1 | grep "Best chromatic"

echo ""
echo "5. EXTREME_MAX (no limit):"
./target/release/prism-cli -i "$GRAPH" --config configs/EXTREME_MAX.toml --attempts 1 2>&1 | grep "Best chromatic"