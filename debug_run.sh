#!/bin/bash

# DEBUG SCRIPT - See what's happening in real-time

export PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6
export RUST_LOG=info

echo "Starting debug run with FAST config..."
echo "Press Ctrl+C to stop if it freezes"
echo ""

# Run with real-time output
timeout 60 ./target/release/prism-cli \
    -i benchmarks/dimacs/DSJC125.5.col \
    --config configs/FAST_17.toml \
    --attempts 1 \
    --verbose 2>&1 | while read line; do
    echo "[$(date +%H:%M:%S)] $line"
done

echo ""
echo "Debug run complete (or timed out after 60s)"