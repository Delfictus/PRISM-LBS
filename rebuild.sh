#!/bin/bash

# PRISM Complete Rebuild Script for WSL2

echo "======================================"
echo "    PRISM COMPLETE REBUILD"
echo "======================================"

# Set environment
export PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6

echo ""
echo "Step 1: Cleaning old builds..."
cargo clean
rm -rf target/ptx

echo ""
echo "Step 2: Compiling PTX kernels..."
./compile_ptx.sh

echo ""
echo "Step 3: Building Rust project..."
cargo build --release --features cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "    BUILD SUCCESSFUL!"
    echo "======================================"
    echo ""
    echo "Binary: ./target/release/prism-cli"
    echo "Config: configs/EXTREME_MAX.toml (μ=0.9)"
    echo ""
    echo "Quick test:"
    echo "  ./run_extreme.sh 1"
else
    echo ""
    echo "BUILD FAILED - Check errors above"
    exit 1
fi