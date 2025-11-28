#!/bin/bash
# Rebuild script for PRISM with conflict repair and adjusted chemical potential
# WSL2 compatible - single line commands

echo "============================================"
echo "PRISM REBUILD WITH CONFLICT REPAIR"
echo "Chemical potential: μ=0.6 (moderate)"
echo "============================================"

# Set environment
export PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6

echo ""
echo "[1/4] Compiling PTX kernels with μ=0.6..."

# Compile quantum kernel with new chemical potential
echo "  - Compiling quantum.cu (μ=0.6)..."
$CUDA_HOME/bin/nvcc -ptx --gpu-architecture=sm_86 -o target/ptx/quantum.ptx prism-gpu/src/kernels/quantum.cu

# Compile thermodynamic kernel with new chemical potential
echo "  - Compiling thermodynamic.cu (μ=0.6)..."
$CUDA_HOME/bin/nvcc -ptx --gpu-architecture=sm_86 -o target/ptx/thermodynamic.ptx prism-gpu/src/kernels/thermodynamic.cu

echo ""
echo "[2/4] Building Rust project with conflict repair..."
cargo build --release --features cuda

if [ $? -ne 0 ]; then
    echo "❌ Build failed! Check error messages above."
    exit 1
fi

echo ""
echo "[3/4] Verifying build..."
if [ -f "target/release/prism-cli" ]; then
    echo "✓ prism-cli built successfully"
else
    echo "❌ prism-cli not found!"
    exit 1
fi

echo ""
echo "[4/4] Testing with DSJC125.5..."
echo "Running with conflict repair enabled..."
timeout 60 ./target/release/prism-cli --input benchmarks/dimacs/DSJC125.5.col --config configs/TUNED_17.toml --attempts 1

echo ""
echo "============================================"
echo "BUILD COMPLETE WITH CONFLICT REPAIR"
echo "============================================"
echo ""
echo "Key changes implemented:"
echo "  ✓ Conflict repair mechanism for Phase 2/3"
echo "  ✓ Chemical potential adjusted to μ=0.6"
echo "  ✓ Memetic evolution on conflicted solutions"
echo ""
echo "To run full tests:"
echo "  ./test_all_graphs.sh DSJC125.5"
echo ""
echo "To monitor telemetry:"
echo "  tail -f telemetry.jsonl | jq '.'"