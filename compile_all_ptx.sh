#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# COMPILE ALL PTX KERNELS FOR PRISM
# ═══════════════════════════════════════════════════════════════════════════
# Compiles all CUDA kernels to PTX format for runtime loading
# Target: sm_86 (RTX 3060)
# ═══════════════════════════════════════════════════════════════════════════

set -e

# Setup CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

# Ensure target directory exists
mkdir -p target/ptx

echo "═══════════════════════════════════════════════════════════════════════════"
echo "  COMPILING ALL PTX KERNELS"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "CUDA Version: $(nvcc --version | grep release)"
echo "Target Architecture: sm_86 (RTX 3060)"
echo ""

# List of all kernel files
KERNELS=(
    "active_inference"
    "cma_es"
    "dendritic_reservoir"
    "ensemble_exchange"
    "floyd_warshall"
    "gnn_inference"
    "molecular_dynamics"
    "pimc"
    "quantum"
    "tda"
    "thermodynamic"
    "transfer_entropy"
)

SUCCESS_COUNT=0
FAIL_COUNT=0

for kernel in "${KERNELS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Compiling: $kernel.cu"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if nvcc -ptx \
        --gpu-architecture=sm_86 \
        -o "target/ptx/$kernel.ptx" \
        "prism-gpu/src/kernels/$kernel.cu" 2>&1; then

        SIZE=$(ls -lh "target/ptx/$kernel.ptx" | awk '{print $5}')
        echo "✅ SUCCESS: $kernel.ptx ($SIZE)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ FAILED: $kernel.cu"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
done

echo "═══════════════════════════════════════════════════════════════════════════"
echo "  COMPILATION COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "✅ Successful: $SUCCESS_COUNT"
echo "❌ Failed: $FAIL_COUNT"
echo ""
echo "PTX files:"
ls -lh target/ptx/*.ptx 2>/dev/null || echo "  (none)"
echo ""
echo "Total PTX size: $(du -sh target/ptx 2>/dev/null | awk '{print $1}')"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
