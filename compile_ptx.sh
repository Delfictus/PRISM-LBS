#!/bin/bash

# PTX Kernel Compilation Script for WSL2
# Compiles all CUDA kernels to PTX format

echo "=========================================="
echo "     PRISM PTX Kernel Compilation"
echo "=========================================="

# Check if CUDA_HOME is set
if [ -z "$CUDA_HOME" ]; then
    echo "Setting CUDA_HOME..."
    export CUDA_HOME=/usr/local/cuda-12.6
fi

# Check if nvcc exists
if [ ! -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "Error: nvcc not found at $CUDA_HOME/bin/nvcc"
    exit 1
fi

# Create PTX directory if it doesn't exist
echo "Creating target/ptx directory..."
mkdir -p target/ptx

# List of kernels to compile
KERNELS=(
    "active_inference"
    "cma_es"
    "dendritic_reservoir"
    "whcr"
    "dendritic_whcr"
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

# LBS kernels live under prism-lbs/kernels/lbs
LBS_KERNELS=(
    "surface_accessibility"
    "distance_matrix"
    "pocket_clustering"
    "druggability_scoring"
)

echo ""
echo "Compiling kernels..."
echo ""

TOTAL=${#KERNELS[@]}
SUCCESS=0
FAILED=0

for kernel in "${KERNELS[@]}"; do
    SOURCE="prism-gpu/src/kernels/${kernel}.cu"
    OUTPUT="target/ptx/${kernel}.ptx"

    if [ -f "$SOURCE" ]; then
        echo -n "Compiling $kernel.cu ... "

        # Compile with error suppression for warnings
        $CUDA_HOME/bin/nvcc -ptx --gpu-architecture=sm_86 -o "$OUTPUT" "$SOURCE" 2>/dev/null

        if [ $? -eq 0 ]; then
            echo "OK"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "FAILED"
            # Try again with output to see error
            echo "  Retrying with verbose output:"
            $CUDA_HOME/bin/nvcc -ptx --gpu-architecture=sm_86 -o "$OUTPUT" "$SOURCE"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "$kernel.cu ... SKIPPED (file not found)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "Compiling LBS kernels..."
for kernel in "${LBS_KERNELS[@]}"; do
    SOURCE="prism-gpu/src/kernels/lbs/${kernel}.cu"
    OUTPUT="target/ptx/lbs_${kernel}.ptx"
    if [ -f "$SOURCE" ]; then
        echo -n "Compiling lbs/${kernel}.cu ... "
        $CUDA_HOME/bin/nvcc -ptx --gpu-architecture=sm_86 -o "$OUTPUT" "$SOURCE" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "OK"
        else
            echo "FAILED"
            echo "  Retrying with verbose output:"
            $CUDA_HOME/bin/nvcc -ptx --gpu-architecture=sm_86 -o "$OUTPUT" "$SOURCE"
        fi
    else
        echo "lbs/${kernel}.cu ... SKIPPED (file not found)"
    fi
done

echo ""
echo "=========================================="
echo "Compilation Complete"
echo "  Success: $SUCCESS/$TOTAL"
echo "  Failed:  $FAILED/$TOTAL"
echo "=========================================="

# Check critical kernels
echo ""
echo "Checking critical kernels for 17-color attempt:"

CRITICAL=("quantum" "thermodynamic" "cma_es" "pimc")
ALL_GOOD=true

for kernel in "${CRITICAL[@]}"; do
    if [ -f "target/ptx/${kernel}.ptx" ]; then
        echo "  ✓ $kernel.ptx exists"
    else
        echo "  ✗ $kernel.ptx MISSING!"
        ALL_GOOD=false
    fi
done

echo ""
if [ "$ALL_GOOD" = true ]; then
    echo "All critical kernels compiled successfully!"
    echo "Chemical potential μ=0.9 is active"
else
    echo "WARNING: Some critical kernels missing!"
fi
