#!/bin/bash
#
# Complete PTX Compilation Script for ALL PRISM GPU Kernels
#
# This script compiles all 12 CUDA kernels to PTX format for GPU acceleration
#
# Usage:
#   ./scripts/compile_all_ptx.sh
#
# Requirements:
#   - CUDA Toolkit 12.0+ (nvcc must be in PATH)
#   - CUDA_HOME environment variable set to /usr/local/cuda-12.6
#
# Output:
#   - Compiled PTX files in target/ptx/
#   - SHA256 signatures in target/ptx/*.ptx.sha256

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CUDA_ARCH="sm_86"  # RTX 3060+, RTX 3070, RTX 3080, RTX 3090, A100
KERNEL_DIR="prism-gpu/src/kernels"
OUTPUT_DIR="target/ptx"
NVCC_FLAGS="-O3 --use_fast_math -lineinfo"

# All 12 kernels from the complete PRISM pipeline
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

# Ensure we're in project root
if [ ! -d "prism-gpu" ]; then
    echo -e "${RED}Error: Must run from PRISM project root${NC}"
    echo "Current directory: $(pwd)"
    echo "Please cd to /mnt/c/Users/Predator/Desktop/PRISM first"
    exit 1
fi

# Check for nvcc
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: nvcc not found. Please install CUDA Toolkit.${NC}"
    echo "Download from: https://developer.nvidia.com/cuda-downloads"
    echo ""
    echo "On WSL2, you may need to:"
    echo "  export PATH=/usr/local/cuda-12.6/bin:\$PATH"
    echo "  export CUDA_HOME=/usr/local/cuda-12.6"
    exit 1
fi

# Verify CUDA_HOME is set
if [ -z "$CUDA_HOME" ]; then
    echo -e "${YELLOW}Warning: CUDA_HOME not set. Attempting to auto-detect...${NC}"
    if [ -d "/usr/local/cuda-12.6" ]; then
        export CUDA_HOME=/usr/local/cuda-12.6
        export PATH=$CUDA_HOME/bin:$PATH
        echo -e "${GREEN}✓ Set CUDA_HOME=$CUDA_HOME${NC}"
    else
        echo -e "${RED}Error: Could not find CUDA installation${NC}"
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to compile a single kernel
compile_kernel() {
    local kernel_name=$1
    local cu_file="${KERNEL_DIR}/${kernel_name}.cu"
    local ptx_file="${OUTPUT_DIR}/${kernel_name}.ptx"
    local sig_file="${ptx_file}.sha256"

    if [ ! -f "$cu_file" ]; then
        echo -e "${YELLOW}Warning: ${cu_file} not found, skipping${NC}"
        return 1
    fi

    echo -e "${BLUE}[${kernel_name}]${NC} Compiling..."

    # Compile CUDA to PTX
    if nvcc -ptx "$cu_file" -o "$ptx_file" \
        --gpu-architecture="$CUDA_ARCH" \
        $NVCC_FLAGS 2>&1 | grep -v "warning" || true; then

        echo -e "${GREEN}✓ Compiled ${kernel_name}.ptx${NC}"

        # Generate SHA256 signature
        if command -v sha256sum &> /dev/null; then
            sha256sum "$ptx_file" | awk '{print $1}' > "$sig_file"
        elif command -v shasum &> /dev/null; then
            shasum -a 256 "$ptx_file" | awk '{print $1}' > "$sig_file"
        fi

        # Show file info
        local size=$(du -h "$ptx_file" | cut -f1)
        echo -e "  ${GREEN}→${NC} Size: $size"
        return 0
    else
        echo -e "${RED}✗ Failed to compile ${kernel_name}.cu${NC}"
        return 1
    fi
}

# Print header
echo ""
echo "========================================="
echo "  PRISM Complete PTX Compilation"
echo "========================================="
echo "  CUDA Architecture: $CUDA_ARCH"
echo "  CUDA Home: $CUDA_HOME"
echo "  Compiler: $(nvcc --version | grep release | awk '{print $5, $6}')"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Total Kernels: ${#KERNELS[@]}"
echo "========================================="
echo ""

# Compile all kernels
SUCCESS_COUNT=0
FAIL_COUNT=0

for kernel in "${KERNELS[@]}"; do
    if compile_kernel "$kernel"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
    echo ""
done

# Summary
echo "========================================="
echo "  Compilation Summary"
echo "========================================="
echo -e "${GREEN}  Successful: $SUCCESS_COUNT${NC}"
if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${RED}  Failed: $FAIL_COUNT${NC}"
fi
echo "========================================="
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "PTX files in $OUTPUT_DIR:"
    ls -lh "$OUTPUT_DIR"/*.ptx 2>/dev/null | awk '{printf "  %s  %s\n", $9, $5}'
    echo ""
    echo -e "${GREEN}✓ PTX compilation complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Test GPU acceleration:"
    echo "     ./target/release/prism-cli --input benchmarks/dimacs/DSJC125.1.col --gpu"
    echo ""
    echo "  2. Verify PTX files are loaded:"
    echo "     grep 'Loading PTX module' logs/prism.log"
    echo ""
else
    echo -e "${RED}✗ No kernels compiled successfully${NC}"
    exit 1
fi
