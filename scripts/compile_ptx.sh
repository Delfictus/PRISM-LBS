#!/bin/bash
#
# PTX Compilation Script for PRISM GPU Kernels
#
# Usage:
#   ./scripts/compile_ptx.sh [all|quantum|dendritic|floyd|tda]
#   ./scripts/compile_ptx.sh quantum     # Compile only quantum.cu
#   ./scripts/compile_ptx.sh all         # Compile all kernels (default)
#
# Requirements:
#   - CUDA Toolkit 11.0+ (nvcc must be in PATH)
#   - Compute capability 8.6+ GPU (RTX 3060+)
#
# Output:
#   - Compiled PTX files in target/ptx/
#   - SHA256 signatures in target/ptx/*.ptx.sha256

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CUDA_ARCH="sm_86"  # RTX 3060, RTX 3070, RTX 3080, RTX 3090, A100
KERNEL_DIR="prism-gpu/src/kernels"
OUTPUT_DIR="target/ptx"
NVCC_FLAGS="-O3 --use_fast_math -lineinfo"

# Ensure we're in project root
if [ ! -d "prism-gpu" ]; then
    echo -e "${RED}Error: Must run from PRISM-v2 project root${NC}"
    exit 1
fi

# Check for nvcc
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: nvcc not found. Please install CUDA Toolkit.${NC}"
    echo "Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
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
        return
    fi

    echo -e "${GREEN}Compiling ${kernel_name}.cu...${NC}"

    # Compile CUDA to PTX
    nvcc -ptx "$cu_file" -o "$ptx_file" \
        --gpu-architecture="$CUDA_ARCH" \
        $NVCC_FLAGS

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Compiled ${ptx_file}${NC}"

        # Generate SHA256 signature
        if command -v sha256sum &> /dev/null; then
            sha256sum "$ptx_file" | awk '{print $1}' > "$sig_file"
            echo -e "${GREEN}✓ Generated signature ${sig_file}${NC}"
        elif command -v shasum &> /dev/null; then
            shasum -a 256 "$ptx_file" | awk '{print $1}' > "$sig_file"
            echo -e "${GREEN}✓ Generated signature ${sig_file}${NC}"
        else
            echo -e "${YELLOW}Warning: sha256sum/shasum not found, signature not generated${NC}"
        fi

        # Show file info
        local size=$(du -h "$ptx_file" | cut -f1)
        echo -e "  Size: $size"
        echo ""
    else
        echo -e "${RED}✗ Failed to compile ${kernel_name}.cu${NC}"
        exit 1
    fi
}

# Function to verify PTX file
verify_ptx() {
    local ptx_file=$1
    local sig_file="${ptx_file}.sha256"

    if [ ! -f "$ptx_file" ]; then
        echo -e "${YELLOW}Warning: ${ptx_file} not found${NC}"
        return
    fi

    if [ ! -f "$sig_file" ]; then
        echo -e "${YELLOW}Warning: ${sig_file} not found${NC}"
        return
    fi

    local expected=$(cat "$sig_file")
    local actual=""

    if command -v sha256sum &> /dev/null; then
        actual=$(sha256sum "$ptx_file" | awk '{print $1}')
    elif command -v shasum &> /dev/null; then
        actual=$(shasum -a 256 "$ptx_file" | awk '{print $1}')
    else
        echo -e "${YELLOW}Warning: Cannot verify signature (sha256sum not available)${NC}"
        return
    fi

    if [ "$expected" = "$actual" ]; then
        echo -e "${GREEN}✓ Signature verified: $(basename $ptx_file)${NC}"
    else
        echo -e "${RED}✗ Signature mismatch: $(basename $ptx_file)${NC}"
        echo "  Expected: $expected"
        echo "  Got:      $actual"
        exit 1
    fi
}

# Print header
echo ""
echo "========================================="
echo "  PRISM GPU Kernel PTX Compilation"
echo "========================================="
echo "  CUDA Architecture: $CUDA_ARCH"
echo "  Compiler: $(nvcc --version | grep release | awk '{print $5, $6}')"
echo "  Output Directory: $OUTPUT_DIR"
echo "========================================="
echo ""

# Determine which kernels to compile
TARGET="${1:-all}"

case "$TARGET" in
    all)
        echo "Compiling all kernels..."
        compile_kernel "quantum"
        compile_kernel "dendritic_reservoir"
        compile_kernel "floyd_warshall"
        compile_kernel "tda"
        ;;
    quantum)
        compile_kernel "quantum"
        ;;
    dendritic)
        compile_kernel "dendritic_reservoir"
        ;;
    floyd)
        compile_kernel "floyd_warshall"
        ;;
    tda)
        compile_kernel "tda"
        ;;
    verify)
        echo "Verifying PTX signatures..."
        verify_ptx "${OUTPUT_DIR}/quantum.ptx"
        verify_ptx "${OUTPUT_DIR}/dendritic_reservoir.ptx"
        verify_ptx "${OUTPUT_DIR}/floyd_warshall.ptx"
        verify_ptx "${OUTPUT_DIR}/tda.ptx"
        echo ""
        echo -e "${GREEN}All signatures verified successfully!${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Error: Unknown target '${TARGET}'${NC}"
        echo "Usage: $0 [all|quantum|dendritic|floyd|tda|verify]"
        exit 1
        ;;
esac

# Summary
echo "========================================="
echo "  Compilation Complete!"
echo "========================================="
echo ""
echo "PTX files:"
ls -lh "$OUTPUT_DIR"/*.ptx 2>/dev/null || echo "  (none found)"
echo ""
echo "Signature files:"
ls -lh "$OUTPUT_DIR"/*.ptx.sha256 2>/dev/null || echo "  (none found)"
echo ""
echo "Next steps:"
echo "  1. Verify signatures: ./scripts/compile_ptx.sh verify"
echo "  2. Build Rust crates: cargo build --workspace --features cuda --release"
echo "  3. Run tests: cargo test -p prism-gpu --features cuda -- --ignored"
echo ""
