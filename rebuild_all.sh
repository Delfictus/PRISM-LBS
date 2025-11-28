#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════
# COMPLETE PRISM REBUILD SCRIPT - PTX KERNELS + RUST
# ═══════════════════════════════════════════════════════════════════════════
# This script rebuilds everything from scratch including all GPU kernels
# ═══════════════════════════════════════════════════════════════════════════

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Environment setup for WSL2
export PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6

echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}${BOLD}         PRISM COMPLETE REBUILD (PTX + RUST)${NC}"
echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Step 1: Clean old builds
echo -e "${YELLOW}[1/4] Cleaning old builds...${NC}"
rm -rf target/release target/debug
cargo clean

echo ""
echo -e "${YELLOW}[2/4] Creating PTX directory...${NC}"
mkdir -p target/ptx

echo ""
echo -e "${YELLOW}[3/4] Compiling PTX kernels...${NC}"

# List of all GPU kernels to compile
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

# Compile each kernel
for kernel in "${KERNELS[@]}"; do
    echo -ne "  Compiling ${CYAN}$kernel.cu${NC}..."

    if [ -f "prism-gpu/src/kernels/$kernel.cu" ]; then
        $CUDA_HOME/bin/nvcc -ptx \
            --gpu-architecture=sm_86 \
            -o target/ptx/$kernel.ptx \
            prism-gpu/src/kernels/$kernel.cu 2>/dev/null

        if [ $? -eq 0 ]; then
            echo -e " ${GREEN}✓${NC}"
        else
            echo -e " ${RED}✗${NC}"
            echo -e "  ${RED}Error compiling $kernel.cu${NC}"
        fi
    else
        echo -e " ${YELLOW}SKIPPED (file not found)${NC}"
    fi
done

echo ""
echo -e "${YELLOW}[4/4] Building Rust project with CUDA features...${NC}"
cargo build --release --features cuda

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}${BOLD}        ✓ BUILD SUCCESSFUL - READY FOR 17 COLORS!${NC}"
    echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${CYAN}Chemical Potential:${NC} ${RED}μ=0.7${NC} (extreme compression)"
    echo -e "${CYAN}Binary location:${NC} ${YELLOW}./target/release/prism-cli${NC}"
    echo ""
    echo -e "${GREEN}Quick test commands:${NC}"
    echo -e "  ${YELLOW}./test_single_graph.sh benchmarks/dimacs/DSJC125.5.col 1${NC}"
    echo -e "  ${YELLOW}./run_17.sh 10${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}${BOLD}✗ BUILD FAILED${NC}"
    echo -e "${RED}Check error messages above${NC}"
    exit 1
fi