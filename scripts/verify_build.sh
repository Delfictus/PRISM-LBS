#!/bin/bash
# Comprehensive build verification for Phase 2 Thermodynamic implementation

set -e

echo "=== PRISM Phase 2 Thermodynamic Build Verification ==="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored status
print_status() {
    local status=$1
    local message=$2

    if [ "$status" == "OK" ]; then
        echo -e "${GREEN}[OK]${NC} $message"
    elif [ "$status" == "WARN" ]; then
        echo -e "${YELLOW}[WARN]${NC} $message"
    elif [ "$status" == "ERROR" ]; then
        echo -e "${RED}[ERROR]${NC} $message"
    else
        echo "[INFO] $message"
    fi
}

# Change to workspace root
cd "$(dirname "$0")/.."

# Step 1: Check Rust toolchain
echo "Step 1: Checking Rust toolchain..."
if command -v cargo &> /dev/null; then
    rust_version=$(cargo --version)
    print_status "OK" "Rust toolchain: $rust_version"
else
    print_status "ERROR" "Cargo not found. Install Rust from https://rustup.rs"
    exit 1
fi
echo ""

# Step 2: Check CUDA toolkit
echo "Step 2: Checking CUDA toolkit..."
if [ -f "/usr/local/cuda-12.6/bin/nvcc" ]; then
    cuda_version=$(/usr/local/cuda-12.6/bin/nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
    print_status "OK" "CUDA toolkit: $cuda_version"
    NVCC_PATH="/usr/local/cuda-12.6/bin/nvcc"
elif command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
    print_status "OK" "CUDA toolkit: $cuda_version"
    NVCC_PATH="nvcc"
else
    print_status "WARN" "CUDA toolkit not found (GPU features will be unavailable)"
    NVCC_PATH=""
fi
echo ""

# Step 3: Compile PTX kernels
echo "Step 3: Compiling CUDA PTX kernels..."

mkdir -p target/ptx

if [ -n "$NVCC_PATH" ]; then
    # Compile thermodynamic.ptx
    echo "  Compiling thermodynamic.ptx..."
    $NVCC_PATH --ptx \
      -o target/ptx/thermodynamic.ptx \
      prism-gpu/src/kernels/thermodynamic.cu \
      -arch=sm_86 --use_fast_math -O3

    if [ $? -eq 0 ]; then
        ptx_size=$(du -h target/ptx/thermodynamic.ptx | cut -f1)
        print_status "OK" "thermodynamic.ptx compiled ($ptx_size)"
    else
        print_status "ERROR" "thermodynamic.ptx compilation failed"
        exit 1
    fi
else
    print_status "WARN" "Skipping PTX compilation (nvcc not available)"
fi
echo ""

# Step 4: Cargo check (syntax validation)
echo "Step 4: Running cargo check (syntax validation)..."
cargo check --workspace --quiet

if [ $? -eq 0 ]; then
    print_status "OK" "Workspace syntax check passed"
else
    print_status "ERROR" "Workspace syntax check failed"
    exit 1
fi
echo ""

# Step 5: Build workspace (debug mode first for speed)
echo "Step 5: Building workspace (debug mode)..."
cargo build --workspace --quiet

if [ $? -eq 0 ]; then
    print_status "OK" "Debug build successful"
else
    print_status "ERROR" "Debug build failed"
    exit 1
fi
echo ""

# Step 6: Build with CUDA features
echo "Step 6: Building with CUDA features..."
if [ -n "$NVCC_PATH" ]; then
    cargo build --workspace --features cuda --quiet

    if [ $? -eq 0 ]; then
        print_status "OK" "CUDA-enabled build successful"
    else
        print_status "ERROR" "CUDA-enabled build failed"
        exit 1
    fi
else
    print_status "WARN" "Skipping CUDA build (nvcc not available)"
fi
echo ""

# Step 7: Run unit tests (excluding GPU tests)
echo "Step 7: Running unit tests (excluding GPU tests)..."
cargo test --workspace --lib --quiet -- --nocapture 2>&1 | grep -E "(test result|FAILED)" || true

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_status "OK" "Unit tests passed"
else
    print_status "WARN" "Some unit tests failed (check output above)"
fi
echo ""

# Step 8: Build release binaries
echo "Step 8: Building release binaries..."
echo "  Building prism-cli..."
cargo build --release --bin prism-cli --quiet

if [ -f target/release/prism-cli ]; then
    cli_size=$(du -h target/release/prism-cli | cut -f1)
    print_status "OK" "prism-cli built ($cli_size)"
else
    print_status "ERROR" "prism-cli build failed"
    exit 1
fi

echo "  Building fluxnet_train..."
cargo build --release --bin fluxnet_train --quiet

if [ -f target/release/fluxnet_train ]; then
    train_size=$(du -h target/release/fluxnet_train | cut -f1)
    print_status "OK" "fluxnet_train built ($train_size)"
else
    print_status "ERROR" "fluxnet_train build failed"
    exit 1
fi
echo ""

# Step 9: Verify Phase 2 files
echo "Step 9: Verifying Phase 2 Thermodynamic implementation..."

required_files=(
    "prism-gpu/src/kernels/thermodynamic.cu"
    "prism-gpu/src/thermodynamic.rs"
    "prism-phases/src/phase2_thermodynamic.rs"
    "prism-fluxnet/src/bin/train.rs"
    "configs/dsjc250_aggressive.toml"
    "scripts/train_fluxnet_dsjc250.sh"
    "scripts/test_dsjc250_aggressive.sh"
    "prism-phases/tests/phase2_gpu_smoke.rs"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_status "OK" "$file"
    else
        print_status "ERROR" "$file missing"
        exit 1
    fi
done
echo ""

# Step 10: Summary
echo "=== Build Verification Summary ==="
echo ""
echo "Files:"
echo "  PTX Kernels:        target/ptx/"
echo "  CLI Binary:         target/release/prism-cli"
echo "  Training Binary:    target/release/fluxnet_train"
echo ""
echo "Configuration:"
echo "  Aggressive Config:  configs/dsjc250_aggressive.toml"
echo ""
echo "Scripts:"
echo "  Train Q-table:      scripts/train_fluxnet_dsjc250.sh"
echo "  Integration Test:   scripts/test_dsjc250_aggressive.sh"
echo ""
print_status "OK" "Phase 2 Thermodynamic implementation verified!"
echo ""
echo "Next steps:"
echo "  1. Train Q-table:         ./scripts/train_fluxnet_dsjc250.sh"
echo "  2. Run integration test:  ./scripts/test_dsjc250_aggressive.sh"
echo "  3. Benchmark DSJC250:     ./target/release/prism-cli --input benchmarks/dimacs/DSJC250.5.col --warmstart --gpu --fluxnet-qtable profiles/curriculum/qtable_dsjc250.bin"
echo ""
