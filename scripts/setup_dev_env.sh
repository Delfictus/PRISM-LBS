#!/usr/bin/env bash
set -euo pipefail

echo "=== PRISM Development Environment Setup ==="

# Check NVIDIA driver
echo "Checking GPU prerequisites..."

if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA driver detected"
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
else
    echo "⚠ NVIDIA driver not found. GPU acceleration will be unavailable."
    echo "  Install from: https://www.nvidia.com/Download/index.aspx"
fi

# Check CUDA Toolkit
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
    echo "✓ CUDA Toolkit found: $CUDA_HOME"

    if command -v nvcc &> /dev/null; then
        nvcc --version | grep "release"
    fi
else
    echo "⚠ CUDA Toolkit not found."
    echo "  Install from: https://developer.nvidia.com/cuda-downloads"
    echo "  Recommended: CUDA 12.x for best compatibility"
fi

# Check Rust toolchain
if command -v cargo &> /dev/null; then
    echo "✓ Rust toolchain found: $(rustc --version)"
else
    echo "✗ Rust not found. Install from: https://rustup.rs"
    exit 1
fi

# Create PTX output directory
mkdir -p target/ptx
echo "✓ Created target/ptx directory"

# PTX Signing Setup
echo ""
echo "Setting up PTX signing..."
mkdir -p target/ptx
if [ ! -f scripts/sign_ptx.sh ]; then
    cat > scripts/sign_ptx.sh << 'SIGNSCRIPT'
#!/usr/bin/env bash
# Sign PTX files with SHA256 hash
set -euo pipefail

if [ ! -d "target/ptx" ]; then
    echo "Error: target/ptx directory not found"
    exit 1
fi

for ptx in target/ptx/*.ptx; do
    if [ -f "$ptx" ]; then
        sha256sum "$ptx" | awk '{print $1}' > "${ptx}.sha256"
        echo "Signed: $(basename $ptx)"
    fi
done

echo "✓ All PTX files signed"
SIGNSCRIPT
    chmod +x scripts/sign_ptx.sh
    echo "✓ Created scripts/sign_ptx.sh"
fi

# Build instructions
echo ""
echo "=== Build Instructions ==="
echo "1. Build with GPU support:"
echo "   cargo build --release --features gpu"
echo ""
echo "2. Compile PTX kernels:"
echo "   ./scripts/compile_ptx.sh quantum"
echo "   ./scripts/compile_ptx.sh dendritic_reservoir"
echo "   ./scripts/compile_ptx.sh floyd_warshall"
echo "   ./scripts/compile_ptx.sh tda"
echo ""
echo "3. Sign PTX files (for secure mode):"
echo "   ./scripts/sign_ptx.sh"
echo ""
echo "4. Run with GPU:"
echo "   ./target/release/prism-cli --input graph.col --file-type dimacs --verbose"
echo ""
echo "✓ Development environment ready"
