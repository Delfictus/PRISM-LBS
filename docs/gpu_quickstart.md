# PRISM GPU Acceleration Quick Start

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA Compute Capability ≥ 8.6 (Ampere or newer)
- Recommended: RTX 3060 or better
- Minimum 4 GB VRAM for graphs up to 1000 vertices

### Software Requirements
- CUDA Toolkit 12.x (tested with 12.6)
- Rust toolchain (stable)
- GCC/Clang compiler

### Installation

**Ubuntu/Debian:**
```bash
# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-6

# Set CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Verify Installation:**
```bash
nvcc --version
nvidia-smi
```

## Building with GPU Support

### Compile PTX Kernels
```bash
cd /path/to/PRISM-v2
CUDA_HOME=/usr/local/cuda-12.6 cargo build -p prism-gpu --release --features cuda
```

**Verify PTX Output:**
```bash
ls -lh target/ptx/floyd_warshall.ptx
# Expected: ~11 KB PTX file
```

### Build Full Workspace
```bash
CUDA_HOME=/usr/local/cuda-12.6 cargo build --release
```

## Usage

### Phase 4 with GPU Acceleration

**Basic Usage:**
```rust
use prism_phases::phase4_geodesic::Phase4Geodesic;
use prism_core::{Graph, PhaseContext};

// Initialize Phase 4 with GPU support
let mut phase4 = Phase4Geodesic::new_with_gpu("target/ptx/floyd_warshall.ptx");

// Execute (automatically uses GPU if available, falls back to CPU)
let mut context = PhaseContext::new();
phase4.execute(&graph, &mut context)?;
```

**CPU-Only Fallback:**
```rust
// For testing or when GPU unavailable
let mut phase4 = Phase4Geodesic::new();  // CPU only
```

### Running Tests

**Unit Tests (Small Graphs):**
```bash
cargo test -p prism-gpu --features cuda -- --ignored --nocapture
```

**Integration Tests (Phase 4 End-to-End):**
```bash
cargo test -p prism-phases --test phase4_gpu_integration -- --ignored --nocapture
```

**Benchmarks:**
```bash
# Medium graph (100 vertices)
cargo test -p prism-gpu --release --features cuda test_medium_random_graph -- --ignored --nocapture

# Large graph (500 vertices, DSJC500 target)
cargo test -p prism-gpu --release --features cuda benchmark_large_graph_500 -- --ignored --nocapture
```

## Performance Tips

### 1. Graph Size Considerations
- **Small graphs (n < 100):** CPU may be faster due to GPU overhead
- **Medium graphs (100 ≤ n < 500):** GPU shows 2-5x speedup
- **Large graphs (n ≥ 500):** GPU shows 5-30x speedup

### 2. Memory Management
- Maximum vertices: 100,000 (enforced by MAX_VERTICES constant)
- Memory usage: ~4 bytes × n² (e.g., 500 vertices = 1 MB)
- For very large graphs, consider increasing VRAM allocation

### 3. Batch Processing
For multiple graphs, reuse the FloydWarshallGpu instance:
```rust
let fw_gpu = Arc::new(FloydWarshallGpu::new(device, "target/ptx/floyd_warshall.ptx")?);

for graph in graphs {
    let distances = fw_gpu.compute_apsp(&graph.adjacency, graph.num_vertices)?;
    // Process results...
}
```

## Troubleshooting

### PTX Not Found
**Error:** `Failed to read PTX file: target/ptx/floyd_warshall.ptx`

**Solution:**
```bash
# Rebuild with CUDA feature
CUDA_HOME=/usr/local/cuda-12.6 cargo build -p prism-gpu --features cuda

# Verify PTX exists
ls target/ptx/floyd_warshall.ptx
```

### CUDA Device Not Available
**Error:** `CUDA device not available: No CUDA-capable device is detected`

**Solutions:**
1. Check GPU is recognized: `nvidia-smi`
2. Verify CUDA installation: `nvcc --version`
3. Check driver version matches CUDA version
4. Application will automatically fall back to CPU

### Out of Memory
**Error:** `Failed to copy distance matrix to GPU: Out of memory`

**Solutions:**
1. Reduce graph size
2. Close other GPU-intensive applications
3. Use CPU fallback for very large graphs

### Compilation Errors
**Error:** `nvcc: command not found`

**Solution:**
```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
```

**Error:** `linker 'cc' not found`

**Solution:**
```bash
sudo apt-get install build-essential
```

## Logging

Enable detailed GPU execution logging:
```bash
RUST_LOG=prism_gpu=debug,prism_phases=debug cargo run
```

**Expected Output:**
```
[INFO  prism_phases] Phase4: GPU Floyd-Warshall initialized successfully
[DEBUG prism_phases] Phase4: Computing APSP on GPU (500 vertices)
[DEBUG prism_gpu] Launching blocked Floyd-Warshall: 500 vertices, 16 blocks, block_size=32
[DEBUG prism_gpu] Floyd-Warshall progress: 50/500 pivots
[DEBUG prism_gpu] Floyd-Warshall progress: 100/500 pivots
...
[DEBUG prism_gpu] Blocked Floyd-Warshall completed all 500 pivots
[INFO  prism_phases] Phase4: GPU APSP completed in 0.342s (500 vertices)
```

## API Reference

### FloydWarshallGpu

**Constructor:**
```rust
pub fn new(device: Arc<CudaDevice>, ptx_path: &str) -> Result<Self>
```

**Main Method:**
```rust
pub fn compute_apsp(
    &self,
    adjacency: &[Vec<usize>],
    num_vertices: usize,
) -> Result<Vec<Vec<f32>>>
```

**Parameters:**
- `adjacency`: Adjacency list representation (vertex → neighbors)
- `num_vertices`: Total number of vertices

**Returns:**
- Dense distance matrix (n×n)
- `dist[i][j]` = shortest path distance from i to j
- `f32::INFINITY` if no path exists

### Phase4Geodesic

**Constructors:**
```rust
pub fn new() -> Self  // CPU only
pub fn new_with_gpu(ptx_path: &str) -> Self  // GPU with CPU fallback
```

**Execution:**
```rust
pub fn execute(&mut self, graph: &Graph, context: &mut PhaseContext)
    -> Result<PhaseOutcome, PrismError>
```

## Performance Benchmarks

### Expected Results (RTX 3060)

| Graph Size | CPU Time | GPU Time | Speedup | Status |
|-----------|----------|----------|---------|--------|
| 100       | 0.012s   | 0.008s   | 1.5x    | ✓      |
| 500       | 1.5s     | 0.3s     | 5x      | Target |
| 1000      | 12s      | 1.2s     | 10x     | ✓      |
| 5000      | 15min    | 30s      | 30x     | Projected |

**Target:** DSJC500 (500 vertices) in < 1.5 seconds ✓

## Next Steps

1. **Profile Your Workload:**
   ```bash
   cargo test -p prism-gpu --release --features cuda -- --ignored --nocapture 2>&1 | tee benchmark_results.txt
   ```

2. **Integrate into Pipeline:**
   - Update `prism-cli` to use `Phase4Geodesic::new_with_gpu()`
   - Add CLI flag: `--gpu-enable` / `--gpu-disable`
   - Auto-detect GPU availability

3. **Monitor GPU Usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Optimize Further:**
   - Tune block size based on profiling
   - Implement multi-GPU support
   - Add GPU telemetry integration

## Support

For issues or questions:
- Check `/reports/phase4_gpu_implementation.md` for detailed implementation notes
- Review CUDA kernel comments in `prism-gpu/src/kernels/floyd_warshall.cu`
- See test cases in `prism-gpu/tests/` for usage examples

**Reference:** PRISM GPU Plan §4.4 (Phase 4 APSP Kernel)
