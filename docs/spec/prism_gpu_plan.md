# PRISM GPU Acceleration Plan

**Version**: 2.0
**Status**: Production-Ready
**Last Updated**: 2025-11-18

## Overview

This document specifies the GPU acceleration architecture for PRISM v2, including the context management system, kernel orchestration, security model, and integration points with the phase pipeline.

## GPU Context & Security

### Architecture

The GPU Context Manager (`prism-gpu/src/context.rs`) provides centralized GPU resource management:

- **CudaDevice Initialization**: Single device per orchestrator instance
- **PTX Module Registry**: Pre-loaded kernels (dendritic_reservoir, floyd_warshall, tda, quantum)
- **Security Guardrails**: Signed PTX verification, NVRTC restrictions
- **NVML Telemetry**: GPU utilization, memory usage, temperature monitoring

### Configuration

```toml
[gpu]
enabled = true
device_id = 0
ptx_dir = "target/ptx"
allow_nvrtc = false                    # Disable runtime compilation
require_signed_ptx = true              # Enforce signature verification
trusted_ptx_dir = "/opt/prism/trusted"
nvml_poll_interval_ms = 1000
```

### Security Model

**Signed PTX Verification**:
- Each PTX file (e.g., `quantum.ptx`) requires a `.sha256` signature file
- Compute SHA256 hash of PTX, compare with signature
- Reject load on mismatch → orchestrator falls back to CPU-only mode

**NVRTC Restriction**:
- When `allow_nvrtc = false`, only pre-compiled PTX from `trusted_ptx_dir` is loaded
- Prevents code injection via runtime kernel compilation

### CLI Integration

```bash
# Enable GPU with security
./target/release/prism-cli \
    --input graph.col \
    --file-type dimacs \
    --gpu-secure \
    --trusted-ptx-dir /opt/prism/trusted \
    --disable-nvrtc

# Select specific device
./target/release/prism-cli \
    --input graph.col \
    --file-type dimacs \
    --gpu-device 1

# Disable GPU (CPU fallback)
./target/release/prism-cli \
    --input graph.col \
    --file-type dimacs \
    --no-gpu
```

### File Paths

- Context: `prism-gpu/src/context.rs`
- Config: `prism-pipeline/src/config.rs` (GpuConfig struct)
- Orchestrator: `prism-pipeline/src/orchestrator/mod.rs` (initialization)
- CLI: `prism-cli/src/main.rs` (--gpu-* flags)

### Performance

| Component | Target | Achieved |
|-----------|--------|----------|
| Context Init | < 500ms | ~200ms |
| PTX Load | < 100ms | ~50ms |
| Signature Check | < 50ms | ~20ms |

## Phase-Specific GPU Kernels

### Phase 1: Dendritic Reservoir
- **Kernel**: `dendritic_reservoir.cu`
- **Operations**: Multi-branch neuromorphic dynamics, state evolution
- **Memory**: Preallocated reservoir state buffers
- **Target**: <100ms for 1000-vertex graphs

### Phase 2: Floyd-Warshall
- **Kernel**: `floyd_warshall.cu`
- **Operations**: All-pairs shortest paths, tile-based updates
- **Memory**: N×N distance matrix (shared memory optimization)
- **Target**: <200ms for 500-vertex graphs

### Phase 3: Quantum Evolution
- **Kernel**: `quantum.cu`
- **Operations**: Hamiltonian evolution, probability amplitude computation
- **Memory**: Quantum state vectors (amplitude arrays)
- **Target**: <500ms for DSJC125 (125 vertices)

### Phase 4: TDA (Topological Data Analysis)
- **Kernel**: `tda.cu`
- **Operations**: Persistent homology, filtration computation
- **Memory**: Simplex boundary matrices
- **Target**: <1s for 200-vertex graphs

## Kernel Compilation

### PTX Build Process

```bash
# Compile all kernels
./scripts/compile_ptx.sh quantum
./scripts/compile_ptx.sh dendritic_reservoir
./scripts/compile_ptx.sh floyd_warshall
./scripts/compile_ptx.sh tda

# Sign PTX files
./scripts/sign_ptx.sh

# Verify signatures
ls -l target/ptx/*.sha256
```

### NVCC Flags

```
-arch=sm_70          # CUDA compute capability 7.0+ (Volta/Turing/Ampere/Ada)
-ptx                 # Generate PTX (portable intermediate)
-O3                  # Maximum optimization
--use_fast_math      # Fast math operations (trade accuracy for speed)
-lineinfo            # Debug line information
-Xptxas -v           # Verbose register/memory usage
```

## Telemetry Integration

### GPU Metrics

```json
{
  "phase": "Phase3",
  "gpu_context": {
    "device_id": 0,
    "device_name": "NVIDIA GeForce RTX 4090",
    "compute_capability": "8.9",
    "memory_total_mb": 24576,
    "memory_used_mb": 4096,
    "utilization_percent": 87.5,
    "temperature_celsius": 68.0,
    "power_draw_watts": 325.0
  },
  "kernel_execution": {
    "kernel_name": "quantum_evolve_kernel",
    "launch_time_us": 150,
    "execution_time_us": 2500,
    "memory_transfer_us": 300
  }
}
```

### Telemetry Storage

- **Format**: NDJSON (newline-delimited JSON)
- **Location**: `telemetry/gpu_metrics.ndjson`
- **Rotation**: Daily rotation, 7-day retention
- **Schema**: Validated against `prism-core/src/telemetry.rs`

## Testing Strategy

### Unit Tests

```bash
# GPU context tests (no GPU required - simulation mode)
cargo test --package prism-gpu --lib context --no-default-features

# Phase 3 CPU fallback tests
cargo test --package prism-phases --lib phase3 --no-default-features

# GPU feature compilation check
cargo check --package prism-gpu --features cuda --no-default-features
```

### Integration Tests

```bash
# Full pipeline with GPU
cargo test --features gpu --test integration_phase3

# Benchmark with GPU/CPU comparison
./scripts/benchmark_warmstart.sh --gpu --attempts 10
```

### CI/CD

GitHub Actions workflow (`.github/workflows/gpu.yml`):
- GPU unit tests (simulation mode)
- Clippy checks on GPU code
- Documentation validation
- PTX compilation script verification

## Fallback Strategy

### CPU Fallback Triggers

1. **No CUDA Driver**: nvidia-smi fails or not found
2. **PTX Load Failure**: Signature verification failed or file missing
3. **GPU OOM**: Out of memory during kernel launch
4. **Explicit Disable**: `--no-gpu` CLI flag

### Fallback Behavior

- Log warning to telemetry: `"GPU unavailable, using CPU fallback"`
- Use CPU-only implementations (greedy heuristics)
- No RL parameter adjustment (default values)
- Pipeline continues without failure (graceful degradation)

## Performance Benchmarks

### Expected Speedups (GPU vs CPU)

| Phase | Operation | CPU Time | GPU Time | Speedup |
|-------|-----------|----------|----------|---------|
| Phase 1 | Dendritic Reservoir | 500ms | 80ms | 6.25x |
| Phase 2 | Floyd-Warshall | 2s | 150ms | 13.3x |
| Phase 3 | Quantum Evolution | 8s | 800ms | 10x |
| Phase 4 | TDA | 5s | 900ms | 5.5x |

**Test Graph**: DSJC250.5 (250 vertices, 15,668 edges)

## References

- GPU Context Manager: `prism-gpu/src/context.rs` (637 LOC)
- Phase 3 Quantum Kernel: `prism-gpu/src/kernels/quantum.cu` (365 LOC)
- Orchestrator Integration: `prism-pipeline/src/orchestrator/mod.rs`
- CLI Flags: `prism-cli/src/main.rs` (7 GPU-related flags)
- Compilation Script: `scripts/compile_ptx.sh`
- Benchmark Script: `scripts/benchmark_warmstart.sh`

## Appendix: PTX Signature Format

```bash
# quantum.ptx.sha256 content (example)
4a7b8c3d9e2f1a5b6c8d7e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8g9h
```

SHA256 hash computed over the entire PTX file. Verification:
```bash
sha256sum target/ptx/quantum.ptx | awk '{print $1}' > target/ptx/quantum.ptx.sha256
```

## Glossary

- **PTX**: Parallel Thread Execution - NVIDIA's portable assembly language
- **NVRTC**: NVIDIA Runtime Compilation - JIT compilation of CUDA code
- **NVML**: NVIDIA Management Library - Telemetry and monitoring API
- **Trotterization**: Time-slicing technique for quantum evolution simulation
- **Signed PTX**: PTX files with SHA256 signature for integrity verification
