# PRISM v2: GPU-Accelerated Graph Coloring Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)

Production-ready graph coloring system with GPU acceleration, reinforcement learning, and enterprise monitoring.

## Features

- **100% GPU Acceleration**: All compute-intensive operations run on CUDA
- **Universal FluxNet RL**: Adaptive reinforcement learning across all 7 phases
- **Dendritic Reservoir**: Multi-branch neuromorphic computing
- **Warmstart System**: Cross-graph learning and curriculum transfer
- **Multi-GPU Support**: Scale across multiple devices
- **Enterprise Monitoring**: Prometheus metrics, Grafana dashboards, telemetry

## Repository Structure

```
prism-v2/
├── prism-core/          # Core types, traits, and error handling
├── prism-gpu/           # CUDA kernels and GPU abstractions
├── prism-fluxnet/       # Universal RL controller and Q-table management
├── prism-phases/        # 7-phase pipeline implementations (Phase0-Phase7)
├── prism-pipeline/      # Orchestrator, telemetry, profiling
├── prism-cli/           # Command-line interface
├── foundation/          # Legacy modules (shared-types, prct-core, quantum, neuromorphic)
├── docs/                # Specification, monitoring, and phase documentation
├── scripts/             # Build, test, and deployment automation
├── dashboards/          # Grafana visualization configs
├── benchmarks/dimacs/   # DIMACS benchmark graphs
├── data/                # Research datasets
└── staging_cleanup/     # Archived legacy files (see ARCHIVE_POLICY.md)
```

## Quick Start

### Prerequisites

- Rust 1.70+ with cargo
- CUDA Toolkit 12.0+ (for GPU features)
- NVIDIA GPU with Compute Capability 7.0+ (RTX 2060 or better)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/prism-v2.git
cd prism-v2

# Build release binary (CPU mode)
cargo build --release

# Build with GPU support
cargo build --release --features cuda
```

### Basic Usage

```bash
# Run with CPU-only mode
./target/release/prism-cli --input benchmarks/dimacs/myciel5.col

# Run with GPU acceleration
./target/release/prism-cli --input benchmarks/dimacs/DSJC500.5.col --gpu

# Multi-GPU mode
./target/release/prism-cli --input benchmarks/dimacs/DSJC1000.5.col --gpu --devices 0,1

# With warmstart curriculum
./target/release/prism-cli \
  --input benchmarks/dimacs/le450_25a.col \
  --gpu \
  --warmstart curriculum.json \
  --curriculum-profile adaptive
```

## Architecture Overview

### 7-Phase Pipeline

1. **Phase 0**: Pre-warmstart graph analysis
2. **Phase 1**: Quantum annealing initialization
3. **Phase 2**: Temperature-driven chromatic compression
4. **Phase 3**: Critical path geodesic routing
5. **Phase 4**: Topological singularity resolution
6. **Phase 5**: Reservoir-stabilized local search
7. **Phase 6**: Hybrid ensemble validation

### RL Integration

Each phase has a FluxNet RL controller that:
- Observes: difficulty, uncertainty, temperature, compaction ratio
- Acts: 8 temperature control actions (Neutral, IncreaseWeak, DecreaseStrong, etc.)
- Learns: Q-tables persist across runs for curriculum building

### Telemetry

All phase executions emit:
- **JSON/NDJSON**: Structured event logs
- **SQLite**: Queryable metrics database
- **Prometheus**: Real-time monitoring endpoints

## Building & Testing

```bash
# Format code
cargo fmt --all

# Lint with clippy
cargo clippy --workspace -- -D warnings

# Run tests (CPU mode)
cargo test --workspace --no-default-features

# Run tests (GPU mode, requires hardware)
cargo test --workspace --features cuda

# Build documentation
cargo doc --workspace --no-deps --open
```

## Monitoring Setup

1. Start Prometheus (port 9090):
   ```bash
   prometheus --config.file=scripts/prometheus.yml
   ```

2. Start Grafana (port 3000):
   ```bash
   grafana-server --config=dashboards/grafana.ini
   ```

3. Import dashboards from `dashboards/prism_overview.json`

See [docs/monitoring.md](docs/monitoring.md) for detailed setup.

## Benchmarks

Run standard DIMACS benchmarks:

```bash
./scripts/run_dimacs_suite.sh --gpu
```

Results stored in `results/dimacs_results.csv`.

## Profiling

Generate flame graphs and memory profiles:

```bash
# CPU profiling
cargo flamegraph --bin prism-cli -- --input graph.col

# Memory profiling
valgrind --tool=massif ./target/release/prism-cli --input graph.col

# GPU kernel profiling
nsys profile ./target/release/prism-cli --input graph.col --gpu
```

See [README_PROFILING.md](README_PROFILING.md) for advanced profiling techniques.

## Configuration

Example `config.toml`:

```toml
[pipeline]
max_iterations = 10000
early_stop_threshold = 0
use_gpu = true

[gpu]
devices = [0]
enable_profiling = true

[fluxnet]
learning_rate = 0.1
discount_factor = 0.95
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

[telemetry]
enable_json = true
enable_sqlite = true
enable_prometheus = true
output_dir = "output/telemetry"
```

## Documentation

- [Specification](docs/spec/prism_gpu_plan.md) - Complete architecture and requirements
- [Glossary](docs/spec/glossary.md) - Technical terminology reference
- [Monitoring](docs/monitoring.md) - Observability and metrics guide
- [Phase Guides](docs/) - Detailed phase documentation (phase*.md)

## Contributing

1. Read [ARCHIVE_POLICY.md](ARCHIVE_POLICY.md) to understand repository structure
2. Follow workspace module boundaries (see docs/spec/)
3. All new features require tests and documentation
4. Run `cargo fmt` and `cargo clippy` before committing
5. GPU code must include CPU fallback paths

## License

MIT License - see [LICENSE](LICENSE) for details

## Citation

If you use PRISM in academic work, please cite:

```bibtex
@software{prism_v2,
  title = {PRISM: GPU-Accelerated Graph Coloring with Reinforcement Learning},
  author = {Your Team},
  year = {2024},
  url = {https://github.com/your-org/prism-v2}
}
```

## Support

- GitHub Issues: https://github.com/your-org/prism-v2/issues
- Documentation: https://prism-docs.example.com
- Email: support@example.com

---

**Status**: Production-ready (v2.0.0)
**Last Updated**: 2025-11-18
**Cleanup Branch**: cleanup-playground
