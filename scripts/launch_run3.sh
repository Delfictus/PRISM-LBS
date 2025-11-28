#!/bin/bash
# Run 3: 512 attempts with phase2_fast (6K iterations, temp_max=12.0)
# Launches immediately after Run 2 completion

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "════════════════════════════════════════════════════════"
echo "PRISM Run 3: Fast Phase 2 + Enhanced Memetic"
echo "════════════════════════════════════════════════════════"
echo "Config: configs/dsjc250_memetic_run3.toml"
echo "Attempts: 512"
echo "Phase 2: 6K iterations (fast profile)"
echo "Memetic: mutation=0.06, local_search=150"
echo "Expected GPU time: ~120 minutes (14s × 512)"
echo "════════════════════════════════════════════════════════"
echo ""

"$PROJECT_ROOT/target/release/prism-cli" \
  --input "$PROJECT_ROOT/benchmarks/dimacs/DSJC250.5.col" \
  --gpu \
  --attempts 512 \
  --config "$PROJECT_ROOT/configs/dsjc250_memetic_run3.toml" \
  --verbose \
  2>&1 | tee "$PROJECT_ROOT/logs/run_512attempts_fast_20251119.log"
