#!/bin/bash
# FluxNet v2 Telemetry Data Collection Script
#
# Runs PRISM on small graphs to collect state/action/reward tuples
# for retraining the FluxNet RL controller.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/target/fluxnet_logs"
PRISM_CLI="$PROJECT_ROOT/target/release/prism-cli"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

echo "════════════════════════════════════════════════════════"
echo "FluxNet v2 Telemetry Data Collection"
echo "════════════════════════════════════════════════════════"
echo ""

# Check if PRISM CLI exists
if [ ! -f "$PRISM_CLI" ]; then
    echo "❌ Error: prism-cli not found at $PRISM_CLI"
    echo "   Run: cargo build --release"
    exit 1
fi

# Collection configuration
GRAPHS=(
    "benchmarks/dimacs/DSJC125.1.col:dsjc125_1"
    "benchmarks/dimacs/DSJC125.5.col:dsjc125_5"
    "benchmarks/dimacs/DSJC125.9.col:dsjc125_9"
    "benchmarks/dimacs/DSJC250.1.col:dsjc250_1"
    "benchmarks/dimacs/DSJC250.5.col:dsjc250_5"
)

ATTEMPTS=10
TOTAL_GRAPHS=${#GRAPHS[@]}
CURRENT=0

for GRAPH_ENTRY in "${GRAPHS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # Parse graph path and name
    IFS=':' read -r GRAPH_PATH GRAPH_NAME <<< "$GRAPH_ENTRY"
    
    echo "[$CURRENT/$TOTAL_GRAPHS] Processing $GRAPH_NAME..."
    echo "  Graph: $GRAPH_PATH"
    echo "  Attempts: $ATTEMPTS"
    echo "  Output: $LOG_DIR/${GRAPH_NAME}_telemetry.jsonl"
    
    # Check if graph file exists
    if [ ! -f "$PROJECT_ROOT/$GRAPH_PATH" ]; then
        echo "  ⚠️  Skipping: Graph file not found"
        continue
    fi
    
    # Run PRISM with telemetry enabled
    "$PRISM_CLI" \
        --input "$PROJECT_ROOT/$GRAPH_PATH" \
        --gpu \
        --attempts "$ATTEMPTS" \
        --verbose \
        2>&1 | tee "$LOG_DIR/${GRAPH_NAME}_run.log" | \
        grep -E "FluxNet|State|Action|Reward|phase.*temperature|coherence_cv|attempt_progress" \
        > "$LOG_DIR/${GRAPH_NAME}_telemetry.jsonl" || true
    
    echo "  ✅ Complete"
    echo ""
done

echo "════════════════════════════════════════════════════════"
echo "Collection Summary"
echo "════════════════════════════════════════════════════════"
echo ""

# Count collected samples
TOTAL_SAMPLES=0
for LOG_FILE in "$LOG_DIR"/*_telemetry.jsonl; do
    if [ -f "$LOG_FILE" ]; then
        SAMPLES=$(wc -l < "$LOG_FILE")
        TOTAL_SAMPLES=$((TOTAL_SAMPLES + SAMPLES))
        echo "$(basename "$LOG_FILE"): $SAMPLES samples"
    fi
done

echo ""
echo "Total telemetry samples: $TOTAL_SAMPLES"
echo "Output directory: $LOG_DIR"
echo ""
echo "Next step: Run training"
echo "  cargo run --release --bin fluxnet_train \\"
echo "    --logs $LOG_DIR \\"
echo "    --output target/fluxnet/qtable_v2.bin"
