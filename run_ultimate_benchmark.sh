#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# PRISM ULTIMATE BENCHMARK SUITE
# ═══════════════════════════════════════════════════════════════════════════
# Runs ULTIMATE_MAX_GPU config across a range of benchmark graphs
# All phases enabled, aggressive parameters, full telemetry
# ═══════════════════════════════════════════════════════════════════════════

set -e

# Change to PRISM directory
cd /mnt/c/Users/Predator/Desktop/PRISM

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Create log directory
mkdir -p logs/ultimate

# Define benchmark suite
declare -a GRAPHS=(
    # EASY: Small graphs (< 200 vertices)
    "benchmarks/dimacs/DSJC125.1.col:Easy:5-6"
    "benchmarks/dimacs/DSJC125.5.col:Easy:17-18"
    "benchmarks/dimacs/DSJC125.9.col:Easy:44-46"

    # MEDIUM: Medium graphs (200-500 vertices)
    "benchmarks/dimacs/DSJC250.5.col:Medium:28-30"
    "benchmarks/dimacs/DSJR500.1.col:Medium:12-13"
    "benchmarks/dimacs/DSJC500.5.col:Medium:48-50"

    # HARD: Large graphs (500-1000 vertices)
    "benchmarks/dimacs/DSJC1000.5.col:Hard:83-88"
    "benchmarks/dimacs/le450_25a.col:Hard:25-26"

    # EXTREME: Dense or challenging graphs
    "benchmarks/dimacs/queen11_11.col:Extreme:11-12"
)

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "═══════════════════════════════════════════════════════════════════════════"
echo "  PRISM ULTIMATE BENCHMARK SUITE"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration: ULTIMATE_MAX_GPU.toml"
echo "Graphs to process: ${#GRAPHS[@]}"
echo "Features enabled:"
echo "  ✓ Complex Quantum Evolution (250 iterations)"
echo "  ✓ PIMC Quantum Annealing (80 replicas)"
echo "  ✓ FluxNet Universal RL (65536 state space)"
echo "  ✓ All 7 Phases + Metaphysical Coupling"
echo "  ✓ Memetic Evolution (30 population, 75 generations)"
echo "  ✓ Full GPU Acceleration"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

# Start time
TOTAL_START=$(date +%s)

# Results summary file
SUMMARY_FILE="results/ultimate/benchmark_summary_$(date +%Y%m%d_%H%M%S).txt"
echo "PRISM Ultimate Benchmark Summary" > "$SUMMARY_FILE"
echo "Started: $(date)" >> "$SUMMARY_FILE"
echo "Config: ULTIMATE_MAX_GPU.toml" >> "$SUMMARY_FILE"
echo "════════════════════════════════════════════════════════════════" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Counter
SUCCESS_COUNT=0
FAIL_COUNT=0
GRAPH_NUM=0

# Process each graph
for graph_entry in "${GRAPHS[@]}"; do
    GRAPH_NUM=$((GRAPH_NUM + 1))

    # Parse graph entry
    IFS=':' read -r graph_path difficulty target_chromatic <<< "$graph_entry"
    graph_name=$(basename "$graph_path" .col)

    echo ""
    echo "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo "${CYAN}  [$GRAPH_NUM/${#GRAPHS[@]}] Processing: ${graph_name} (${difficulty})${NC}"
    echo "${CYAN}  Target chromatic number: ${target_chromatic}${NC}"
    echo "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Check if graph exists
    if [ ! -f "$graph_path" ]; then
        echo "${RED}✗ Graph file not found: $graph_path${NC}"
        echo "SKIP: $graph_name - File not found" >> "$SUMMARY_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # Log file
    LOG_FILE="logs/ultimate/${graph_name}_ultimate.log"

    # Record start time
    START_TIME=$(date +%s)

    echo "${YELLOW}Starting PRISM with ULTIMATE_MAX_GPU config...${NC}"
    echo ""

    # Run PRISM
    if timeout 3600 ./target/release/prism-cli \
        --config configs/ULTIMATE_MAX_GPU.toml \
        --input "$graph_path" \
        --gpu \
        --verbose 2>&1 | tee "$LOG_FILE"; then

        # Calculate runtime
        END_TIME=$(date +%s)
        RUNTIME=$((END_TIME - START_TIME))

        # Extract results from log file
        CHROMATIC=$(grep -oP 'chromatic_number"?:\s*\K\d+' "$LOG_FILE" | tail -1 || echo "N/A")
        CONFLICTS=$(grep -oP 'conflicts"?:\s*\K\d+' "$LOG_FILE" | tail -1 || echo "N/A")

        echo ""
        echo "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
        echo "${GREEN}  ✓ SUCCESS: ${graph_name}${NC}"
        echo "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
        echo "${GREEN}  Chromatic number: ${CHROMATIC} (target: ${target_chromatic})${NC}"
        echo "${GREEN}  Conflicts: ${CONFLICTS}${NC}"
        echo "${GREEN}  Runtime: ${RUNTIME}s${NC}"
        echo "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"

        # Write to summary
        echo "✓ $graph_name ($difficulty)" >> "$SUMMARY_FILE"
        echo "  Chromatic: $CHROMATIC (target: $target_chromatic)" >> "$SUMMARY_FILE"
        echo "  Conflicts: $CONFLICTS" >> "$SUMMARY_FILE"
        echo "  Runtime: ${RUNTIME}s" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"

        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        RUNTIME=$((END_TIME - START_TIME))

        echo ""
        echo "${RED}═══════════════════════════════════════════════════════════════════════════${NC}"
        echo "${RED}  ✗ FAILED: ${graph_name} (exit code: $EXIT_CODE)${NC}"
        echo "${RED}  Runtime before failure: ${RUNTIME}s${NC}"
        echo "${RED}═══════════════════════════════════════════════════════════════════════════${NC}"

        echo "✗ $graph_name - Failed (exit $EXIT_CODE) after ${RUNTIME}s" >> "$SUMMARY_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # Brief pause between runs
    sleep 2
done

# Calculate total time
TOTAL_END=$(date +%s)
TOTAL_RUNTIME=$((TOTAL_END - TOTAL_START))
HOURS=$((TOTAL_RUNTIME / 3600))
MINUTES=$(((TOTAL_RUNTIME % 3600) / 60))
SECONDS=$((TOTAL_RUNTIME % 60))

# Final summary
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  BENCHMARK SUITE COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "Total graphs: ${#GRAPHS[@]}"
echo "${GREEN}Successful: ${SUCCESS_COUNT}${NC}"
echo "${RED}Failed: ${FAIL_COUNT}${NC}"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved to: $SUMMARY_FILE"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"

# Write final summary
echo "" >> "$SUMMARY_FILE"
echo "════════════════════════════════════════════════════════════════" >> "$SUMMARY_FILE"
echo "FINAL SUMMARY" >> "$SUMMARY_FILE"
echo "════════════════════════════════════════════════════════════════" >> "$SUMMARY_FILE"
echo "Total graphs: ${#GRAPHS[@]}" >> "$SUMMARY_FILE"
echo "Successful: $SUCCESS_COUNT" >> "$SUMMARY_FILE"
echo "Failed: $FAIL_COUNT" >> "$SUMMARY_FILE"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s" >> "$SUMMARY_FILE"
echo "Completed: $(date)" >> "$SUMMARY_FILE"

# Display summary file
cat "$SUMMARY_FILE"

echo ""
echo "✓ Benchmark suite complete!"
echo ""
