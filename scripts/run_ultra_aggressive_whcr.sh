#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# ULTRA-AGGRESSIVE WHCR WORLD RECORD ATTEMPT SCRIPT
# ═══════════════════════════════════════════════════════════════════════════
# Target: DSJC125.5 → 17 colors, 0 conflicts
# Strategy: μ=0.9 compression + WHCR multi-phase repair
# ═══════════════════════════════════════════════════════════════════════════

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
GRAPH="${1:-benchmarks/dimacs/DSJC125.5.col}"
CONFIG="configs/ULTRA_AGGRESSIVE_WHCR.toml"
OUTPUT_DIR="results/ultra_whcr_$(date +%Y%m%d_%H%M%S)"
NUM_ATTEMPTS="${2:-5}"

echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}   PRISM ULTRA-AGGRESSIVE WHCR - WORLD RECORD ATTEMPT${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Graph:        ${YELLOW}${GRAPH}${NC}"
echo -e "  Config:       ${YELLOW}${CONFIG}${NC}"
echo -e "  Output:       ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "  Attempts:     ${YELLOW}${NUM_ATTEMPTS}${NC}"
echo -e "  Strategy:     ${MAGENTA}μ=0.9 + WHCR checkpoints (Phase 2,3,5,7)${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: PRE-FLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════════════════

echo -e "${CYAN}[1/5] Pre-flight checks...${NC}"

# Check graph file exists
if [ ! -f "$GRAPH" ]; then
    echo -e "${RED}ERROR: Graph file not found: $GRAPH${NC}"
    echo -e "${YELLOW}Available graphs:${NC}"
    ls -1 benchmarks/dimacs/*.col 2>/dev/null || echo "  (no .col files in benchmarks/dimacs/)"
    exit 1
fi

# Check config file exists
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}ERROR: Config file not found: $CONFIG${NC}"
    exit 1
fi

# Check if prism-cli is built
if [ ! -f "target/release/prism-cli" ]; then
    echo -e "${YELLOW}WARNING: prism-cli not found. Building now...${NC}"
    cargo build --release --features cuda || {
        echo -e "${RED}ERROR: Build failed!${NC}"
        exit 1
    }
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}✓ Pre-flight checks passed${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: VERIFY CHEMICAL POTENTIAL SETTING
# ═══════════════════════════════════════════════════════════════════════════

echo -e "${CYAN}[2/5] Verifying chemical potential μ=0.9...${NC}"

QUANTUM_KERNEL="prism-gpu/src/kernels/quantum.cu"
if [ -f "$QUANTUM_KERNEL" ]; then
    CHEMICAL_POT=$(grep -E "^\s*float\s+chemical_potential\s*=" "$QUANTUM_KERNEL" | head -1 || echo "")

    if echo "$CHEMICAL_POT" | grep -q "0.9"; then
        echo -e "${GREEN}✓ Chemical potential μ=0.9 confirmed in kernel${NC}"
        echo -e "  ${CHEMICAL_POT}"
    else
        echo -e "${YELLOW}⚠️  WARNING: Chemical potential may not be set to 0.9${NC}"
        echo -e "  Current: ${CHEMICAL_POT}"
        echo -e "  ${MAGENTA}Please edit ${QUANTUM_KERNEL} line ~431${NC}"
        echo -e "  ${MAGENTA}Set: float chemical_potential = 0.9f;${NC}"
        echo ""
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${RED}Aborted. Please set μ=0.9 and rebuild.${NC}"
            exit 1
        fi
    fi
else
    echo -e "${YELLOW}⚠️  WARNING: Cannot verify chemical potential (kernel file not found)${NC}"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: VERIFY PTX KERNELS
# ═══════════════════════════════════════════════════════════════════════════

echo -e "${CYAN}[3/5] Verifying GPU kernels (PTX)...${NC}"

PTX_KERNELS=(
    "target/ptx/quantum.ptx"
    "target/ptx/thermodynamic.ptx"
    "target/ptx/whcr.ptx"
)

MISSING_PTX=0
for ptx in "${PTX_KERNELS[@]}"; do
    if [ -f "$ptx" ]; then
        SIZE=$(ls -lh "$ptx" | awk '{print $5}')
        echo -e "${GREEN}✓${NC} $ptx (${SIZE})"
    else
        echo -e "${RED}✗${NC} $ptx ${RED}(MISSING)${NC}"
        MISSING_PTX=1
    fi
done

if [ $MISSING_PTX -eq 1 ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Missing PTX kernels. Attempting to compile...${NC}"

    # Try to compile PTX
    if [ -f "scripts/compile_ptx.sh" ]; then
        bash scripts/compile_ptx.sh
    else
        echo -e "${YELLOW}Compiling PTX kernels manually...${NC}"
        nvcc --ptx -o target/ptx/quantum.ptx prism-gpu/src/kernels/quantum.cu -arch=sm_70 --std=c++14 -Xcompiler -fPIC || true
        nvcc --ptx -o target/ptx/thermodynamic.ptx prism-gpu/src/kernels/thermodynamic.cu -arch=sm_70 --std=c++14 -Xcompiler -fPIC || true
        nvcc --ptx -o target/ptx/whcr.ptx prism-gpu/src/kernels/whcr.cu -arch=sm_70 --std=c++14 -Xcompiler -fPIC || true
    fi

    # Rebuild Rust
    echo -e "${YELLOW}Rebuilding with fresh PTX...${NC}"
    cargo build --release --features cuda
fi

echo -e "${GREEN}✓ PTX kernels ready${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: CHECK GPU AVAILABILITY
# ═══════════════════════════════════════════════════════════════════════════

echo -e "${CYAN}[4/5] Checking GPU availability...${NC}"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | while read line; do
        echo -e "${GREEN}✓${NC} GPU: $line"
    done
else
    echo -e "${YELLOW}⚠️  nvidia-smi not found. Cannot verify GPU.${NC}"
    echo -e "${YELLOW}   Make sure CUDA drivers are installed.${NC}"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: RUN ATTEMPTS
# ═══════════════════════════════════════════════════════════════════════════

echo -e "${CYAN}[5/5] Running ${NUM_ATTEMPTS} world record attempts...${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""

BEST_COLORS=999
BEST_RESULT=""
SUCCESS=0

for attempt in $(seq 1 $NUM_ATTEMPTS); do
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}  ATTEMPT $attempt / $NUM_ATTEMPTS${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    RESULT_FILE="$OUTPUT_DIR/attempt_${attempt}.json"
    TELEMETRY_FILE="$OUTPUT_DIR/telemetry_${attempt}.jsonl"
    LOG_FILE="$OUTPUT_DIR/log_${attempt}.txt"

    echo -e "${BLUE}Running PRISM...${NC}"
    echo -e "  Output:    $RESULT_FILE"
    echo -e "  Telemetry: $TELEMETRY_FILE"
    echo -e "  Log:       $LOG_FILE"
    echo ""

    # Run PRISM (capture start time)
    START_TIME=$(date +%s)

    ./target/release/prism-cli \
        --input "$GRAPH" \
        --config "$CONFIG" \
        --gpu \
        --verbose \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo ""
    echo -e "${BLUE}Completed in ${ELAPSED}s${NC}"
    echo ""

    if [ $EXIT_CODE -ne 0 ]; then
        echo -e "${RED}✗ Attempt $attempt FAILED (exit code $EXIT_CODE)${NC}"
        echo ""
        continue
    fi

    # Extract results
    if [ -f "$RESULT_FILE" ]; then
        NUM_COLORS=$(jq -r '.num_colors // "N/A"' "$RESULT_FILE" 2>/dev/null || echo "N/A")
        NUM_CONFLICTS=$(jq -r '.num_conflicts // "N/A"' "$RESULT_FILE" 2>/dev/null || echo "N/A")
        STRESS=$(jq -r '.geometric_stress // "N/A"' "$RESULT_FILE" 2>/dev/null || echo "N/A")

        echo -e "${CYAN}Results:${NC}"
        echo -e "  Colors:    ${YELLOW}${NUM_COLORS}${NC}"
        echo -e "  Conflicts: ${YELLOW}${NUM_CONFLICTS}${NC}"
        echo -e "  Stress:    ${YELLOW}${STRESS}${NC}"
        echo ""

        # Check for world record
        if [ "$NUM_COLORS" = "17" ] && [ "$NUM_CONFLICTS" = "0" ]; then
            echo -e "${GREEN}★★★ WORLD RECORD ACHIEVED! ★★★${NC}"
            echo -e "${GREEN}17 colors, 0 conflicts, stress=${STRESS}${NC}"
            echo ""
            SUCCESS=1
            BEST_COLORS=17
            BEST_RESULT="$RESULT_FILE"

            # Copy to special location
            cp "$RESULT_FILE" "$OUTPUT_DIR/WORLD_RECORD.json"
            cp "$TELEMETRY_FILE" "$OUTPUT_DIR/WORLD_RECORD_telemetry.jsonl"

            break  # Stop attempts, we won!
        elif [ "$NUM_COLORS" -le "$BEST_COLORS" ]; then
            BEST_COLORS=$NUM_COLORS
            BEST_RESULT="$RESULT_FILE"
            echo -e "${GREEN}✓ New best: ${NUM_COLORS} colors${NC}"
        else
            echo -e "${YELLOW}✓ Completed: ${NUM_COLORS} colors${NC}"
        fi
    else
        echo -e "${RED}✗ Result file not created${NC}"
    fi

    echo ""
done

# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}   FINAL SUMMARY${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""

if [ $SUCCESS -eq 1 ]; then
    echo -e "${GREEN}🏆 WORLD RECORD ACHIEVED! 🏆${NC}"
    echo -e "${GREEN}   17 colors, 0 conflicts on DSJC125.5${NC}"
    echo ""
    echo -e "${CYAN}Solution saved to:${NC}"
    echo -e "  ${YELLOW}$OUTPUT_DIR/WORLD_RECORD.json${NC}"
    echo ""
    echo -e "${CYAN}Verification command:${NC}"
    echo -e "  ${YELLOW}./target/release/prism-cli --verify $OUTPUT_DIR/WORLD_RECORD.json --input $GRAPH${NC}"
else
    echo -e "${YELLOW}Best result: ${BEST_COLORS} colors${NC}"

    if [ -n "$BEST_RESULT" ]; then
        echo -e "${CYAN}Best solution saved to:${NC}"
        echo -e "  ${YELLOW}${BEST_RESULT}${NC}"

        cp "$BEST_RESULT" "$OUTPUT_DIR/BEST.json"
        echo -e "${CYAN}Also copied to:${NC}"
        echo -e "  ${YELLOW}$OUTPUT_DIR/BEST.json${NC}"
    fi

    echo ""
    echo -e "${YELLOW}Recommendations:${NC}"

    if [ "$BEST_COLORS" -eq 18 ]; then
        echo -e "  • Very close! Try increasing evolution_iterations to 800"
        echo -e "  • Verify chemical_potential = 0.9f in quantum.cu kernel"
        echo -e "  • Check WHCR checkpoints in telemetry (should reduce conflicts)"
    elif [ "$BEST_COLORS" -ge 19 ]; then
        echo -e "  • Increase chemical potential aggressiveness"
        echo -e "  • Increase Phase 2 compaction_factor to 0.85"
        echo -e "  • Increase quantum coupling_strength to 12.0"
    fi
fi

echo ""
echo -e "${CYAN}All results in: ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
