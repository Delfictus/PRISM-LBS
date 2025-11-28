#!/bin/bash

# EXTREME MAXIMUM AGGRESSION TEST
# Chemical Potential μ=0.9, Unlimited Memetic

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
MAGENTA='\033[0;35m'
NC='\033[0m'

ATTEMPTS="${1:-10}"

export PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6
export RUST_LOG=info

echo -e "${MAGENTA}${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}${BOLD}    EXTREME MAXIMUM AGGRESSION (μ=0.9)${NC}"
echo -e "${MAGENTA}${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${RED}Chemical Potential: μ=0.9 (MAXIMUM)${NC}"
echo -e "${RED}Memetic: 1000 pop, 10000 gens (UNLIMITED)${NC}"
echo -e "${RED}All Features: MAXIMUM SETTINGS${NC}"
echo -e "${YELLOW}Attempts: $ATTEMPTS${NC}"
echo ""
echo -e "${CYAN}WARNING: Each attempt may take 3-5 minutes!${NC}"
echo ""

BEST=999
SEVENTEEN_COUNT=0

for i in $(seq 1 $ATTEMPTS); do
    echo -e "${CYAN}════════════════════════════════════════${NC}"
    echo -e "${CYAN}Attempt $i/$ATTEMPTS starting...${NC}"

    START=$(date +%s)

    # No timeout - let it run to completion
    ./target/release/prism-cli \
        -i benchmarks/dimacs/DSJC125.5.col \
        --config configs/EXTREME_MAX.toml \
        --attempts 1 2>&1 | tee extreme_attempt_${i}.log | \
        grep -E "(Phase|Best chromatic|conflicts|Memetic Gen.*colors)" | tail -20

    END=$(date +%s)
    DURATION=$((END - START))
    MINS=$((DURATION / 60))
    SECS=$((DURATION % 60))

    colors=$(grep "Best chromatic number:" extreme_attempt_${i}.log | tail -1 | awk '{print $5}')
    conflicts=$(grep "Best conflicts:" extreme_attempt_${i}.log | tail -1 | awk '{print $4}')

    echo ""
    if [ -n "$colors" ]; then
        if [ "$colors" -eq 17 ]; then
            echo -e "${GREEN}${BOLD}╔══════════════════════════════════════╗${NC}"
            echo -e "${GREEN}${BOLD}║   🏆 WORLD RECORD: 17 COLORS! 🏆    ║${NC}"
            echo -e "${GREEN}${BOLD}╚══════════════════════════════════════╝${NC}"
            SEVENTEEN_COUNT=$((SEVENTEEN_COUNT + 1))
            cp extreme_attempt_${i}.log "WORLD_RECORD_17_$(date +%s).log"
        elif [ "$colors" -le 19 ]; then
            echo -e "${YELLOW}${BOLD}EXCELLENT: $colors colors${NC}"
        elif [ "$colors" -le 21 ]; then
            echo -e "${CYAN}Good: $colors colors${NC}"
        else
            echo -e "Result: $colors colors"
        fi

        echo -e "Time: ${MINS}m ${SECS}s"

        if [ "$colors" -lt "$BEST" ]; then
            BEST=$colors
            echo -e "${GREEN}New best: $BEST colors!${NC}"
        fi
    else
        echo -e "${RED}Failed or invalid${NC}"
    fi
done

echo ""
echo -e "${MAGENTA}${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}${BOLD}                    FINAL RESULTS${NC}"
echo -e "${MAGENTA}${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}Best achieved: ${YELLOW}$BEST colors${NC}"
if [ $SEVENTEEN_COUNT -gt 0 ]; then
    echo -e "${GREEN}${BOLD}17-color solutions found: $SEVENTEEN_COUNT${NC}"
fi
echo ""