#!/bin/bash

# PRACTICAL 17-COLOR TEST SCRIPT

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ATTEMPTS="${1:-10}"

export PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6
export RUST_LOG=info

echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════${NC}"
echo -e "${CYAN}${BOLD}    17-COLOR TEST (μ=0.5 AGGRESSIVE)${NC}"
echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════${NC}"
echo -e "Attempts: ${YELLOW}$ATTEMPTS${NC}"
echo -e "Config: ${YELLOW}AGGRESSIVE_17.toml${NC}"
echo -e "Chemical Potential: ${RED}μ=0.5${NC}"
echo ""

BEST=999
SUCCESSES=0

for i in $(seq 1 $ATTEMPTS); do
    echo -ne "${CYAN}Attempt $i/$ATTEMPTS:${NC} "

    START=$(date +%s)
    output=$(timeout 90 ./target/release/prism-cli \
        -i benchmarks/dimacs/DSJC125.5.col \
        --config configs/AGGRESSIVE_17.toml \
        --attempts 1 2>&1)
    END=$(date +%s)
    DURATION=$((END - START))

    colors=$(echo "$output" | grep "Best chromatic number:" | tail -1 | awk '{print $5}')
    conflicts=$(echo "$output" | grep "Best conflicts:" | tail -1 | awk '{print $4}')

    if [ -n "$colors" ]; then
        if [ "$colors" -le 17 ]; then
            echo -e "${GREEN}${BOLD}★ $colors colors!${NC} (${DURATION}s)"
            SUCCESSES=$((SUCCESSES + 1))
            echo "$output" > "solution_${colors}_colors_$(date +%s).log"
        elif [ "$colors" -le 19 ]; then
            echo -e "${YELLOW}$colors colors${NC} (${DURATION}s)"
        else
            echo -e "${CYAN}$colors colors${NC} (${DURATION}s)"
        fi

        if [ "$colors" -lt "$BEST" ]; then
            BEST=$colors
        fi
    else
        echo -e "${RED}Failed/Timeout${NC}"
    fi
done

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo -e "Best: ${YELLOW}$BEST colors${NC}"
if [ $SUCCESSES -gt 0 ]; then
    echo -e "${GREEN}17-color solutions: $SUCCESSES${NC}"
fi