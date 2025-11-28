#!/bin/bash

# BEAT ALL RECORDS - AGGRESSIVE MULTI-GRAPH ATTACK
# Runs all graphs with EXTREME settings targeting world records

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
MAGENTA='\033[0;35m'
NC='\033[0m'

export PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6
export RUST_LOG=info

echo -e "${MAGENTA}${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}${BOLD}    PRISM WORLD RECORD ATTACK - ALL GRAPHS${NC}"
echo -e "${MAGENTA}${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${RED}Chemical Potential: μ=0.9${NC}"
echo -e "${RED}Memetic: 1000 pop, 10000 gens${NC}"
echo -e "${RED}All Features: MAXIMUM${NC}"
echo ""

# Test configurations
declare -A GRAPHS
declare -A TARGETS
declare -A ATTEMPTS

# Graph files and their targets
GRAPHS[125.1]="benchmarks/dimacs/DSJC125.1.col"
TARGETS[125.1]=5
ATTEMPTS[125.1]=10

GRAPHS[125.5]="benchmarks/dimacs/DSJC125.5.col"
TARGETS[125.5]=17
ATTEMPTS[125.5]=50

GRAPHS[125.9]="benchmarks/dimacs/DSJC125.9.col"
TARGETS[125.9]=43
ATTEMPTS[125.9]=20

GRAPHS[250.5]="benchmarks/dimacs/DSJC250.5.col"
TARGETS[250.5]=27
ATTEMPTS[250.5]=10

GRAPHS[500.5]="benchmarks/dimacs/DSJC500.5.col"
TARGETS[500.5]=47
ATTEMPTS[500.5]=5

# Results tracking
RESULTS_FILE="world_record_attempts_$(date +%Y%m%d_%H%M%S).txt"

for key in 125.1 125.5 125.9 250.5 500.5; do
    GRAPH=${GRAPHS[$key]}
    TARGET=${TARGETS[$key]}
    NUM_ATTEMPTS=${ATTEMPTS[$key]}

    echo -e "${CYAN}════════════════════════════════════════${NC}"
    echo -e "${YELLOW}Testing DSJC$key${NC}"
    echo -e "Target: ${GREEN}$TARGET colors${NC}"
    echo -e "Attempts: ${YELLOW}$NUM_ATTEMPTS${NC}"
    echo ""

    BEST=999
    BEAT_RECORD=false

    for i in $(seq 1 $NUM_ATTEMPTS); do
        echo -ne "Attempt $i/$NUM_ATTEMPTS: "

        START=$(date +%s)
        output=$(timeout 600 ./target/release/prism-cli \
            -i "$GRAPH" \
            --config configs/EXTREME_MAX.toml \
            --attempts 1 2>&1)
        END=$(date +%s)

        colors=$(echo "$output" | grep "Best chromatic number:" | tail -1 | awk '{print $5}')

        if [ -n "$colors" ]; then
            if [ "$colors" -le "$TARGET" ]; then
                echo -e "${GREEN}${BOLD}★ BEAT RECORD: $colors colors! ★${NC}"
                echo "DSJC$key: RECORD BEATEN - $colors colors (target was $TARGET)" >> "$RESULTS_FILE"
                echo "$output" > "RECORD_DSJC${key}_${colors}colors_$(date +%s).log"
                BEAT_RECORD=true
            elif [ "$colors" -lt "$BEST" ]; then
                echo -e "${CYAN}$colors colors${NC}"
                BEST=$colors
            else
                echo "$colors colors"
            fi
        else
            echo -e "${RED}Failed${NC}"
        fi
    done

    if [ "$BEAT_RECORD" = true ]; then
        echo -e "${GREEN}${BOLD}✓ RECORD BEATEN FOR DSJC$key!${NC}"
    else
        echo -e "Best achieved: $BEST colors (target was $TARGET)"
    fi
    echo ""
done

echo -e "${MAGENTA}${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}${BOLD}                    SUMMARY${NC}"
echo -e "${MAGENTA}${BOLD}═══════════════════════════════════════════════════════${NC}"
cat "$RESULTS_FILE" 2>/dev/null || echo "No records beaten yet"