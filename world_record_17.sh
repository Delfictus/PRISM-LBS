#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════
# PRISM WORLD RECORD ATTEMPT - 17 COLORS FOR DSJC125.5
# ═══════════════════════════════════════════════════════════════════════════
# Chemical Potential μ=0.5 forces extreme color compression
# This script attempts to achieve the theoretical minimum of 17 colors
# ═══════════════════════════════════════════════════════════════════════════

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
GRAPH="benchmarks/dimacs/DSJC125.5.col"
CONFIG="configs/WORLD_RECORD_17.toml"
ATTEMPTS="${1:-100}"  # Default 100 attempts

# Environment setup for WSL2
export PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6
export RUST_LOG=info

clear
echo -e "${MAGENTA}${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}${BOLD}     PRISM WORLD RECORD ATTEMPT - TARGET: 17 COLORS${NC}"
echo -e "${MAGENTA}${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${CYAN}Graph:${NC} ${YELLOW}DSJC125.5.col${NC} (125 vertices, 50% edge density)"
echo -e "${CYAN}Target:${NC} ${GREEN}17 colors${NC} (theoretical minimum)"
echo -e "${CYAN}Current Best Known:${NC} ${YELLOW}17 colors${NC} (world record)"
echo -e "${CYAN}Chemical Potential:${NC} ${RED}μ=0.5${NC} (10x aggressive compression)"
echo -e "${CYAN}Attempts:${NC} ${YELLOW}$ATTEMPTS${NC}"
echo ""
echo -e "${BLUE}Configuration Highlights:${NC}"
echo -e "  • Population: ${YELLOW}500${NC} individuals (massive search)"
echo -e "  • Generations: ${YELLOW}5000${NC} (extreme depth)"
echo -e "  • Local Search: ${YELLOW}95%${NC} intensity"
echo -e "  • Quantum Colors: ${YELLOW}30${NC} max (forcing compression)"
echo -e "  • Coupling: ${YELLOW}7.0${NC} (very strong anti-ferromagnetic)"
echo ""
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Check if graph exists
if [ ! -f "$GRAPH" ]; then
    echo -e "${RED}Error: Graph file not found: $GRAPH${NC}"
    exit 1
fi

# Results tracking
BEST_COLORS=999
BEST_CONFLICTS=999
SUCCESS_COUNT=0
START_TIME=$(date +%s)
LOG_FILE="world_record_attempt_$(date +%Y%m%d_%H%M%S).log"

echo -e "${GREEN}Starting world record attempt...${NC}\n"

# Function to run single attempt
run_attempt() {
    local attempt_num=$1

    echo -ne "${CYAN}Attempt $attempt_num/$ATTEMPTS:${NC} "

    # Run with timeout (10 minutes per attempt for deep search)
    local output=$(timeout 600 ./target/release/prism-cli \
        -i "$GRAPH" \
        --config "$CONFIG" \
        --attempts 1 \
        --verbose 2>&1)

    # Extract results
    local colors=$(echo "$output" | grep "Best chromatic number:" | tail -1 | awk '{print $5}')
    local conflicts=$(echo "$output" | grep "Best conflicts:" | tail -1 | awk '{print $4}')
    local valid=$(echo "$output" | grep "Valid:" | tail -1 | awk '{print $2}')

    # Log full output for this attempt
    echo "$output" >> "$LOG_FILE"
    echo "---END ATTEMPT $attempt_num---" >> "$LOG_FILE"

    # Display result
    if [ "$valid" = "true" ] && [ -n "$colors" ]; then
        if [ "$colors" -eq 17 ]; then
            echo -e "${GREEN}${BOLD}★★★ WORLD RECORD! 17 COLORS! ★★★${NC}"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo "WORLD RECORD ACHIEVED: Attempt $attempt_num - 17 colors" >> "$LOG_FILE"

            # Save the solution
            echo "$output" > "WORLD_RECORD_17_SOLUTION_$(date +%Y%m%d_%H%M%S).log"

            return 0  # Signal success
        elif [ "$colors" -lt "$BEST_COLORS" ]; then
            echo -e "${GREEN}NEW BEST: $colors colors${NC}"
            BEST_COLORS=$colors
            BEST_CONFLICTS=$conflicts
        elif [ "$colors" -le 20 ]; then
            echo -e "${YELLOW}Close: $colors colors${NC}"
        else
            echo -e "${CYAN}$colors colors${NC}"
        fi
    else
        echo -e "${RED}Invalid or timeout${NC}"
    fi

    return 1  # Continue searching
}

# Main loop
for i in $(seq 1 $ATTEMPTS); do
    run_attempt $i

    # Check if we found 17 colors
    if [ $? -eq 0 ]; then
        # Found world record, but continue to find more instances
        echo -e "\n${GREEN}Continuing to find more 17-color solutions...${NC}"
    fi

    # Progress report every 10 attempts
    if [ $((i % 10)) -eq 0 ]; then
        echo -e "\n${BLUE}Progress: $i/$ATTEMPTS attempts completed${NC}"
        echo -e "${BLUE}Best so far: $BEST_COLORS colors${NC}"
        if [ $SUCCESS_COUNT -gt 0 ]; then
            echo -e "${GREEN}17-color solutions found: $SUCCESS_COUNT${NC}"
        fi
        echo ""
    fi
done

# Final report
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo -e "\n${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}${BOLD}                    FINAL REPORT${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo -e "${GREEN}${BOLD}🏆 WORLD RECORD ACHIEVED! 🏆${NC}"
    echo -e "${GREEN}17-color solutions found: $SUCCESS_COUNT / $ATTEMPTS${NC}"
else
    echo -e "${YELLOW}World record not achieved (yet)${NC}"
    echo -e "${YELLOW}Best result: $BEST_COLORS colors${NC}"
fi

echo -e "${CYAN}Total attempts: $ATTEMPTS${NC}"
echo -e "${CYAN}Runtime: ${MINUTES}m ${SECONDS}s${NC}"
echo -e "${CYAN}Log file: $LOG_FILE${NC}"
echo ""

# Suggest next steps
if [ $BEST_COLORS -gt 17 ] && [ $BEST_COLORS -le 19 ]; then
    echo -e "${BLUE}Suggestions to reach 17:${NC}"
    echo -e "  1. Increase attempts (try 500-1000)"
    echo -e "  2. Tune chemical potential (try μ=0.6 or μ=0.7)"
    echo -e "  3. Increase memetic generations to 10000"
    echo -e "  4. Use multiple parallel runs"
fi

exit 0