#!/bin/bash

# OVERNIGHT MARATHON - Runs continuously logging all improvements

export PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6
export RUST_LOG=error  # Minimal logging for speed

LOG_FILE="marathon_$(date +%Y%m%d_%H%M%S).log"
BEST_125_5=999
BEST_125_9=999
BEST_250_5=999

echo "Starting overnight marathon at $(date)" | tee "$LOG_FILE"
echo "Press Ctrl+C to stop" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    echo "=== Iteration $ITERATION ===" | tee -a "$LOG_FILE"

    # DSJC125.5 - Target 17
    echo -n "DSJC125.5: " | tee -a "$LOG_FILE"
    output=$(timeout 300 ./target/release/prism-cli \
        -i benchmarks/dimacs/DSJC125.5.col \
        --config configs/EXTREME_MAX.toml \
        --attempts 1 2>&1)
    colors=$(echo "$output" | grep "Best chromatic" | tail -1 | awk '{print $5}')
    if [ -n "$colors" ]; then
        echo "$colors colors" | tee -a "$LOG_FILE"
        if [ "$colors" -lt "$BEST_125_5" ]; then
            BEST_125_5=$colors
            echo "  NEW BEST for 125.5: $colors" | tee -a "$LOG_FILE"
            if [ "$colors" -eq 17 ]; then
                echo "  ★★★ WORLD RECORD 17 COLORS! ★★★" | tee -a "$LOG_FILE"
                echo "$output" > "WORLD_RECORD_17_$(date +%s).log"
            fi
        fi
    fi

    # DSJC125.9 - Target 43
    echo -n "DSJC125.9: " | tee -a "$LOG_FILE"
    output=$(timeout 300 ./target/release/prism-cli \
        -i benchmarks/dimacs/DSJC125.9.col \
        --config configs/EXTREME_MAX.toml \
        --attempts 1 2>&1)
    colors=$(echo "$output" | grep "Best chromatic" | tail -1 | awk '{print $5}')
    if [ -n "$colors" ]; then
        echo "$colors colors" | tee -a "$LOG_FILE"
        if [ "$colors" -lt "$BEST_125_9" ]; then
            BEST_125_9=$colors
            echo "  NEW BEST for 125.9: $colors" | tee -a "$LOG_FILE"
        fi
    fi

    # DSJC250.5 - Target 27
    echo -n "DSJC250.5: " | tee -a "$LOG_FILE"
    output=$(timeout 600 ./target/release/prism-cli \
        -i benchmarks/dimacs/DSJC250.5.col \
        --config configs/EXTREME_MAX.toml \
        --attempts 1 2>&1)
    colors=$(echo "$output" | grep "Best chromatic" | tail -1 | awk '{print $5}')
    if [ -n "$colors" ]; then
        echo "$colors colors" | tee -a "$LOG_FILE"
        if [ "$colors" -lt "$BEST_250_5" ]; then
            BEST_250_5=$colors
            echo "  NEW BEST for 250.5: $colors" | tee -a "$LOG_FILE"
        fi
    fi

    echo "Current bests: 125.5=$BEST_125_5, 125.9=$BEST_125_9, 250.5=$BEST_250_5" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done