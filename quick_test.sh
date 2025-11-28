#!/bin/bash

# QUICK TEST - 60 SECOND TIMEOUT, NO FREEZING!

export PATH="/home/diddy/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6

echo "Quick test with 60-second timeout..."
echo ""

for i in 1 2 3 4 5; do
    echo -n "Attempt $i: "

    # 60 SECOND HARD TIMEOUT - WON'T FREEZE!
    output=$(timeout 60 ./target/release/prism-cli \
        -i benchmarks/dimacs/DSJC125.5.col \
        --config configs/EXTREME_MAX.toml \
        --attempts 1 2>&1)

    if [ $? -eq 124 ]; then
        echo "TIMEOUT (60s)"
    else
        colors=$(echo "$output" | grep "Best chromatic" | tail -1 | awk '{print $5}')
        if [ -n "$colors" ]; then
            echo "$colors colors"
        else
            echo "Failed"
        fi
    fi
done