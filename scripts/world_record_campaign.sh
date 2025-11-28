#!/bin/bash
# PRISM World Record Hypertuning Campaign
# Automated parameter search to achieve 17 colors with zero conflicts
# Based on telemetry showing Phase 3 quantum achieves 17 colors with conflicts

set -e

# ============================================================================
# Configuration
# ============================================================================

CAMPAIGN_NAME="${1:-world_record_$(date +%Y%m%d_%H%M%S)}"
BASE_CONFIG="${2:-/mnt/c/Users/Predator/Desktop/PRISM/configs/WORLD_RECORD_ATTEMPT.toml}"
GRAPH_FILE="${3:-/mnt/c/Users/Predator/Desktop/PRISM/benchmarks/dimacs/DSJC125.5.col}"
TARGET_CHROMATIC=17
MAX_ITERATIONS=20
PRISM_ROOT="/mnt/c/Users/Predator/Desktop/PRISM"

# Campaign workspace
CAMPAIGN_DIR="$PRISM_ROOT/campaigns/$CAMPAIGN_NAME"
CONFIG_DIR="$CAMPAIGN_DIR/configs"
TELEMETRY_DIR="$CAMPAIGN_DIR/telemetry"
RESULTS_DIR="$CAMPAIGN_DIR/results"
STATE_FILE="$CAMPAIGN_DIR/state.json"

# Parameter sweep ranges (will be explored intelligently)
declare -a MU_VALUES=(0.50 0.55 0.60 0.65 0.70)
declare -a COUPLING_VALUES=(7.0 7.5 8.0 8.5 9.0)
declare -a EVOLUTION_TIME_VALUES=(0.12 0.14 0.16 0.18)
declare -a CONFLICT_REPAIR_INCREASE=(0 1 2)

# Best result tracking
BEST_CHROMATIC=999
BEST_CONFLICTS=999
BEST_CONFIG=""
BEST_ITERATION=-1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${CYAN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_section() {
    echo ""
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "${MAGENTA}$1${NC}"
    echo -e "${MAGENTA}========================================${NC}"
}

# Initialize campaign workspace
init_campaign() {
    log_section "Initializing Campaign: $CAMPAIGN_NAME"

    mkdir -p "$CONFIG_DIR" "$TELEMETRY_DIR" "$RESULTS_DIR"

    # Copy base config
    cp "$BASE_CONFIG" "$CONFIG_DIR/base.toml"

    # Initialize state
    cat > "$STATE_FILE" <<EOF
{
  "campaign_name": "$CAMPAIGN_NAME",
  "start_time": "$(date -Iseconds)",
  "target_chromatic": $TARGET_CHROMATIC,
  "graph_file": "$GRAPH_FILE",
  "iterations_completed": 0,
  "best_chromatic": 999,
  "best_conflicts": 999,
  "best_iteration": -1,
  "target_achieved": false
}
EOF

    log_success "Campaign workspace created at: $CAMPAIGN_DIR"
    log_info "Base config: $BASE_CONFIG"
    log_info "Target graph: $GRAPH_FILE"
    log_info "Target: $TARGET_CHROMATIC colors with 0 conflicts"
}

# Load campaign state (for resume capability)
load_state() {
    if [ -f "$STATE_FILE" ]; then
        BEST_CHROMATIC=$(jq -r '.best_chromatic' "$STATE_FILE")
        BEST_CONFLICTS=$(jq -r '.best_conflicts' "$STATE_FILE")
        BEST_ITERATION=$(jq -r '.best_iteration' "$STATE_FILE")
        ITERATIONS_COMPLETED=$(jq -r '.iterations_completed' "$STATE_FILE")

        log_info "Resuming campaign from iteration $ITERATIONS_COMPLETED"
        log_info "Current best: $BEST_CHROMATIC colors, $BEST_CONFLICTS conflicts"
    fi
}

# Save campaign state
save_state() {
    local iter=$1
    local chromatic=$2
    local conflicts=$3
    local target_achieved=$4

    cat > "$STATE_FILE" <<EOF
{
  "campaign_name": "$CAMPAIGN_NAME",
  "start_time": "$(jq -r '.start_time' "$STATE_FILE" 2>/dev/null || date -Iseconds)",
  "last_update": "$(date -Iseconds)",
  "target_chromatic": $TARGET_CHROMATIC,
  "graph_file": "$GRAPH_FILE",
  "iterations_completed": $iter,
  "best_chromatic": $BEST_CHROMATIC,
  "best_conflicts": $BEST_CONFLICTS,
  "best_iteration": $BEST_ITERATION,
  "best_config": "$BEST_CONFIG",
  "target_achieved": $target_achieved
}
EOF
}

# Generate config with specific parameters
generate_config() {
    local iter=$1
    local mu=$2
    local coupling=$3
    local evo_time=$4
    local repair_increase=$5
    local output_config="$CONFIG_DIR/iter_${iter}.toml"

    # Copy base config
    cp "$CONFIG_DIR/base.toml" "$output_config"

    # Update parameters using sed (more portable than toml CLI tools)
    # Note: Chemical potential μ is in the CUDA kernel, but we'll track it in config comments

    # Add parameter annotations at the top
    sed -i "1i# Campaign Iteration $iter" "$output_config"
    sed -i "2i# Chemical Potential (μ): $mu" "$output_config"
    sed -i "3i# Quantum Coupling Strength: $coupling" "$output_config"
    sed -i "4i# Quantum Evolution Time: $evo_time" "$output_config"
    sed -i "5i# Conflict Repair Max Increase: $repair_increase" "$output_config"
    sed -i "6i#" "$output_config"

    # Update quantum parameters
    sed -i "s/^coupling_strength = .*/coupling_strength = $coupling/" "$output_config"
    sed -i "s/^evolution_time = .*/evolution_time = $evo_time/" "$output_config"

    # Update conflict repair (if section exists)
    if grep -q "\[conflict_repair\]" "$output_config"; then
        sed -i "/\[conflict_repair\]/,/^\[/ s/^max_color_increase = .*/max_color_increase = $repair_increase/" "$output_config"
    fi

    # Increase quantum iterations for better convergence
    sed -i "s/^evolution_iterations = .*/evolution_iterations = 800/" "$output_config"

    echo "$output_config"
}

# Compile GPU kernel with specific chemical potential
compile_kernel_with_mu() {
    local mu=$1
    local kernel_file="$PRISM_ROOT/prism-gpu/src/kernels/thermodynamic.cu"
    local backup_file="$kernel_file.backup_campaign"

    # Backup original if not already backed up
    if [ ! -f "$backup_file" ]; then
        cp "$kernel_file" "$backup_file"
        log_info "Backed up original thermodynamic kernel"
    fi

    # Update chemical potential in kernel
    log_info "Setting chemical potential μ = $mu in GPU kernel"
    sed -i "s/const float chemical_potential = [0-9.]*f;/const float chemical_potential = ${mu}f;/" "$kernel_file"

    # Recompile GPU kernels
    log_info "Recompiling GPU kernels..."
    cd "$PRISM_ROOT/prism-gpu"

    if cargo build --release --features=cuda 2>&1 | tee "$CAMPAIGN_DIR/build_iter_${CURRENT_ITER}.log"; then
        log_success "GPU kernels compiled successfully with μ = $mu"
        cd "$PRISM_ROOT"
        return 0
    else
        log_error "GPU kernel compilation failed"
        cd "$PRISM_ROOT"
        return 1
    fi
}

# Run PRISM with specific configuration
run_prism() {
    local iter=$1
    local config=$2
    local telemetry_file="$TELEMETRY_DIR/iter_${iter}.jsonl"
    local log_file="$RESULTS_DIR/iter_${iter}.log"

    log_info "Running PRISM iteration $iter..."
    log_info "Config: $(basename $config)"

    cd "$PRISM_ROOT"

    # Run with timeout to prevent hanging (using prism-cli binary)
    timeout 60 ./target/release/prism-cli \
        --config "$config" \
        --input "$GRAPH_FILE" \
        --gpu \
        > "$log_file" 2>&1 || {
        log_warning "PRISM run timed out or failed (exit code: $?)"
        return 1
    }

    log_success "PRISM run completed"
    return 0
}

# Extract metrics from telemetry
extract_metrics() {
    local telemetry_file=$1

    if [ ! -f "$telemetry_file" ]; then
        echo "999 999 999.0"
        return
    fi

    # Extract final chromatic number (from last successful coloring phase)
    local chromatic=$(jq -s '
        map(select(.metrics.num_colors != null and .outcome == "Success"))
        | map(.metrics.num_colors)
        | min // 999
    ' "$telemetry_file")

    # Extract total conflicts (max across all phases)
    local conflicts=$(jq -s '
        map(select(.metrics.conflicts != null))
        | map(.metrics.conflicts)
        | max // 999
    ' "$telemetry_file")

    # Extract max geometric stress
    local stress=$(jq -s '
        map(select(.geometry.stress != null))
        | map(.geometry.stress)
        | max // 999.0
    ' "$telemetry_file")

    echo "$chromatic $conflicts $stress"
}

# Smart parameter selection using Bayesian-inspired heuristics
select_parameters() {
    local iter=$1

    # Stage 1: Grid search (iterations 1-8)
    if [ $iter -le 8 ]; then
        local mu_idx=$(( (iter - 1) % ${#MU_VALUES[@]} ))
        local coupling_idx=$(( (iter - 1) / ${#MU_VALUES[@]} % ${#COUPLING_VALUES[@]} ))

        MU=${MU_VALUES[$mu_idx]}
        COUPLING=${COUPLING_VALUES[$coupling_idx]}
        EVOLUTION_TIME=0.14  # Middle value
        REPAIR_INCREASE=1    # Middle value

    # Stage 2: Refinement around best (iterations 9-14)
    elif [ $iter -le 14 ]; then
        # Explore around best parameters found so far
        if [ "$BEST_CONFIG" != "" ]; then
            # Extract best parameters and perturb slightly
            MU=$(echo "$MU + (0.02 * ($RANDOM % 3 - 1))" | bc -l)
            COUPLING=$(echo "$COUPLING + (0.5 * ($RANDOM % 3 - 1))" | bc -l)
            EVOLUTION_TIME=$(echo "$EVOLUTION_TIME + (0.02 * ($RANDOM % 3 - 1))" | bc -l)
        else
            # Fallback to middle values
            MU=0.60
            COUPLING=8.0
            EVOLUTION_TIME=0.14
        fi
        REPAIR_INCREASE=$(( $RANDOM % 3 ))

    # Stage 3: Aggressive exploration (iterations 15+)
    else
        # Try extreme values if target not achieved
        local extreme_idx=$(( (iter - 15) % 4 ))
        case $extreme_idx in
            0) MU=0.50; COUPLING=9.0; EVOLUTION_TIME=0.18 ;;
            1) MU=0.70; COUPLING=7.0; EVOLUTION_TIME=0.12 ;;
            2) MU=0.60; COUPLING=8.5; EVOLUTION_TIME=0.16 ;;
            3) MU=0.55; COUPLING=7.5; EVOLUTION_TIME=0.14 ;;
        esac
        REPAIR_INCREASE=$(( $RANDOM % 3 ))
    fi

    # Clamp values to valid ranges
    MU=$(echo "$MU" | awk '{if ($1 < 0.5) print 0.5; else if ($1 > 0.7) print 0.7; else print $1}')
    COUPLING=$(echo "$COUPLING" | awk '{if ($1 < 7.0) print 7.0; else if ($1 > 9.0) print 9.0; else print $1}')
    EVOLUTION_TIME=$(echo "$EVOLUTION_TIME" | awk '{if ($1 < 0.12) print 0.12; else if ($1 > 0.18) print 0.18; else print $1}')
}

# Display iteration summary
display_iteration_summary() {
    local iter=$1
    local chromatic=$2
    local conflicts=$3
    local stress=$4
    local mu=$5
    local coupling=$6
    local evo_time=$7

    echo ""
    log_section "Iteration $iter Results"
    echo -e "Parameters:"
    echo -e "  Chemical Potential (μ):  ${BLUE}$mu${NC}"
    echo -e "  Coupling Strength:       ${BLUE}$coupling${NC}"
    echo -e "  Evolution Time:          ${BLUE}$evo_time${NC}"
    echo ""
    echo -e "Results:"
    if [ "$chromatic" == "999" ]; then
        echo -e "  Chromatic Number:        ${RED}FAILED${NC}"
    elif [ "$chromatic" -le "$TARGET_CHROMATIC" ]; then
        echo -e "  Chromatic Number:        ${GREEN}$chromatic${NC} ⭐"
    else
        echo -e "  Chromatic Number:        ${YELLOW}$chromatic${NC}"
    fi

    if [ "$conflicts" == "999" ]; then
        echo -e "  Conflicts:               ${RED}FAILED${NC}"
    elif [ "$conflicts" == "0" ]; then
        echo -e "  Conflicts:               ${GREEN}$conflicts${NC} ✓"
    else
        echo -e "  Conflicts:               ${YELLOW}$conflicts${NC}"
    fi

    echo -e "  Geometric Stress:        ${BLUE}$(printf '%.2f' $stress)${NC}"
    echo ""

    # Check if this is a new best
    if [ "$chromatic" != "999" ]; then
        if [ "$chromatic" -lt "$BEST_CHROMATIC" ] ||
           ([ "$chromatic" -eq "$BEST_CHROMATIC" ] && [ "$conflicts" -lt "$BEST_CONFLICTS" ]); then
            log_success "NEW BEST RESULT!"
            BEST_CHROMATIC=$chromatic
            BEST_CONFLICTS=$conflicts
            BEST_CONFIG="$CONFIG_DIR/iter_${iter}.toml"
            BEST_ITERATION=$iter

            # Save best config
            cp "$BEST_CONFIG" "$CAMPAIGN_DIR/CHAMPION.toml"
        fi
    fi

    echo -e "Current Best: ${GREEN}$BEST_CHROMATIC colors${NC}, ${GREEN}$BEST_CONFLICTS conflicts${NC} (iter $BEST_ITERATION)"

    # Check if target achieved
    if [ "$chromatic" -le "$TARGET_CHROMATIC" ] && [ "$conflicts" == "0" ]; then
        echo ""
        log_success "🏆 TARGET ACHIEVED! 🏆"
        log_success "Configuration: $CONFIG_DIR/iter_${iter}.toml"
        return 0
    fi

    return 1
}

# Generate campaign summary report
generate_summary() {
    local summary_file="$CAMPAIGN_DIR/CAMPAIGN_SUMMARY.md"

    log_info "Generating campaign summary report..."

    cat > "$summary_file" <<EOF
# PRISM World Record Campaign Summary

**Campaign Name:** $CAMPAIGN_NAME
**Date:** $(date)
**Target:** $TARGET_CHROMATIC colors with 0 conflicts

## Results

EOF

    if [ "$BEST_CHROMATIC" -le "$TARGET_CHROMATIC" ] && [ "$BEST_CONFLICTS" == "0" ]; then
        cat >> "$summary_file" <<EOF
### 🏆 TARGET ACHIEVED! 🏆

- **Best Chromatic Number:** $BEST_CHROMATIC
- **Conflicts:** $BEST_CONFLICTS
- **Achieved at Iteration:** $BEST_ITERATION
- **Champion Configuration:** \`CHAMPION.toml\`

EOF
    else
        cat >> "$summary_file" <<EOF
### Best Result (Target not fully achieved)

- **Best Chromatic Number:** $BEST_CHROMATIC (target: $TARGET_CHROMATIC)
- **Best Conflicts:** $BEST_CONFLICTS (target: 0)
- **Best Iteration:** $BEST_ITERATION
- **Best Configuration:** \`$(basename "$BEST_CONFIG")\`

EOF
    fi

    cat >> "$summary_file" <<EOF
## Campaign Statistics

- **Total Iterations:** $(jq -r '.iterations_completed' "$STATE_FILE")
- **Graph:** $(basename "$GRAPH_FILE")
- **Duration:** $(date -d@$(( $(date +%s) - $(date -d "$(jq -r '.start_time' "$STATE_FILE")" +%s) )) -u +%H:%M:%S)

## Parameter Exploration

### Chemical Potential (μ) Range
- Min: ${MU_VALUES[0]}
- Max: ${MU_VALUES[-1]}

### Quantum Coupling Strength Range
- Min: ${COUPLING_VALUES[0]}
- Max: ${COUPLING_VALUES[-1]}

### Evolution Time Range
- Min: ${EVOLUTION_TIME_VALUES[0]}
- Max: ${EVOLUTION_TIME_VALUES[-1]}

## Iteration Results

| Iter | Chromatic | Conflicts | Stress | μ | Coupling | Evo Time |
|------|-----------|-----------|--------|---|----------|----------|
EOF

    # Add iteration results
    for i in $(seq 1 $(jq -r '.iterations_completed' "$STATE_FILE")); do
        telemetry="$TELEMETRY_DIR/iter_${i}.jsonl"
        if [ -f "$telemetry" ]; then
            read chromatic conflicts stress <<< $(extract_metrics "$telemetry")
            config="$CONFIG_DIR/iter_${i}.toml"
            mu=$(grep "# Chemical Potential" "$config" | awk '{print $5}')
            coupling=$(grep "^coupling_strength" "$config" | awk '{print $3}')
            evo_time=$(grep "^evolution_time" "$config" | awk '{print $3}')

            echo "| $i | $chromatic | $conflicts | $(printf '%.2f' $stress) | $mu | $coupling | $evo_time |" >> "$summary_file"
        fi
    done

    cat >> "$summary_file" <<EOF

## Files

- **Campaign Directory:** \`$CAMPAIGN_DIR\`
- **Configurations:** \`$CONFIG_DIR/\`
- **Telemetry:** \`$TELEMETRY_DIR/\`
- **Logs:** \`$RESULTS_DIR/\`
- **State:** \`state.json\`

## Next Steps

EOF

    if [ "$BEST_CHROMATIC" -le "$TARGET_CHROMATIC" ] && [ "$BEST_CONFLICTS" == "0" ]; then
        cat >> "$summary_file" <<EOF
The target has been achieved! To reproduce:

\`\`\`bash
cargo run --release --features=cuda -- \\
    --config $CAMPAIGN_DIR/CHAMPION.toml \\
    --graph $GRAPH_FILE \\
    --telemetry verification.jsonl
\`\`\`
EOF
    else
        cat >> "$summary_file" <<EOF
Target not fully achieved. Recommendations:

1. Run additional campaign iterations with refined parameter ranges
2. Analyze telemetry patterns in successful vs. failed runs
3. Consider adjusting memetic algorithm parameters
4. Increase ensemble diversity settings
5. Try longer quantum evolution iterations

To resume this campaign:

\`\`\`bash
./scripts/world_record_campaign.sh $CAMPAIGN_NAME
\`\`\`
EOF
    fi

    log_success "Campaign summary saved to: $summary_file"
}

# ============================================================================
# Main Campaign Loop
# ============================================================================

main() {
    log_section "PRISM World Record Hypertuning Campaign"

    # Check prerequisites
    if [ ! -f "$BASE_CONFIG" ]; then
        log_error "Base configuration not found: $BASE_CONFIG"
        exit 1
    fi

    if [ ! -f "$GRAPH_FILE" ]; then
        log_error "Graph file not found: $GRAPH_FILE"
        exit 1
    fi

    # Initialize or resume campaign
    if [ ! -d "$CAMPAIGN_DIR" ]; then
        init_campaign
    else
        log_warning "Campaign directory exists, resuming..."
        load_state
    fi

    # Main iteration loop
    for CURRENT_ITER in $(seq 1 $MAX_ITERATIONS); do
        log_section "Iteration $CURRENT_ITER / $MAX_ITERATIONS"

        # Select parameters for this iteration
        select_parameters $CURRENT_ITER

        log_info "Selected parameters:"
        log_info "  μ (chemical potential): $MU"
        log_info "  Coupling strength: $COUPLING"
        log_info "  Evolution time: $EVOLUTION_TIME"
        log_info "  Conflict repair increase: $REPAIR_INCREASE"

        # Compile kernel with new μ value
        if ! compile_kernel_with_mu "$MU"; then
            log_error "Failed to compile kernel, skipping iteration"
            continue
        fi

        # Generate configuration
        CONFIG=$(generate_config $CURRENT_ITER $MU $COUPLING $EVOLUTION_TIME $REPAIR_INCREASE)
        log_success "Generated config: $(basename $CONFIG)"

        # Run PRISM
        if ! run_prism $CURRENT_ITER "$CONFIG"; then
            log_warning "PRISM run failed, continuing to next iteration"
            save_state $CURRENT_ITER 999 999 false
            continue
        fi

        # Extract and display results
        read CHROMATIC CONFLICTS STRESS <<< $(extract_metrics "$TELEMETRY_DIR/iter_${CURRENT_ITER}.jsonl")

        if display_iteration_summary $CURRENT_ITER $CHROMATIC $CONFLICTS $STRESS $MU $COUPLING $EVOLUTION_TIME; then
            # Target achieved!
            save_state $CURRENT_ITER $CHROMATIC $CONFLICTS true
            generate_summary
            log_success "Campaign completed successfully!"
            exit 0
        fi

        # Save state
        save_state $CURRENT_ITER $CHROMATIC $CONFLICTS false

        # Brief pause between iterations
        sleep 2
    done

    # Campaign completed without achieving target
    log_section "Campaign Completed"
    log_info "Maximum iterations reached without achieving target"
    log_info "Best result: $BEST_CHROMATIC colors, $BEST_CONFLICTS conflicts"

    generate_summary
}

# ============================================================================
# Entry Point
# ============================================================================

# Handle Ctrl+C gracefully
trap 'log_warning "Campaign interrupted by user"; generate_summary; exit 130' INT TERM

# Run main function
main

exit 0
