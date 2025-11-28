#!/bin/bash
# PRISM Campaign Results Analysis
# Deep analysis and visualization of campaign results

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

CAMPAIGN_DIR="${1:-$(ls -td /mnt/c/Users/Predator/Desktop/PRISM/campaigns/* 2>/dev/null | head -1)}"

if [ -z "$CAMPAIGN_DIR" ] || [ ! -d "$CAMPAIGN_DIR" ]; then
    echo "Usage: $0 <campaign_directory>"
    echo ""
    echo "Available campaigns:"
    ls -td /mnt/c/Users/Predator/Desktop/PRISM/campaigns/* 2>/dev/null | head -5
    exit 1
fi

STATE_FILE="$CAMPAIGN_DIR/state.json"
TELEMETRY_DIR="$CAMPAIGN_DIR/telemetry"
CONFIG_DIR="$CAMPAIGN_DIR/configs"
ANALYSIS_DIR="$CAMPAIGN_DIR/analysis"

# Create analysis directory
mkdir -p "$ANALYSIS_DIR"

log_section() {
    echo ""
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "${MAGENTA}$1${NC}"
    echo -e "${MAGENTA}========================================${NC}"
}

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Extract comprehensive metrics from telemetry
extract_all_metrics() {
    local telemetry_file=$1

    if [ ! -f "$telemetry_file" ]; then
        echo "{}"
        return
    fi

    jq -s '
    {
        chromatic: (map(select(.metrics.num_colors != null and .outcome == "Success")) | map(.metrics.num_colors) | min),
        conflicts: (map(select(.metrics.conflicts != null)) | map(.metrics.conflicts) | max),
        max_stress: (map(select(.geometry.stress != null)) | map(.geometry.stress) | max),
        avg_stress: (map(select(.geometry.stress != null)) | map(.geometry.stress) | add / length),
        total_time_ms: (map(select(.metrics.execution_time_ms != null)) | map(.metrics.execution_time_ms) | add),
        phases_completed: length,
        phase_breakdown: group_by(.phase) | map({phase: .[0].phase, count: length, avg_time: (map(.metrics.execution_time_ms // 0) | add / length)}),
        quantum_metrics: (map(select(.phase | startswith("Phase3"))) | {
            purity: (map(.metrics.purity // null) | map(select(. != null)) | add / length),
            num_colors: (map(.metrics.num_colors // null) | map(select(. != null)) | min),
            conflicts: (map(.metrics.conflicts // null) | map(select(. != null)) | max)
        }),
        thermodynamic_metrics: (map(select(.phase | startswith("Phase2"))) | {
            guard_triggers: (map(.metrics.guard_triggers // null) | map(select(. != null)) | max),
            compaction_ratio: (map(.metrics.compaction_ratio // null) | map(select(. != null)) | add / length),
            acceptance_rate: (map(.metrics.acceptance_rate // null) | map(select(. != null)) | add / length)
        })
    }
    ' "$telemetry_file"
}

# Generate CSV data for plotting
generate_csv() {
    local output_csv="$ANALYSIS_DIR/campaign_results.csv"

    log_info "Generating CSV data..."

    # Header
    echo "iteration,chromatic,conflicts,max_stress,avg_stress,total_time_ms,mu,coupling,evolution_time,repair_increase,quantum_purity,thermo_guard_triggers,phases_completed" > "$output_csv"

    # Read state
    local iterations=$(jq -r '.iterations_completed' "$STATE_FILE")

    for i in $(seq 1 $iterations); do
        local telemetry="$TELEMETRY_DIR/iter_${i}.jsonl"
        local config="$CONFIG_DIR/iter_${i}.toml"

        if [ ! -f "$telemetry" ]; then
            continue
        fi

        # Extract metrics
        local metrics=$(extract_all_metrics "$telemetry")

        local chromatic=$(echo "$metrics" | jq -r '.chromatic // 999')
        local conflicts=$(echo "$metrics" | jq -r '.conflicts // 999')
        local max_stress=$(echo "$metrics" | jq -r '.max_stress // 999.0')
        local avg_stress=$(echo "$metrics" | jq -r '.avg_stress // 999.0')
        local total_time=$(echo "$metrics" | jq -r '.total_time_ms // 0')
        local phases=$(echo "$metrics" | jq -r '.phases_completed // 0')
        local quantum_purity=$(echo "$metrics" | jq -r '.quantum_metrics.purity // 0.0')
        local guard_triggers=$(echo "$metrics" | jq -r '.thermodynamic_metrics.guard_triggers // 0')

        # Extract parameters from config
        local mu=$(grep "# Chemical Potential" "$config" | awk '{print $5}' | head -1)
        local coupling=$(grep "^coupling_strength" "$config" | awk '{print $3}' | head -1)
        local evo_time=$(grep "^evolution_time" "$config" | awk '{print $3}' | head -1)
        local repair=$(grep "# Conflict Repair Max Increase" "$config" | awk '{print $6}' | head -1)

        [ -z "$mu" ] && mu="0.0"
        [ -z "$coupling" ] && coupling="0.0"
        [ -z "$evo_time" ] && evo_time="0.0"
        [ -z "$repair" ] && repair="0"

        echo "$i,$chromatic,$conflicts,$max_stress,$avg_stress,$total_time,$mu,$coupling,$evo_time,$repair,$quantum_purity,$guard_triggers,$phases" >> "$output_csv"
    done

    log_success "CSV data saved to: $output_csv"
}

# Statistical analysis
perform_statistical_analysis() {
    log_section "Statistical Analysis"

    local csv="$ANALYSIS_DIR/campaign_results.csv"
    local report="$ANALYSIS_DIR/statistical_report.txt"

    if [ ! -f "$csv" ]; then
        log_info "CSV file not found, generating..."
        generate_csv
    fi

    {
        echo "PRISM Campaign Statistical Analysis"
        echo "===================================="
        echo ""
        echo "Campaign: $(basename $CAMPAIGN_DIR)"
        echo "Date: $(date)"
        echo ""

        # Overall statistics
        echo "OVERALL STATISTICS"
        echo "------------------"

        # Chromatic number stats
        echo ""
        echo "Chromatic Number:"
        awk -F',' 'NR>1 && $2<999 {sum+=$2; count++; if(min==""){min=max=$2}; if($2<min){min=$2}; if($2>max){max=$2}} END {if(count>0) printf "  Min: %d\n  Max: %d\n  Mean: %.2f\n  Count: %d\n", min, max, sum/count, count}' "$csv"

        # Conflict stats
        echo ""
        echo "Conflicts:"
        awk -F',' 'NR>1 && $3<999 {sum+=$3; count++; if(min==""){min=max=$3}; if($3<min){min=$3}; if($3>max){max=$3}; if($3==0){zero++}} END {if(count>0) printf "  Min: %d\n  Max: %d\n  Mean: %.2f\n  Zero-conflict runs: %d (%.1f%%)\n", min, max, sum/count, zero, 100*zero/count}' "$csv"

        # Stress stats
        echo ""
        echo "Geometric Stress:"
        awk -F',' 'NR>1 && $4<999 {sum+=$4; count++; if(min==""){min=max=$4}; if($4<min){min=$4}; if($4>max){max=$4}} END {if(count>0) printf "  Min: %.2f\n  Max: %.2f\n  Mean: %.2f\n", min, max, sum/count}' "$csv"

        # Runtime stats
        echo ""
        echo "Execution Time (ms):"
        awk -F',' 'NR>1 && $6>0 {sum+=$6; count++; if(min==""){min=max=$6}; if($6<min){min=$6}; if($6>max){max=$6}} END {if(count>0) printf "  Min: %.0f\n  Max: %.0f\n  Mean: %.0f\n  Total: %.0f\n", min, max, sum/count, sum}' "$csv"

        # Parameter correlation analysis
        echo ""
        echo ""
        echo "PARAMETER ANALYSIS"
        echo "------------------"

        # Best performing parameter combinations
        echo ""
        echo "Top 5 Configurations (by chromatic number + conflicts):"
        awk -F',' 'NR>1 && $2<999 {score=$2*100+$3; print score,$0}' "$csv" | sort -n | head -5 | nl | while read num score rest; do
            echo "  $num. Score: $score | $(echo $rest | awk -F',' '{printf "Iter %s: %s colors, %s conflicts (μ=%s, coupling=%s, evo_time=%s)", $1, $2, $3, $7, $8, $9}')"
        done

        # Chemical potential analysis
        echo ""
        echo "Chemical Potential (μ) Impact:"
        awk -F',' 'NR>1 && $2<999 && $7>0 {mu=$7; colors[mu]+=$2; conflicts[mu]+=$3; count[mu]++} END {for(mu in count) printf "  μ=%.2f: Avg Colors=%.1f, Avg Conflicts=%.1f (n=%d)\n", mu, colors[mu]/count[mu], conflicts[mu]/count[mu], count[mu]}' "$csv" | sort -n

        # Coupling strength analysis
        echo ""
        echo "Quantum Coupling Strength Impact:"
        awk -F',' 'NR>1 && $2<999 && $8>0 {coupling=$8; colors[coupling]+=$2; conflicts[coupling]+=$3; count[coupling]++} END {for(c in count) printf "  Coupling=%.1f: Avg Colors=%.1f, Avg Conflicts=%.1f (n=%d)\n", c, colors[c]/count[c], conflicts[c]/count[c], count[c]}' "$csv" | sort -n

        # Success rate by parameter ranges
        echo ""
        echo ""
        echo "SUCCESS METRICS"
        echo "---------------"

        # Target achievement
        local target=$(jq -r '.target_chromatic' "$STATE_FILE")
        echo ""
        echo "Runs achieving target ($target colors):"
        local target_runs=$(awk -F',' -v target="$target" 'NR>1 && $2<=target {count++} END {print count+0}' "$csv")
        local total_runs=$(awk 'NR>1' "$csv" | wc -l)
        echo "  Count: $target_runs / $total_runs ($(echo "scale=1; 100*$target_runs/$total_runs" | bc)%)"

        echo ""
        echo "Zero-conflict runs:"
        local zero_conflict=$(awk -F',' 'NR>1 && $3==0 {count++} END {print count+0}' "$csv")
        echo "  Count: $zero_conflict / $total_runs ($(echo "scale=1; 100*$zero_conflict/$total_runs" | bc)%)"

        echo ""
        echo "Perfect runs ($target colors + 0 conflicts):"
        local perfect=$(awk -F',' -v target="$target" 'NR>1 && $2<=target && $3==0 {count++} END {print count+0}' "$csv")
        echo "  Count: $perfect / $total_runs ($(echo "scale=1; 100*$perfect/$total_runs" | bc)%)"

    } | tee "$report"

    log_success "Statistical report saved to: $report"
}

# Phase performance analysis
analyze_phase_performance() {
    log_section "Phase Performance Analysis"

    local report="$ANALYSIS_DIR/phase_analysis.txt"

    {
        echo "Phase-by-Phase Performance Analysis"
        echo "===================================="
        echo ""

        local iterations=$(jq -r '.iterations_completed' "$STATE_FILE")

        # Aggregate phase metrics across all iterations
        declare -A phase_times
        declare -A phase_counts
        declare -A phase_colors
        declare -A phase_conflicts

        for i in $(seq 1 $iterations); do
            local telemetry="$TELEMETRY_DIR/iter_${i}.jsonl"
            [ ! -f "$telemetry" ] && continue

            # Extract phase-specific metrics
            jq -r '.phase + " " + (.metrics.execution_time_ms // 0 | tostring) + " " + (.metrics.num_colors // 999 | tostring) + " " + (.metrics.conflicts // 999 | tostring)' "$telemetry" | while read phase time colors conflicts; do
                phase_times[$phase]=$((${phase_times[$phase]:-0} + ${time%.*}))
                phase_counts[$phase]=$((${phase_counts[$phase]:-0} + 1))

                if [ "$colors" != "999" ]; then
                    phase_colors[$phase]=$((${phase_colors[$phase]:-0} + colors))
                fi

                if [ "$conflicts" != "999" ]; then
                    phase_conflicts[$phase]=$((${phase_conflicts[$phase]:-0} + conflicts))
                fi
            done
        done

        # Phase 2 (Thermodynamic) deep dive
        echo "Phase 2 - Thermodynamic Annealing"
        echo "----------------------------------"
        echo ""
        echo "Key Metrics Across All Iterations:"

        jq -s '
        map(select(.phase | startswith("Phase2"))) |
        {
            avg_guard_triggers: (map(.metrics.guard_triggers // 0) | add / length),
            max_guard_triggers: (map(.metrics.guard_triggers // 0) | max),
            avg_compaction_ratio: (map(.metrics.compaction_ratio // 0) | add / length),
            avg_acceptance_rate: (map(.metrics.acceptance_rate // 0) | add / length),
            avg_colors: (map(.metrics.num_colors // 999) | map(select(. < 999)) | add / length),
            min_colors: (map(.metrics.num_colors // 999) | min)
        }
        ' "$TELEMETRY_DIR"/*.jsonl 2>/dev/null | jq -r '
        "  Average Guard Triggers: \(.avg_guard_triggers | . * 10 | round / 10)",
        "  Max Guard Triggers: \(.max_guard_triggers)",
        "  Average Compaction Ratio: \(.avg_compaction_ratio | . * 100 | round / 100)",
        "  Average Acceptance Rate: \(.avg_acceptance_rate | . * 100 | round / 100)",
        "  Average Colors: \(.avg_colors | . * 10 | round / 10)",
        "  Best Colors: \(.min_colors)"
        '

        echo ""
        echo ""

        # Phase 3 (Quantum) deep dive
        echo "Phase 3 - Quantum Evolution"
        echo "---------------------------"
        echo ""
        echo "Key Metrics Across All Iterations:"

        jq -s '
        map(select(.phase | startswith("Phase3"))) |
        {
            avg_purity: (map(.metrics.purity // 0) | add / length),
            min_purity: (map(.metrics.purity // 0) | min),
            avg_colors: (map(.metrics.num_colors // 999) | map(select(. < 999)) | add / length),
            min_colors: (map(.metrics.num_colors // 999) | min),
            avg_conflicts: (map(.metrics.conflicts // 0) | add / length),
            success_rate: (map(select(.outcome == "Success")) | length) / length
        }
        ' "$TELEMETRY_DIR"/*.jsonl 2>/dev/null | jq -r '
        "  Average Purity: \(.avg_purity | . * 1000 | round / 1000)",
        "  Min Purity: \(.min_purity | . * 1000 | round / 1000)",
        "  Average Colors: \(.avg_colors | . * 10 | round / 10)",
        "  Best Colors: \(.min_colors)",
        "  Average Conflicts: \(.avg_conflicts | . * 10 | round / 10)",
        "  Success Rate: \(.success_rate | . * 100 | round)%"
        '

        echo ""
        echo ""

        # Conflict repair analysis
        echo "Conflict Repair Effectiveness"
        echo "-----------------------------"
        echo ""

        for i in $(seq 1 $iterations); do
            local telemetry="$TELEMETRY_DIR/iter_${i}.jsonl"
            [ ! -f "$telemetry" ] && continue

            # Get colors before and after repair
            local before_colors=$(jq -s 'map(select(.phase | startswith("Phase3") and .outcome == "Success")) | map(.metrics.num_colors) | min // 999' "$telemetry")
            local after_colors=$(jq -s 'map(select(.phase == "ConflictRepair" and .outcome == "Success")) | map(.metrics.num_colors) | min // 999' "$telemetry")

            if [ "$before_colors" != "999" ] && [ "$after_colors" != "999" ]; then
                local before_conflicts=$(jq -s 'map(select(.phase | startswith("Phase3") and .outcome == "Success")) | map(.metrics.conflicts) | max // 0' "$telemetry")
                local after_conflicts=$(jq -s 'map(select(.phase == "ConflictRepair" and .outcome == "Success")) | map(.metrics.conflicts) | max // 0' "$telemetry")

                echo "  Iteration $i:"
                echo "    Before: $before_colors colors, $before_conflicts conflicts"
                echo "    After:  $after_colors colors, $after_conflicts conflicts"
                echo "    Change: +$((after_colors - before_colors)) colors, $((after_conflicts - before_conflicts)) conflicts"
                echo ""
            fi
        done

    } | tee "$report"

    log_success "Phase analysis saved to: $report"
}

# Generate recommendations
generate_recommendations() {
    log_section "Generating Recommendations"

    local report="$ANALYSIS_DIR/recommendations.txt"
    local csv="$ANALYSIS_DIR/campaign_results.csv"

    {
        echo "PRISM Hypertuning Recommendations"
        echo "=================================="
        echo ""
        echo "Based on campaign: $(basename $CAMPAIGN_DIR)"
        echo "Date: $(date)"
        echo ""

        # Find best parameters
        local best_line=$(awk -F',' 'NR>1 && $2<999 {score=$2*100+$3; print score,$0}' "$csv" | sort -n | head -1)
        local best_mu=$(echo "$best_line" | awk -F',' '{print $8}')
        local best_coupling=$(echo "$best_line" | awk -F',' '{print $9}')
        local best_evo_time=$(echo "$best_line" | awk -F',' '{print $10}')

        echo "OPTIMAL PARAMETER RANGES"
        echo "------------------------"
        echo ""
        echo "Based on best-performing runs:"
        echo "  Chemical Potential (μ): $best_mu (±0.05)"
        echo "  Coupling Strength: $best_coupling (±0.5)"
        echo "  Evolution Time: $best_evo_time (±0.02)"
        echo ""

        # Identify problems
        echo ""
        echo "IDENTIFIED ISSUES"
        echo "-----------------"
        echo ""

        # Check for high conflict rates
        local high_conflict_rate=$(awk -F',' 'NR>1 && $3>50 {count++} END {print count+0}' "$csv")
        local total_runs=$(awk 'NR>1' "$csv" | wc -l)

        if [ "$high_conflict_rate" -gt $((total_runs / 3)) ]; then
            echo "⚠ High Conflict Rate Detected ($high_conflict_rate / $total_runs runs)"
            echo "  Recommendation:"
            echo "  - Reduce chemical potential μ by 0.05-0.10"
            echo "  - Increase thermodynamic temperature range"
            echo "  - Increase quantum evolution iterations"
            echo ""
        fi

        # Check for high stress
        local high_stress=$(awk -F',' 'NR>1 && $4>5.0 {count++} END {print count+0}' "$csv")
        if [ "$high_stress" -gt $((total_runs / 4)) ]; then
            echo "⚠ High Geometric Stress Detected ($high_stress / $total_runs runs)"
            echo "  Recommendation:"
            echo "  - Adjust metaphysical coupling feedback_strength"
            echo "  - Increase ensemble diversity_weight"
            echo "  - Review warmstart ratio balance"
            echo ""
        fi

        # Check for suboptimal chromatic numbers
        local target=$(jq -r '.target_chromatic' "$STATE_FILE")
        local above_target=$(awk -F',' -v target="$target" 'NR>1 && $2>target && $2<999 {count++} END {print count+0}' "$csv")

        if [ "$above_target" -gt $((total_runs / 2)) ]; then
            echo "⚠ Chromatic Number Often Above Target ($above_target / $total_runs runs)"
            echo "  Recommendation:"
            echo "  - Increase chemical potential μ (if conflicts are low)"
            echo "  - Increase quantum coupling strength"
            echo "  - Increase memetic population size and generations"
            echo "  - Extend quantum evolution time"
            echo ""
        fi

        echo ""
        echo "NEXT STEPS"
        echo "----------"
        echo ""

        if [ "$(jq -r '.target_achieved' "$STATE_FILE")" == "true" ]; then
            echo "✓ Target achieved! Recommended actions:"
            echo ""
            echo "1. Verify result with multiple runs using champion config"
            echo "2. Test on other DIMACS graphs (DSJC250.5, DSJC1000.5)"
            echo "3. Consider submitting results to graph coloring benchmarks"
            echo "4. Document parameter configuration in paper/report"
        else
            echo "Target not yet achieved. Recommended actions:"
            echo ""
            echo "1. Run extended campaign (40-50 iterations)"
            echo "2. Focus parameter search around: μ=$best_mu, coupling=$best_coupling"
            echo "3. Enable more aggressive conflict repair (max_color_increase=2)"
            echo "4. Increase ensemble replicas to 64 or 128"
            echo "5. Consider multi-fidelity approach: start with smaller graphs"
        fi

        echo ""
        echo ""
        echo "SUGGESTED CONFIGURATION"
        echo "-----------------------"
        echo ""
        echo "Create configs/REFINED_ATTEMPT.toml with:"
        echo ""
        echo "[phase2_thermodynamic]"
        echo "initial_temperature = $(awk -F',' -v mu="$best_mu" 'NR>1 && $7==mu {print 4.0; exit}' "$csv")"
        echo "cooling_rate = 0.92"
        echo ""
        echo "[phase3_quantum]"
        echo "coupling_strength = $best_coupling"
        echo "evolution_time = $best_evo_time"
        echo "evolution_iterations = 800"
        echo ""
        echo "# NOTE: Set μ=$best_mu in prism-gpu/src/kernels/thermodynamic.cu"

    } | tee "$report"

    log_success "Recommendations saved to: $report"
}

# Create visualization script (Python)
create_visualization_script() {
    log_section "Creating Visualization Script"

    local plot_script="$ANALYSIS_DIR/plot_results.py"

    cat > "$plot_script" <<'EOF'
#!/usr/bin/env python3
"""
PRISM Campaign Results Visualization
Generates plots for parameter exploration and convergence analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data(csv_path):
    """Load campaign results CSV"""
    return pd.read_csv(csv_path)

def plot_convergence(df, output_dir):
    """Plot convergence over iterations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PRISM Campaign Convergence Analysis', fontsize=16)

    # Chromatic number over time
    ax = axes[0, 0]
    ax.plot(df['iteration'], df['chromatic'], 'o-', linewidth=2, markersize=6)
    ax.axhline(y=17, color='r', linestyle='--', label='Target')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Chromatic Number')
    ax.set_title('Chromatic Number Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Conflicts over time
    ax = axes[0, 1]
    ax.plot(df['iteration'], df['conflicts'], 'o-', color='orange', linewidth=2, markersize=6)
    ax.axhline(y=0, color='g', linestyle='--', label='Target')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Conflicts')
    ax.set_title('Conflicts Over Iterations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Geometric stress over time
    ax = axes[1, 0]
    ax.plot(df['iteration'], df['max_stress'], 'o-', color='purple', linewidth=2, markersize=6)
    ax.axhline(y=1.0, color='r', linestyle='--', label='Threshold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Geometric Stress')
    ax.set_title('Geometric Stress Over Iterations')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Execution time over time
    ax = axes[1, 1]
    ax.plot(df['iteration'], df['total_time_ms'] / 1000, 'o-', color='green', linewidth=2, markersize=6)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Runtime Over Iterations')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'convergence.png')}")

def plot_parameter_exploration(df, output_dir):
    """Plot parameter space exploration"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Parameter Space Exploration', fontsize=16)

    # Filter valid results
    df_valid = df[df['chromatic'] < 999]

    # Chemical potential vs chromatic
    ax = axes[0, 0]
    scatter = ax.scatter(df_valid['mu'], df_valid['chromatic'],
                        c=df_valid['conflicts'], cmap='RdYlGn_r',
                        s=100, alpha=0.6, edgecolors='black')
    ax.axhline(y=17, color='r', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Chemical Potential (μ)')
    ax.set_ylabel('Chromatic Number')
    ax.set_title('μ vs Chromatic Number (colored by conflicts)')
    plt.colorbar(scatter, ax=ax, label='Conflicts')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Coupling strength vs chromatic
    ax = axes[0, 1]
    scatter = ax.scatter(df_valid['coupling'], df_valid['chromatic'],
                        c=df_valid['conflicts'], cmap='RdYlGn_r',
                        s=100, alpha=0.6, edgecolors='black')
    ax.axhline(y=17, color='r', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Quantum Coupling Strength')
    ax.set_ylabel('Chromatic Number')
    ax.set_title('Coupling vs Chromatic Number')
    plt.colorbar(scatter, ax=ax, label='Conflicts')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2D heatmap: mu vs coupling
    ax = axes[1, 0]
    pivot = df_valid.pivot_table(values='chromatic', index='coupling', columns='mu', aggfunc='min')
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Chromatic'})
    ax.set_title('Parameter Heatmap: Best Chromatic Number')
    ax.set_xlabel('Chemical Potential (μ)')
    ax.set_ylabel('Coupling Strength')

    # Evolution time impact
    ax = axes[1, 1]
    scatter = ax.scatter(df_valid['evolution_time'], df_valid['chromatic'],
                        c=df_valid['mu'], cmap='viridis',
                        s=100, alpha=0.6, edgecolors='black')
    ax.axhline(y=17, color='r', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Quantum Evolution Time')
    ax.set_ylabel('Chromatic Number')
    ax.set_title('Evolution Time vs Chromatic (colored by μ)')
    plt.colorbar(scatter, ax=ax, label='μ')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_exploration.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'parameter_exploration.png')}")

def plot_phase_analysis(df, output_dir):
    """Plot phase-specific metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Phase Performance Analysis', fontsize=16)

    # Quantum purity over iterations
    ax = axes[0]
    ax.plot(df['iteration'], df['quantum_purity'], 'o-', color='blue', linewidth=2, markersize=6)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Quantum Purity')
    ax.set_title('Quantum Phase - Purity Over Iterations')
    ax.grid(True, alpha=0.3)

    # Thermodynamic guard triggers
    ax = axes[1]
    ax.plot(df['iteration'], df['thermo_guard_triggers'], 'o-', color='red', linewidth=2, markersize=6)
    ax.axhline(y=100, color='orange', linestyle='--', label='Instability Threshold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Guard Triggers')
    ax.set_title('Thermodynamic Phase - Conflict Escalations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'phase_analysis.png')}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <campaign_results.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = os.path.dirname(csv_path)

    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)

    print(f"Loaded {len(df)} iterations")
    print(f"Best chromatic: {df['chromatic'].min()}")
    print(f"Best conflicts: {df[df['chromatic'] < 999]['conflicts'].min()}")

    print("\nGenerating plots...")
    plot_convergence(df, output_dir)
    plot_parameter_exploration(df, output_dir)
    plot_phase_analysis(df, output_dir)

    print("\nVisualization complete!")

if __name__ == '__main__':
    main()
EOF

    chmod +x "$plot_script"
    log_success "Visualization script created: $plot_script"
}

# Main analysis function
main() {
    log_section "PRISM Campaign Results Analysis"

    log_info "Campaign: $(basename $CAMPAIGN_DIR)"
    log_info "Analysis directory: $ANALYSIS_DIR"

    # Generate CSV data
    generate_csv

    # Perform analyses
    perform_statistical_analysis
    analyze_phase_performance
    generate_recommendations

    # Create visualization script
    create_visualization_script

    # Try to run Python visualization if available
    if command -v python3 &> /dev/null; then
        log_info "Running visualization script..."
        python3 "$ANALYSIS_DIR/plot_results.py" "$ANALYSIS_DIR/campaign_results.csv" 2>/dev/null || {
            log_info "Visualization requires matplotlib, seaborn, pandas"
            log_info "Install with: pip install matplotlib seaborn pandas"
        }
    fi

    log_section "Analysis Complete"
    echo ""
    echo "Generated files:"
    echo "  - $ANALYSIS_DIR/campaign_results.csv"
    echo "  - $ANALYSIS_DIR/statistical_report.txt"
    echo "  - $ANALYSIS_DIR/phase_analysis.txt"
    echo "  - $ANALYSIS_DIR/recommendations.txt"
    echo "  - $ANALYSIS_DIR/plot_results.py"
    echo ""
    echo "To generate visualizations:"
    echo "  python3 $ANALYSIS_DIR/plot_results.py $ANALYSIS_DIR/campaign_results.csv"
}

# Run analysis
main
