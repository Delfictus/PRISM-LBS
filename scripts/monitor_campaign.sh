#!/bin/bash
# PRISM Campaign Real-Time Monitor
# Provides live updates on campaign progress with visualization

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

CAMPAIGN_DIR="${1:-$(ls -td /mnt/c/Users/Predator/Desktop/PRISM/campaigns/* 2>/dev/null | head -1)}"
REFRESH_INTERVAL=${2:-5}  # seconds

if [ -z "$CAMPAIGN_DIR" ] || [ ! -d "$CAMPAIGN_DIR" ]; then
    echo "Usage: $0 <campaign_directory> [refresh_interval]"
    echo ""
    echo "Available campaigns:"
    ls -td /mnt/c/Users/Predator/Desktop/PRISM/campaigns/* 2>/dev/null | head -5
    exit 1
fi

STATE_FILE="$CAMPAIGN_DIR/state.json"
TELEMETRY_DIR="$CAMPAIGN_DIR/telemetry"

# Clear screen and move cursor to top
clear_screen() {
    printf '\033[2J\033[H'
}

# Draw progress bar
draw_progress_bar() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))

    printf "["
    printf "${GREEN}%0.s=" $(seq 1 $filled)
    printf "%0.s-" $(seq 1 $empty)
    printf "${NC}] ${BOLD}%3d%%${NC} (%d/%d)" $percentage $current $total
}

# Format duration
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

# Create sparkline for trend visualization
create_sparkline() {
    local -a values=("$@")
    local min=999
    local max=0
    local spark=""
    local ticks="▁▂▃▄▅▆▇█"

    # Find min/max
    for val in "${values[@]}"; do
        if [ "$val" != "999" ]; then
            [ "$val" -lt "$min" ] && min=$val
            [ "$val" -gt "$max" ] && max=$val
        fi
    done

    # Generate sparkline
    if [ "$max" -gt "$min" ]; then
        for val in "${values[@]}"; do
            if [ "$val" == "999" ]; then
                spark+="?"
            else
                local normalized=$(((val - min) * 7 / (max - min)))
                spark+="${ticks:$normalized:1}"
            fi
        done
    else
        spark=$(printf '▄%.0s' "${values[@]}")
    fi

    echo "$spark"
}

# Extract metrics from telemetry file
extract_metrics() {
    local telemetry_file=$1

    if [ ! -f "$telemetry_file" ]; then
        echo "999 999 999.0 999.0"
        return
    fi

    local chromatic=$(jq -s 'map(select(.metrics.num_colors != null and .outcome == "Success")) | map(.metrics.num_colors) | min // 999' "$telemetry_file" 2>/dev/null || echo "999")
    local conflicts=$(jq -s 'map(select(.metrics.conflicts != null)) | map(.metrics.conflicts) | max // 999' "$telemetry_file" 2>/dev/null || echo "999")
    local stress=$(jq -s 'map(select(.geometry.stress != null)) | map(.geometry.stress) | max // 999.0' "$telemetry_file" 2>/dev/null || echo "999.0")
    local exec_time=$(jq -s 'map(select(.metrics.execution_time_ms != null)) | map(.metrics.execution_time_ms) | add // 999.0' "$telemetry_file" 2>/dev/null || echo "999.0")

    echo "$chromatic $conflicts $stress $exec_time"
}

# Main monitoring loop
monitor_campaign() {
    local last_iter=0

    while true; do
        clear_screen

        # Check if state file exists
        if [ ! -f "$STATE_FILE" ]; then
            echo -e "${RED}Campaign state file not found: $STATE_FILE${NC}"
            sleep $REFRESH_INTERVAL
            continue
        fi

        # Load campaign state
        local campaign_name=$(jq -r '.campaign_name' "$STATE_FILE")
        local start_time=$(jq -r '.start_time' "$STATE_FILE")
        local target_chromatic=$(jq -r '.target_chromatic' "$STATE_FILE")
        local iterations_completed=$(jq -r '.iterations_completed' "$STATE_FILE")
        local best_chromatic=$(jq -r '.best_chromatic' "$STATE_FILE")
        local best_conflicts=$(jq -r '.best_conflicts' "$STATE_FILE")
        local best_iteration=$(jq -r '.best_iteration' "$STATE_FILE")
        local target_achieved=$(jq -r '.target_achieved' "$STATE_FILE")

        # Calculate duration
        local start_epoch=$(date -d "$start_time" +%s 2>/dev/null || date +%s)
        local current_epoch=$(date +%s)
        local duration=$((current_epoch - start_epoch))

        # Header
        echo -e "${BOLD}${MAGENTA}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${BOLD}${MAGENTA}║${NC}          ${BOLD}PRISM WORLD RECORD CAMPAIGN - LIVE MONITOR${NC}                    ${BOLD}${MAGENTA}║${NC}"
        echo -e "${BOLD}${MAGENTA}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""

        # Campaign info
        echo -e "${CYAN}Campaign:${NC}        $campaign_name"
        echo -e "${CYAN}Started:${NC}         $(date -d "$start_time" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo 'Unknown')"
        echo -e "${CYAN}Duration:${NC}        $(format_duration $duration)"
        echo -e "${CYAN}Target:${NC}          ${BOLD}$target_chromatic colors${NC} with ${BOLD}0 conflicts${NC}"
        echo ""

        # Progress
        echo -e "${BOLD}Progress${NC}"
        echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        draw_progress_bar $iterations_completed 20
        echo ""
        echo ""

        # Best results
        echo -e "${BOLD}Best Result${NC}"
        echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        if [ "$target_achieved" == "true" ]; then
            echo -e "${GREEN}${BOLD}🏆 TARGET ACHIEVED! 🏆${NC}"
            echo ""
        fi

        if [ "$best_chromatic" == "999" ]; then
            echo -e "  ${YELLOW}No successful runs yet${NC}"
        else
            # Determine color based on progress
            local chromatic_color="${YELLOW}"
            [ "$best_chromatic" -le "$target_chromatic" ] && chromatic_color="${GREEN}"

            local conflict_color="${YELLOW}"
            [ "$best_conflicts" == "0" ] && conflict_color="${GREEN}"

            echo -e "  Chromatic Number:  ${chromatic_color}${BOLD}$best_chromatic${NC} / $target_chromatic"
            echo -e "  Conflicts:         ${conflict_color}${BOLD}$best_conflicts${NC}"
            echo -e "  Found at:          Iteration $best_iteration"
        fi
        echo ""

        # Recent iterations
        echo -e "${BOLD}Recent Iterations${NC}"
        echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        printf "%-6s %-12s %-10s %-8s %-10s %s\n" "Iter" "Chromatic" "Conflicts" "Stress" "Time(ms)" "Status"
        echo "─────────────────────────────────────────────────────────────────────────────"

        # Collect data for sparklines
        local -a chromatic_trend=()
        local -a conflict_trend=()

        # Show last 10 iterations
        local start_iter=$((iterations_completed - 9))
        [ $start_iter -lt 1 ] && start_iter=1

        for i in $(seq $start_iter $iterations_completed); do
            local telemetry="$TELEMETRY_DIR/iter_${i}.jsonl"

            if [ -f "$telemetry" ]; then
                read chromatic conflicts stress exec_time <<< $(extract_metrics "$telemetry")

                chromatic_trend+=($chromatic)
                conflict_trend+=($conflicts)

                # Format values
                local chromatic_str="$chromatic"
                local conflicts_str="$conflicts"
                local stress_str=$(printf "%.2f" $stress)
                local time_str=$(printf "%.0f" $exec_time)

                # Color coding
                local status=""
                if [ "$chromatic" == "999" ]; then
                    status="${RED}FAILED${NC}"
                    chromatic_str="${RED}---${NC}"
                    conflicts_str="${RED}---${NC}"
                elif [ "$chromatic" -le "$target_chromatic" ] && [ "$conflicts" == "0" ]; then
                    status="${GREEN}⭐ PERFECT${NC}"
                    chromatic_str="${GREEN}$chromatic${NC}"
                    conflicts_str="${GREEN}$conflicts${NC}"
                elif [ "$chromatic" -le "$target_chromatic" ]; then
                    status="${YELLOW}⚠ HAS CONFLICTS${NC}"
                    chromatic_str="${YELLOW}$chromatic${NC}"
                    conflicts_str="${YELLOW}$conflicts${NC}"
                elif [ "$conflicts" == "0" ]; then
                    status="${CYAN}✓ NO CONFLICTS${NC}"
                    chromatic_str="${CYAN}$chromatic${NC}"
                    conflicts_str="${GREEN}$conflicts${NC}"
                else
                    status="${YELLOW}SUBOPTIMAL${NC}"
                    chromatic_str="${YELLOW}$chromatic${NC}"
                    conflicts_str="${YELLOW}$conflicts${NC}"
                fi

                printf "%-6s %-12s %-10s %-8s %-10s %b\n" \
                    "$i" "$chromatic_str" "$conflicts_str" "$stress_str" "$time_str" "$status"
            else
                printf "%-6s %-12s\n" "$i" "${YELLOW}In Progress...${NC}"
            fi
        done

        echo ""

        # Trend visualization
        if [ ${#chromatic_trend[@]} -gt 0 ]; then
            echo -e "${BOLD}Trends${NC}"
            echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo -e "  Chromatic:  $(create_sparkline "${chromatic_trend[@]}")"
            echo -e "  Conflicts:  $(create_sparkline "${conflict_trend[@]}")"
            echo ""
        fi

        # Current activity
        echo -e "${BOLD}Current Activity${NC}"
        echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # Check if there's a run in progress
        local latest_log="$CAMPAIGN_DIR/results/iter_${iterations_completed}.log"
        if [ -f "$latest_log" ]; then
            # Check if PRISM is currently running
            if pgrep -f "prism-cli.*$campaign_name" > /dev/null; then
                echo -e "  ${GREEN}●${NC} PRISM is running (iteration $iterations_completed)"

                # Show last few log lines
                echo -e "  ${CYAN}Recent log output:${NC}"
                tail -3 "$latest_log" 2>/dev/null | sed 's/^/    /'
            else
                echo -e "  ${YELLOW}○${NC} Waiting for next iteration..."
            fi
        else
            echo -e "  ${BLUE}◌${NC} Initializing..."
        fi

        echo ""
        echo -e "${CYAN}Last updated:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
        echo -e "${CYAN}Refresh rate:${NC} ${REFRESH_INTERVAL}s (press Ctrl+C to exit)"

        # Check if campaign is complete
        if [ "$target_achieved" == "true" ]; then
            echo ""
            echo -e "${GREEN}${BOLD}Campaign completed successfully!${NC}"
            break
        fi

        # Detect new iteration
        if [ $iterations_completed -ne $last_iter ]; then
            last_iter=$iterations_completed
            # Play bell on new iteration (if terminal supports it)
            printf '\a'
        fi

        sleep $REFRESH_INTERVAL
    done
}

# Start monitoring
echo "Starting campaign monitor for: $CAMPAIGN_DIR"
sleep 2
monitor_campaign
