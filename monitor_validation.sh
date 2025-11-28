#!/bin/bash
# Monitor 128-attempt validation progress
# Usage: ./monitor_validation.sh

LOG_FILE="artifacts/logs/gpu_run_with_qtable_128att.log"

echo "=== 128-Attempt Validation Monitor ==="
echo "Log: $LOG_FILE"
echo "Started: 2025-11-19 19:14 UTC"
echo "Estimated completion: ~22:30 UTC"
echo ""

# Check if log exists
if [ ! -f "$LOG_FILE" ]; then
    echo "ERROR: Log file not found!"
    exit 1
fi

# Show progress
echo "--- Current Progress ---"
ATTEMPT_COUNT=$(grep -c "Attempt [0-9]\+/128" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Attempts completed: $ATTEMPT_COUNT / 128"

if [ "$ATTEMPT_COUNT" -gt 0 ]; then
    LATEST=$(grep "Attempt [0-9]\+/128" "$LOG_FILE" | tail -1)
    echo "Latest: $LATEST"

    # Calculate ETA
    START_TIME="2025-11-19 19:14:00"
    ELAPSED=$(($(date +%s) - $(date -d "$START_TIME" +%s 2>/dev/null || echo "0")))
    if [ "$ELAPSED" -gt 0 ] && [ "$ATTEMPT_COUNT" -gt 0 ]; then
        AVG_PER_ATTEMPT=$((ELAPSED / ATTEMPT_COUNT))
        REMAINING=$((128 - ATTEMPT_COUNT))
        ETA_SECONDS=$((REMAINING * AVG_PER_ATTEMPT))
        ETA_HOURS=$((ETA_SECONDS / 3600))
        ETA_MINS=$(((ETA_SECONDS % 3600) / 60))
        echo "Average: ${AVG_PER_ATTEMPT}s per attempt"
        echo "ETA: ${ETA_HOURS}h ${ETA_MINS}m remaining"
    fi
fi

echo ""
echo "--- Best Results So Far ---"
grep "NEW BEST" "$LOG_FILE" | tail -5

echo ""
echo "--- Geometry Reward Bonuses ---"
REWARD_COUNT=$(grep -c "Geometry reward bonus" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Reward bonus logs: $REWARD_COUNT"
if [ "$REWARD_COUNT" -gt 0 ]; then
    grep "Geometry reward bonus" "$LOG_FILE" | tail -5
fi

echo ""
echo "--- Check Completion ---"
if grep -q "Multi-attempt optimization completed" "$LOG_FILE"; then
    echo "✅ VALIDATION COMPLETE!"
    echo ""
    grep "Multi-attempt optimization completed" "$LOG_FILE" -A 10
else
    echo "🟢 Still running..."
    echo ""
    echo "Monitor commands:"
    echo "  tail -f $LOG_FILE"
    echo "  grep 'Attempt' $LOG_FILE | tail -10"
    echo "  ./monitor_validation.sh"
fi
