#!/bin/bash
# Finalize integration documentation after 128-attempt validation completes
# This script extracts results and updates all documentation

set -e

LOG_FILE="artifacts/logs/gpu_run_with_qtable_128att.log"
TELEMETRY_FILE="telemetry_deep_coupling.jsonl"

echo "=== Integration Finalization ==="
echo ""

# Check if validation completed
if ! grep -q "Multi-attempt optimization completed" "$LOG_FILE"; then
    echo "ERROR: 128-attempt validation not yet complete!"
    echo "Current status:"
    ./monitor_validation.sh
    exit 1
fi

echo "✅ 128-attempt validation COMPLETE"
echo ""

# Extract final results
echo "--- Extracting Results ---"
BEST_CHROMATIC=$(grep "Best chromatic number:" "$LOG_FILE" | tail -1 | awk '{print $5}')
BEST_CONFLICTS=$(grep "Best conflicts:" "$LOG_FILE" | tail -1 | awk '{print $4}')
TOTAL_RUNTIME=$(grep "Total runtime:" "$LOG_FILE" | tail -1 | awk '{print $4}')
AVG_RUNTIME=$(grep "Avg per attempt:" "$LOG_FILE" | tail -1 | awk '{print $5}')
VALID=$(grep "Valid:" "$LOG_FILE" | tail -1 | awk '{print $3}')

echo "Best Chromatic: $BEST_CHROMATIC colors"
echo "Best Conflicts: $BEST_CONFLICTS"
echo "Total Runtime: $TOTAL_RUNTIME"
echo "Avg per Attempt: $AVG_RUNTIME"
echo "Valid: $VALID"

# Check geometry reward bonuses
REWARD_COUNT=$(grep -c "Geometry reward bonus" "$LOG_FILE" || echo "0")
echo "Geometry Reward Logs: $REWARD_COUNT"

# Check telemetry
TELEMETRY_LINES=$(wc -l < "$TELEMETRY_FILE" 2>/dev/null || echo "0")
echo "Telemetry Lines: $TELEMETRY_LINES"

echo ""
echo "--- Results Summary ---"
cat > artifacts/128_attempt_results.txt << EOF
128-Attempt Q-Table Validation Results
=======================================
Date: $(date -u +"%Y-%m-%d %H:%M UTC")
Log: $LOG_FILE

Final Results:
- Best Chromatic Number: $BEST_CHROMATIC colors
- Best Conflicts: $BEST_CONFLICTS
- Total Runtime: $TOTAL_RUNTIME
- Avg per Attempt: $AVG_RUNTIME
- Valid: $VALID

Geometry Coupling:
- Reward bonus logs: $REWARD_COUNT entries
- Telemetry lines: $TELEMETRY_LINES

Key Observations:
$(grep "NEW BEST" "$LOG_FILE" | head -5)

Geometry Stress (sample):
$(grep "Geometry telemetry" "$LOG_FILE" | head -3)
EOF

cat artifacts/128_attempt_results.txt

echo ""
echo "✅ Results extracted to artifacts/128_attempt_results.txt"
echo ""
echo "=== Next Steps (Manual) ==="
echo ""
echo "1. Update artifacts/COMPARATIVE_ANALYSIS.md Section 3:"
echo "   - Add 128-attempt results table"
echo "   - Update performance comparison"
echo "   - Finalize recommendations"
echo ""
echo "2. Update artifacts/PROGRESS_SUMMARY.md:"
echo "   - Mark 128-attempt as COMPLETE"
echo "   - Add final chromatic results"
echo "   - Update integration verdict"
echo ""
echo "3. Update INTEGRATION_STATUS_HONEST.md:"
echo "   - Change status to 'INTEGRATION-READY'"
echo "   - Add 128-attempt summary"
echo "   - Finalize known limitations"
echo ""
echo "4. Update docs/deep_coupling_integration_notes.md Section 8:"
echo "   - Add validation results table"
echo "   - Document Q-table effectiveness"
echo "   - Add tuning recommendations"
echo ""
echo "Use the following data in your updates:"
echo "  Best: $BEST_CHROMATIC colors"
echo "  Runtime: $TOTAL_RUNTIME"
echo "  Avg: $AVG_RUNTIME/attempt"
echo "  Rewards: $REWARD_COUNT geometry bonuses logged"
echo ""
echo "Files to update:"
echo "  - artifacts/COMPARATIVE_ANALYSIS.md"
echo "  - artifacts/PROGRESS_SUMMARY.md"
echo "  - INTEGRATION_STATUS_HONEST.md"
echo "  - docs/deep_coupling_integration_notes.md"
