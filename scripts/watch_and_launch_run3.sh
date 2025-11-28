#!/bin/bash
# Watches Run 2 completion and auto-launches Run 3

LOG_FILE="/mnt/c/Users/Predator/Desktop/PRISM-v2/logs/run_256attempts_20251119.log"
LAUNCH_SCRIPT="/mnt/c/Users/Predator/Desktop/PRISM-v2/scripts/launch_run3.sh"

echo "════════════════════════════════════════════════════════"
echo "Watching Run 2 for completion..."
echo "════════════════════════════════════════════════════════"

while true; do
  # Check if memetic evolution completed
  if grep -q "Memetic: Evolution complete" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "✅ Run 2 COMPLETE! Detected memetic completion."
    echo "Extracting final results..."
    
    # Extract final results
    FINAL_CHROMATIC=$(grep "Final chromatic number:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
    FINAL_CONFLICTS=$(grep "Final conflicts:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
    BEST_CHROMATIC=$(grep "Best chromatic number:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
    
    echo "  Final chromatic: $FINAL_CHROMATIC"
    echo "  Final conflicts: $FINAL_CONFLICTS"
    echo "  Best chromatic: $BEST_CHROMATIC"
    echo ""
    echo "🚀 Launching Run 3 in 5 seconds..."
    sleep 5
    
    # Launch Run 3
    bash "$LAUNCH_SCRIPT"
    break
  fi
  
  # Show progress every 2 minutes
  ATTEMPTS=$(grep -c "Attempt [0-9]*/256" "$LOG_FILE" 2>/dev/null || echo "0")
  echo "[$(date +%H:%M:%S)] Run 2 progress: $ATTEMPTS/256 attempts"
  
  sleep 120
done
