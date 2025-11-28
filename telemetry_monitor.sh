#!/bin/bash
# Real-time telemetry monitor for PRISM benchmark

LOG="memetic_100attempt_baseline.log"

echo "═══════════════════════════════════════════════════════════"
echo "PRISM Telemetry Snapshot - $(date +%H:%M:%S)"
echo "═══════════════════════════════════════════════════════════"

# Attempt progress
ATTEMPTS=$(grep -c "Attempt.*NEW BEST\|Attempt.*41 colors" "$LOG" 2>/dev/null || echo 0)
BEST_CHROMATIC=$(grep "Attempt.*NEW BEST\|41 colors" "$LOG" | tail -1 | grep -oP '\d+ colors' | grep -oP '\d+' || echo "N/A")
BEST_CONFLICTS=$(grep "Attempt.*NEW BEST" "$LOG" | tail -1 | grep -oP '\d+ conflicts' | grep -oP '\d+' || echo "N/A")
LAST_TIME=$(grep "Attempt.*NEW BEST\|Attempt.*41 colors" "$LOG" | tail -1 | grep -oP '\(\K[0-9.]+s' || echo "N/A")

echo ""
echo "📊 GPU ATTEMPT PROGRESS"
echo "  Attempts completed:  $ATTEMPTS / 100"
echo "  Best chromatic:      $BEST_CHROMATIC colors"
echo "  Best conflicts:      $BEST_CONFLICTS"
echo "  Last attempt time:   $LAST_TIME"

# Phase 6 Coherence
COHERENCE_CV=$(grep "Phase6 Coherence:" "$LOG" | tail -1 | grep -oP 'CV=\K[0-9.]+' || echo "N/A")
COHERENCE_WARNING=$(grep -c "Phase6 Coherence Warning" "$LOG" 2>/dev/null || echo 0)

echo ""
echo "🔬 PHASE 6 TDA COHERENCE"
echo "  Current CV:          $COHERENCE_CV"
echo "  CV warnings:         $COHERENCE_WARNING"
if (( $(echo "$COHERENCE_CV < 0.1" | bc -l 2>/dev/null || echo 0) )); then
    echo "  ⚠️  ALERT: CV < 0.1 - LOW TOPOLOGICAL VARIANCE!"
fi

# Phase 2 Annealing
PHASE2_INITIAL=$(grep "Phase2.*from warmstart\|Phase2.*greedy" "$LOG" | head -1 | grep -oP '\d+ colors' | grep -oP '\d+' || echo "N/A")
PHASE2_FINAL=$(grep "Phase2 GPU:" "$LOG" | tail -1 | grep -oP '\d+ colors' | grep -oP '\d+' || echo "N/A")
PHASE2_TIME=$(grep "Phase2 GPU:" "$LOG" | tail -1 | grep -oP '[0-9.]+ms' | head -1 || echo "N/A")

echo ""
echo "🌡️  PHASE 2 THERMODYNAMIC"
echo "  Initial colors:      $PHASE2_INITIAL"
echo "  Final colors:        $PHASE2_FINAL"
echo "  Annealing time:      $PHASE2_TIME"

# Memetic evolution (if started)
MEMETIC_STARTED=$(grep -c "Starting memetic evolution" "$LOG" 2>/dev/null || echo 0)
if [ "$MEMETIC_STARTED" -gt 0 ]; then
    MEMETIC_GEN=$(grep "Generation" "$LOG" | tail -1 | grep -oP 'Generation \K\d+' || echo "N/A")
    MEMETIC_BEST=$(grep "Generation" "$LOG" | tail -1 | grep -oP 'best=\K\d+' || echo "N/A")
    echo ""
    echo "🧬 MEMETIC EVOLUTION"
    echo "  Current generation:  $MEMETIC_GEN / 500"
    echo "  Best chromatic:      $MEMETIC_BEST colors"
else
    echo ""
    echo "🧬 MEMETIC EVOLUTION"
    echo "  Status:              Waiting for GPU phase to complete..."
fi

# Current phase
CURRENT_PHASE=$(tail -20 "$LOG" | grep "Phase.*executing\|Attempt.*/" | tail -1)
echo ""
echo "⚙️  CURRENT ACTIVITY"
echo "  $CURRENT_PHASE"

echo ""
echo "═══════════════════════════════════════════════════════════"
