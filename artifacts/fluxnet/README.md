# FluxNet RL Q-Table Artifacts

This directory contains pretrained FluxNet Q-tables for PRISM's universal RL controller.

---

## curriculum_bank_v3_geometry.bin

**Status**: ✅ TRAINING COMPLETED (2025-11-19 18:41 UTC)

**Description**: Q-table trained with Deep Metaphysical Coupling geometry reward shaping

**Training Graph**: DSJC250.5
- Vertices: 250
- Edges: 15668
- Density: 0.503

**Training Parameters**:
- **Epochs**: 1000
- **Alpha** (learning rate): 0.1
- **Gamma** (discount factor): 0.95
- **Epsilon Start**: 0.3 → 0.05 (min), decay 0.995
- **State Space**: Compact mode (4096 discrete states)
- **Action Space**: 88 universal actions (all phases + geometry)
- **Geometry Reward Shaping**: ENABLED (scale=2.0)
- **Replay Buffer**: 10000 transitions, batch size 32

**Training Command**:
```bash
RUST_LOG=info ./target/release/fluxnet_train \
  benchmarks/dimacs/DSJC250.5.col \
  1000 \
  artifacts/fluxnet/curriculum_bank_v3_geometry.bin
```

**Actual Training Results**:
- **Training Time**: 0.1s (1000 epochs)
- **Best Chromatic Number**: 242 colors (reached at epoch 116)
- **Average Reward**: 1498.37
- **Final Epsilon**: 0.050
- **Binary Size**: 9.9 MB (10,322,170 bytes)
- **SHA256**: `d18e20be97e8aec7ccd2a7377728295718dd1f75de22e30b46081410abbad217`

**Outcome**: Q-table with learned policies that minimize geometric stress

---

## Critical Fix: Deadlock Resolution (2025-11-19)

**Problem**: Training binary consistently hung at epoch 3 with 0% CPU usage.

**Root Cause**: Deadlock in `replay_batch()` method (prism-fluxnet/src/core/controller.rs:293-318):
- `replay_batch` acquired READ lock on `replay_buffer`
- Called `update_qtable` while holding the READ lock
- `update_qtable` attempted to acquire WRITE lock on same `replay_buffer`
- **RwLock deadlock**: Cannot upgrade from READ to WRITE lock

**Solution**: Clone transitions before releasing READ lock (controller.rs:305-317):
```rust
// Clone transitions while holding read lock to avoid deadlock
let transitions: Vec<Transition> = indices
    .iter()
    .filter_map(|&idx| buffer.get(idx).cloned())
    .collect();

drop(buffer); // Release lock before updating Q-table

// Now update Q-tables without holding any locks
for (state, action, reward, next_state) in transitions {
    self.update_qtable(&state, &action, reward, &next_state, phase);
}
```

**Verification**:
- 10-epoch test: ✅ PASS (completed in <0.1s)
- 1000-epoch production: ✅ PASS (completed in 0.1s)
- All epochs 1-1000 executed successfully, no hang at epoch 3

**Files Modified**:
- `prism-fluxnet/src/core/controller.rs` (+6 lines, refactored replay_batch)

**Logs**:
- Test run: `artifacts/logs/fluxnet_train_test_deadlock_fix.log`
- Production run: `artifacts/logs/fluxnet_train_v3_fixed.log`

---

**Geometry Reward Shaping**:
- Positive reward when geometry stress decreases
- Negative penalty when geometry stress increases
- Reward delta = (previous_stress - current_stress) × 2.0
- Enables RL to learn stress-minimizing policies

---

## Usage with PRISM CLI

Load the Q-table with the `--fluxnet-qtable` flag:

```bash
cargo run --release --features cuda --bin prism-cli -- \
  --input benchmarks/dimacs/DSJC250.5.col \
  --config configs/dsjc250_deep_coupling.toml \
  --attempts 16 \
  --warmstart \
  --gpu \
  --fluxnet-qtable artifacts/fluxnet/curriculum_bank_v3_geometry.bin
```

---

## Expected Improvements vs Baseline

**Baseline (No Q-Table)**:
- Best Chromatic Number: 41 colors
- Total Runtime: 752.144s (16 attempts)
- Avg per Attempt: 47.009s

**Expected with Q-Table**:
- Chromatic Number: < 41 colors (target: 38-40)
- Faster Convergence: Better phase parameter selection
- Stress Reduction: Lower final geometry stress
- RL Guidance: Learned policies from 1000 training episodes

---

## Training Log Analysis

Monitor training progress:
```bash
tail -f ../logs/fluxnet_train_v3.log
```

Check final training results:
```bash
grep "Training complete" ../logs/fluxnet_train_v3.log
grep "Final best chromatic" ../logs/fluxnet_train_v3.log
```

---

## Technical Details

**Q-Table Structure**:
- Format: Binary (bincode serialization)
- Size: ~2-5 MB (depends on state space size)
- Layout: HashMap<(StateIndex, ActionIndex), QValue>
- State Space: 4096 discrete states (12-bit hash)
- Action Space: 7 actions per state

**Geometry Integration**:
- UniversalRLState includes geometry_stress_level
- UniversalRLState includes geometry_overlap_density
- UniversalRLState includes geometry_hotspot_count
- UniversalRLState includes previous_geometry_stress (for delta computation)
- Reward bonus computed via compute_geometry_reward_bonus()

**References**:
- FluxNet RL Implementation: `prism-fluxnet/src/core/`
- Training Binary: `prism-fluxnet/src/bin/train.rs`
- State Representation: `prism-fluxnet/src/core/state.rs`
- Reward Shaping: `prism-fluxnet/src/core/controller.rs:235-244`
- Training Spec: `docs/fluxnet_retraining_spec.md`

---

**Last Updated**: 2025-11-19
