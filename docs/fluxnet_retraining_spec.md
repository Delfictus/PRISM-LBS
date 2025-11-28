# FluxNet Retraining Specification

**Document Version:** 1.0
**Date:** 2025-01-18
**Purpose:** Guide for retraining FluxNet Q-tables to exploit metaphysical telemetry coupling features

---

## Overview

FluxNet v2 introduces **geometry-aware reward shaping** that leverages metaphysical telemetry coupling from Phase 4 (Geodesic) and Phase 6 (TDA). To fully exploit these new features, Q-tables must be retrained with geometry feedback enabled.

**Key Enhancement:** The RL controller now receives bonus rewards when actions reduce geometric stress (measured by `stress_scalar` from `GeometryTelemetry`). This enables FluxNet to learn policies that minimize conflicts by understanding the geometric structure of the coloring problem.

---

## New State Dimensions

The `UniversalRLState` now includes **three additional geometry-aware dimensions** for metaphysical coupling:

### 1. `geometry_stress_level: f64`
- **Range:** 0.0 (no stress) to 1.0 (critical stress)
- **Source:** `GeometryTelemetry::stress_scalar` computed from Phase 4/6
- **Meaning:** Composite metric of geometric conflict intensity
- **Formula:** `stress = 0.4 * overlap_density + 0.3 * growth_rate + 0.3 * (bounding_box_area - 1.0).max(0.0)`
- **Thresholds:**
  - `< 0.5`: Low stress (stable geometry)
  - `0.5 - 0.8`: Moderate stress (needs adjustment)
  - `> 0.8`: Critical stress (aggressive intervention required)

### 2. `geometry_overlap_density: f64`
- **Range:** 0.0 (no conflicts) to 1.0 (all edges conflict)
- **Source:** `GeometryTelemetry::overlap_density`
- **Meaning:** Fraction of edges with same-color endpoints
- **Formula:** `conflicts / num_edges`
- **Use Case:** Direct measure of coloring quality

### 3. `geometry_hotspot_count: usize`
- **Range:** 0 to N (number of high-conflict vertices)
- **Source:** `GeometryTelemetry::anchor_hotspots.len()`
- **Meaning:** Number of vertices requiring priority attention
- **Use Case:** Guides local search and warmstart anchoring

### 4. `previous_geometry_stress: f64`
- **Range:** 0.0 to 1.0
- **Purpose:** Tracks stress from previous step for reward delta computation
- **Updated by:** `UniversalRLState::update_geometry_stress(new_stress)`
- **Not hashed:** Only used for reward computation, not state discretization

---

## Reward Shaping Formula

The RL controller now computes **geometry reward bonuses** based on stress reduction:

```rust
/// Computes geometry-based reward bonus for reinforcement learning.
///
/// # Algorithm
/// - Stress decrease: positive reward proportional to delta
/// - Stress increase: negative reward (penalty)
/// - Scale factor: 2.0 (makes geometry feedback significant)
///
/// # Returns
/// Reward bonus in range [-2.0, +2.0]
pub fn compute_geometry_reward_bonus(&self) -> f64 {
    let stress_delta = self.previous_geometry_stress - self.geometry_stress_level;
    stress_delta * 2.0  // Scale by reward_shaping_scale (default: 2.0)
}
```

### Example Scenarios

| Scenario | Previous Stress | Current Stress | Delta | Reward Bonus |
|----------|----------------|----------------|-------|--------------|
| Improvement | 0.80 | 0.50 | +0.30 | **+0.60** |
| Deterioration | 0.30 | 0.60 | -0.30 | **-0.60** |
| No change | 0.50 | 0.50 | 0.00 | 0.00 |
| Large improvement | 0.90 | 0.20 | +0.70 | **+1.40** |

### Integration with Q-Learning

The shaped reward is combined with the outcome-based reward:

```rust
// Compute geometry reward bonus from stress reduction
let geometry_bonus = next_state.compute_geometry_reward_bonus();
let shaped_reward = reward + geometry_bonus as f32;

// Q-learning update with shaped reward
let new_q = current_q + alpha * (shaped_reward + gamma * max_next_q - current_q);
```

**Logging:** Geometry bonuses are logged at INFO level when `|bonus| > 0.01`:

```
[INFO] FluxNet: Geometry reward bonus +0.60 (stress decreased from 0.80 to 0.50)
```

---

## Configuration Flags

Control geometry coupling behavior via `MetaphysicalCouplingConfig`:

### `enable_reward_shaping: bool` (default: `true`)
- Toggles geometry bonus rewards in FluxNet Q-table updates
- Set `false` to disable reward shaping (for ablation studies)

### `reward_shaping_scale: f64` (default: `2.0`)
- Multiplier for geometry reward bonuses
- Higher values → stronger geometry influence
- Lower values → outcome-based rewards dominate
- Recommended range: 1.0 - 5.0

### `enable_early_phase_seeding: bool` (default: `true`)
- Uses Phase 0/1 proxy metrics for geometry before Phase 4/6 run
- Enables earlier metaphysical coupling feedback
- Proxy mapping:
  - `overlap_density ≈ mean_uncertainty` (from Phase 1 Active Inference)
  - `bounding_box_area ≈ mean_difficulty` (from Phase 0 Reservoir)

### Example TOML Configuration

```toml
[metaphysical_coupling]
enabled = true
enable_reward_shaping = true
reward_shaping_scale = 2.0
enable_early_phase_seeding = true
stress_hot_threshold = 0.5
stress_critical_threshold = 0.8
```

---

## Retraining Command

To retrain Q-tables with geometry coupling features, use the FluxNet training binary:

```bash
cargo run --bin fluxnet_train -- \
    --graph-set benchmarks/dsjc*.col \
    --episodes 10000 \
    --output curriculum_bank_v3.json \
    --geometry-coupling-enabled
```

### Command Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--graph-set` | Glob pattern for training graphs | `benchmarks/dsjc*.col` |
| `--episodes` | Number of training episodes per graph | `10000` |
| `--output` | Output path for curriculum Q-tables | `curriculum_bank_v3.json` |
| `--geometry-coupling-enabled` | Enable geometry reward shaping | (flag) |
| `--reward-shaping-scale` | Custom scale factor | `--reward-shaping-scale 3.0` |

### Training Recommendations

1. **Curriculum Structure:**
   - Train on diverse graph profiles (sparse, dense, medium, hard)
   - Use at least 10,000 episodes per profile for convergence
   - Monitor epsilon decay: should reach `epsilon_min` (~0.05) by final episodes

2. **Hyperparameters:**
   - **Alpha (learning rate):** 0.1 (default, stable convergence)
   - **Gamma (discount factor):** 0.95 (long-term planning)
   - **Epsilon decay:** 0.995 (gradual exploration reduction)
   - **Reward shaping scale:** 2.0 (balances geometry and outcome rewards)

3. **Validation:**
   - Compare Q-table statistics before/after retraining:
     ```rust
     let (mean, min, max) = controller.qtable_stats("Phase2-Thermodynamic");
     println!("Phase2: mean={:.3}, min={:.3}, max={:.3}", mean, min, max);
     ```
   - Expected: Higher Q-values near geometry hotspots, negative Q for stress-increasing actions

4. **A/B Testing:**
   - Run benchmark suite with old Q-tables (v2 without geometry)
   - Run same benchmarks with new Q-tables (v3 with geometry)
   - Compare chromatic numbers, conflicts, and convergence speed

---

## Expected Performance Impact

### With Geometry Coupling (v3 Q-tables)

**Advantages:**
- ✅ **Faster convergence:** Geometry feedback accelerates learning
- ✅ **Better conflict resolution:** Actions directly minimize geometric stress
- ✅ **Adaptive behavior:** FluxNet learns to prioritize hotspot regions
- ✅ **Cross-phase coordination:** Phase 4/6 geometry informs Phase 1/2/3 parameters

**Quantitative Expectations (DSJC250.5):**
- 10-15% reduction in conflicts per iteration
- 5-10% improvement in final chromatic number
- 20-30% faster convergence to local optima

### Without Geometry Coupling (v2 Q-tables)

**Limitations:**
- ❌ Blind to geometric structure (relies only on outcome rewards)
- ❌ Slower exploration in conflict-heavy regions
- ❌ No feedback loop between Phase 4/6 and RL controller

---

## Backward Compatibility

**Q-Table Format:** Binary compatible (same data structure)

**Migration Path:**
1. Retrain Q-tables with `--geometry-coupling-enabled`
2. Save as `curriculum_bank_v3.json`
3. Update pipeline config to use new catalog:
   ```toml
   [warmstart_config]
   curriculum_catalog_path = "curriculum_bank_v3.json"
   ```
4. Old Q-tables (v2) will still work but won't exploit geometry features

**Feature Flag:** Set `enable_reward_shaping = false` to revert to v2 behavior

---

## State Space Impact

### Discretization Changes

The state hash now includes geometry metrics:

```rust
// Geometry stress metrics (added in FluxNet v2)
quantize(self.geometry_stress_level).hash(&mut hasher);
quantize(self.geometry_overlap_density).hash(&mut hasher);
(self.geometry_hotspot_count % 256).hash(&mut hasher);
```

**Impact:**
- State space size unchanged (still 4096 compact / 65536 extended)
- Hash distribution more uniform (geometry adds entropy)
- Collision rate may decrease slightly due to additional dimensions

### Recommended Mode

**For production:** Use `DiscretizationMode::Extended` (65536 states)
- Provides better precision for geometry-aware policies
- Higher memory usage (65536 × 64 actions × 4 bytes ≈ 16MB per phase)
- Worth the tradeoff for improved coloring quality

**For development:** `DiscretizationMode::Compact` (4096 states) is sufficient

---

## Telemetry Output

Geometry metrics are now serialized in every telemetry event:

### Example JSON Output

```json
{
  "timestamp": "2025-01-18T12:34:56.789Z",
  "phase": "Phase2-Thermodynamic",
  "metrics": {
    "temperature": 1.25,
    "energy": -345.6,
    "acceptance_rate": 0.42
  },
  "outcome": "Success",
  "geometry": {
    "stress": 0.35,
    "overlap": 0.22,
    "hotspots": 5
  }
}
```

### Monitoring Geometry Trends

Use `jq` to extract geometry stress across phases:

```bash
cat telemetry.jsonl | jq -r '[.phase, .geometry.stress] | @tsv' | grep -v null
```

Expected output:
```
Phase1-ActiveInference   0.60
Phase2-Thermodynamic     0.45
Phase3-QuantumClassical  0.50
Phase4-Geodesic          0.35
Phase6-TDA               0.28
Phase7-Ensemble          0.25
```

---

## Testing and Validation

### Unit Tests

Geometry reward shaping is covered by existing tests in `prism-fluxnet/src/core/state.rs`:

```rust
#[test]
fn test_geometry_reward_bonus() {
    let mut state = UniversalRLState::new();
    state.previous_geometry_stress = 0.8;
    state.geometry_stress_level = 0.5;

    let bonus = state.compute_geometry_reward_bonus();
    assert!((bonus - 0.6).abs() < 0.01); // 2.0 * (0.8 - 0.5)
}
```

### Integration Tests

Run the full pipeline on a test graph:

```bash
cargo test --package prism-pipeline --lib -- --nocapture
```

Verify log output includes geometry reward messages:

```
[INFO] FluxNet: Geometry reward bonus +0.60 (stress decreased from 0.80 to 0.50)
```

### Benchmark Suite

Compare performance with/without geometry coupling:

```bash
# Run with geometry (v3 Q-tables)
./run_benchmarks.sh --curriculum curriculum_bank_v3.json --output results_v3.json

# Run without geometry (v2 Q-tables)
./run_benchmarks.sh --curriculum curriculum_bank_v2.json --output results_v2.json

# Compare results
python scripts/compare_results.py results_v2.json results_v3.json
```

---

## Future Enhancements

### Adaptive Reward Shaping Scale

Dynamically adjust `reward_shaping_scale` based on training progress:

```rust
// Start high (exploration), decay to lower (exploitation)
let scale = 5.0 * (1.0 - episode as f64 / max_episodes as f64).max(1.0);
```

### Phase-Specific Geometry Weights

Different phases may benefit from different geometry sensitivity:

```toml
[metaphysical_coupling.phase_weights]
Phase1 = 1.5  # Active Inference highly sensitive to stress
Phase2 = 2.0  # Thermodynamic benefits most from geometry
Phase3 = 1.0  # Quantum less dependent on geometry
```

### Geometry Action Space

Introduce dedicated geometry actions (already implemented in `UniversalAction`):

- `Geometry(GeometryAction::IncreaseStressThreshold)`
- `Geometry(GeometryAction::DecreaseStressThreshold)`
- `Geometry(GeometryAction::PrioritizeHotspots)`
- `Geometry(GeometryAction::RelaxConstraints)`

These actions explicitly adjust metaphysical coupling parameters during execution.

---

## References

- **PRISM GPU Plan §3.3:** UniversalRLController
- **PRISM GPU Plan §3.1:** UniversalRLState
- **Metaphysical Telemetry Coupling:** `prism-core/src/types.rs:815-871`
- **Geometry Reward Shaping:** `prism-fluxnet/src/core/state.rs:180-197`
- **FluxNet Controller Update:** `prism-fluxnet/src/core/controller.rs:215-276`

---

## Appendix: Q-Table Versioning

| Version | Features | Compatibility | Date |
|---------|----------|---------------|------|
| v1.0 | Basic RL (7 phases, no warmstart) | Legacy | 2024-Q3 |
| v2.0 | Warmstart + multi-attempt + memetic | PRISM v2 | 2024-Q4 |
| **v3.0** | **Geometry reward shaping** | **PRISM v2.1** | **2025-Q1** |

**Migration:** Q-tables are forward-compatible. v2 tables work in v3, but won't exploit geometry features. Retrain for best results.

---

**End of Specification**
