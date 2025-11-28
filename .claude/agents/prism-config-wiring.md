# prism-config-wiring Agent

**Purpose:** Systematically wire up ALL hardcoded phases to be fully tunable via TOML configuration

**Scope:** Wire up Phase 0, 1, 4, 6, 7 + complete Metaphysical Coupling integration

---

## Mission

Transform PRISM from partially-configurable to FULLY-TUNABLE by implementing the Phase 3 `::with_config()` pattern across all remaining phases.

**Current Status:**
- ✅ Phase 2, 3, Memetic, Global: FULLY TUNABLE
- ❌ Phase 0, 1, 4, 6, 7: HARDCODED (need wiring)
- ⚠️ Metaphysical Coupling: PARTIAL (need completion)

**Goal:** Make ALL phases tunable via TOML with no hardcoded values.

---

## Reference Specification

The complete specification is in `CONFIG_WIRE_UP_SPECIFICATION.md` at the project root.

**Key Pattern to Follow (from Phase 3):**
1. Create config struct in phase file with `#[serde(default)]`
2. Add `::with_config()` constructor
3. Parse in CLI `prism-cli/src/main.rs`
4. Add field to orchestrator + setter method
5. Update orchestrator initialization
6. Phase uses config values (not hardcoded)

---

## Tasks (Execute in Priority Order)

### Task 1: Wire Up Phase 4 Geodesic ⏱️ ~30 min
**Priority:** HIGHEST (geometry producer)

**Steps:**
1. Read `prism-phases/src/phase4_geodesic.rs` to understand current implementation
2. Create `Phase4Config` struct with parameters from spec
3. Add `::with_config()` and `::with_config_and_gpu()` constructors
4. Add config fields to `Phase4Geodesic` struct
5. Parse `[phase4_geodesic]` in CLI
6. Add `phase4_config` field to orchestrator
7. Add `set_phase4_config()` setter
8. Update orchestrator initialization (around line 326)
9. Apply config in CLI after creating orchestrator
10. Build and test with custom config

**Success Criteria:**
- ✅ Build succeeds
- ✅ Log shows "Phase 4: Initializing with custom TOML config"
- ✅ Telemetry shows custom parameter values

### Task 2: Wire Up Phase 1 Active Inference ⏱️ ~30 min
**Priority:** HIGH (geometry consumer)

**Steps:** Same pattern as Task 1, targeting `prism-phases/src/phase1_active_inference.rs`

**Config Parameters:**
```toml
[phase1_active_inference]
prior_precision = 1.0
likelihood_precision = 2.0
learning_rate = 0.001
free_energy_threshold = 0.01
num_iterations = 1000
hidden_states = 64
policy_depth = 3
exploration_bonus = 0.1
gpu_enabled = true
```

### Task 3: Wire Up Phase 6 TDA ⏱️ ~30 min
**Priority:** HIGH (geometry merger)

**Steps:** Same pattern, targeting `prism-phases/src/phase6_tda.rs`

**Config Parameters:**
```toml
[phase6_tda]
persistence_threshold = 0.1
max_dimension = 2
coherence_cv_threshold = 0.3
vietoris_rips_radius = 2.0
num_landmarks = 100
use_witness_complex = false
gpu_enabled = true
```

### Task 4: Wire Up Phase 0 Dendritic ⏱️ ~30 min
**Priority:** MEDIUM (foundational)

**Steps:** Same pattern, targeting `prism-phases/src/phase0/controller.rs`

**Config Parameters:**
```toml
[phase0_dendritic]
num_branches = 10
branch_depth = 6
learning_rate = 0.01
plasticity = 0.05
activation_threshold = 0.5
reservoir_size = 512
readout_size = 128
gpu_enabled = true
```

### Task 5: Wire Up Phase 7 Ensemble ⏱️ ~30 min
**Priority:** MEDIUM (ensemble)

**Steps:** Same pattern, targeting `prism-phases/src/phase7_ensemble.rs`

**Note:** Memetic config already works, just wire ensemble-specific params.

**Config Parameters:**
```toml
[phase7_ensemble]
num_replicas = 64
exchange_interval = 10
temperature_range = [0.1, 2.0]
diversity_weight = 0.1
consensus_threshold = 0.7
voting_method = "weighted"
replica_selection = "best"
gpu_enabled = false
```

### Task 6: Complete Metaphysical Coupling ⏱️ ~20 min
**Priority:** MEDIUM (cross-phase integration)

**Integrate geometry metrics into:**

**Phase 2 (`prism-phases/src/phase2_thermodynamic.rs`):**
```rust
// In execute() before GPU call:
if let Some(ref geom) = context.geometry_metrics {
    let stress_factor = geom.stress_scalar / 100.0;
    if stress_factor > 0.5 {
        adjusted_temp_max *= 1.0 + (stress_factor - 0.5);
        log::debug!("Phase2: Geometry stress → adjusted temp_max to {:.3}", adjusted_temp_max);
    }
}
```

**Phase 3 (`prism-phases/src/phase3_quantum.rs`):**
```rust
// In execute() before quantum evolution:
if let Some(ref geom) = context.geometry_metrics {
    let stress_factor = geom.stress_scalar / 100.0;
    if stress_factor > 0.5 {
        adjusted_coupling *= 1.0 + (stress_factor - 0.5) * 0.5;
        log::debug!("Phase3: Geometry stress → adjusted coupling to {:.3}", adjusted_coupling);
    }
}
```

**Phase 7 (`prism-phases/src/phase7_ensemble.rs`):**
```rust
// In ensemble selection:
if let Some(ref geom) = context.geometry_metrics {
    let hotspot_count = geom.anchor_hotspots.len();
    if hotspot_count > 10 {
        adjusted_diversity *= 1.0 + (hotspot_count as f32 / 20.0);
        log::debug!("Phase7: {} hotspots → adjusted diversity", hotspot_count);
    }
}
```

### Task 7: Update Documentation ⏱️ ~15 min

**Files to Update:**

1. **AGENT_READY_HYPERTUNING_GUIDE.md:**
   - Move Phase 0, 1, 4, 6, 7 from "❌ FAKE" to "✅ REAL"
   - Add complete parameter reference for each
   - Update summary table

2. **VERIFIED_CONFIG_FLOW.md:**
   - Update status from "FAKE" to "REAL"
   - Add flow diagrams for each phase

3. **Create example config:**
   `configs/full_tunable_example.toml` with ALL parameters exposed

---

## Agent Guidelines

### Must Follow

1. **Use Phase 3 Pattern:** Always follow the `::with_config()` pattern from Phase 3
2. **Test After Each Phase:** Build and verify each phase works before moving to next
3. **Keep Backwards Compatibility:** Always provide `::new()` fallback
4. **Log Config Loading:** Add `log::info!("Phase X: Initializing with custom TOML config")`
5. **Document Changes:** Comment what each config parameter does

### Code Quality Standards

- ✅ Use `#[serde(default = "function_name")]` for all optional fields
- ✅ Create separate default functions (e.g., `fn default_num_branches() -> usize { 10 }`)
- ✅ Add doc comments to config structs
- ✅ Use `#[cfg(feature = "cuda")]` guards for GPU code
- ✅ Preserve existing functionality (no breaking changes)

### Build Requirements

After each phase:
```bash
cargo build --release --features cuda
```

Must succeed with no errors.

### Testing Requirements

For each phase:
1. Create test config with custom values
2. Run: `./target/release/prism-cli --config test_config.toml --input benchmarks/dimacs/DSJC125.5.col --attempts 1`
3. Verify logs show: "Phase X: Initializing with custom TOML config"
4. Check telemetry shows custom parameter values

---

## Progress Tracking

Use TodoWrite tool to track:
```
[1] Wire Phase 4 Geodesic
[2] Wire Phase 1 Active Inference
[3] Wire Phase 6 TDA
[4] Wire Phase 0 Dendritic
[5] Wire Phase 7 Ensemble
[6] Complete Metaphysical Coupling
[7] Update Documentation
```

Mark each completed after successful build + test.

---

## Verification Checklist

After all tasks complete:

### Per-Phase Verification
- [ ] Config struct created with all parameters
- [ ] `::with_config()` constructor added
- [ ] CLI parsing added
- [ ] Orchestrator field + setter added
- [ ] Orchestrator initialization updated
- [ ] Phase uses config values (not hardcoded)
- [ ] Build succeeds
- [ ] Test shows custom config working

### Final Verification
- [ ] All 5 phases (0,1,4,6,7) now tunable via TOML
- [ ] Metaphysical coupling integrated in Phases 2, 3, 7
- [ ] Documentation updated
- [ ] Example config created with ALL parameters
- [ ] Build succeeds: `cargo build --release --features cuda`
- [ ] Test run with full config works

---

## Deliverables

1. ✅ Phase 0 fully tunable
2. ✅ Phase 1 fully tunable
3. ✅ Phase 4 fully tunable
4. ✅ Phase 6 fully tunable
5. ✅ Phase 7 fully tunable
6. ✅ Metaphysical coupling complete
7. ✅ Updated AGENT_READY_HYPERTUNING_GUIDE.md
8. ✅ Updated VERIFIED_CONFIG_FLOW.md
9. ✅ Example `configs/full_tunable_example.toml`
10. ✅ All builds succeed
11. ✅ All tests pass

---

## Estimated Completion Time

**Total:** ~3 hours

**Breakdown:**
- Phase wiring: 2.5 hours (5 phases × 30 min)
- Metaphysical coupling: 20 min
- Documentation: 15 min
- Testing/validation: 15 min

---

## Reference Files

**Primary Specification:**
- `CONFIG_WIRE_UP_SPECIFICATION.md` - Complete detailed specification

**Reference Implementation (Working Example):**
- `prism-core/src/types.rs:983-1019` - Phase3Config struct
- `prism-cli/src/main.rs:990-999` - CLI parsing
- `prism-pipeline/src/orchestrator/mod.rs:74,82-84,263-269` - Orchestrator
- `prism-phases/src/phase3_quantum.rs:145-170` - Constructor

**Files to Modify:**
- `prism-phases/src/phase0/controller.rs`
- `prism-phases/src/phase1_active_inference.rs`
- `prism-phases/src/phase2_thermodynamic.rs`
- `prism-phases/src/phase3_quantum.rs`
- `prism-phases/src/phase4_geodesic.rs`
- `prism-phases/src/phase6_tda.rs`
- `prism-phases/src/phase7_ensemble.rs`
- `prism-cli/src/main.rs`
- `prism-pipeline/src/orchestrator/mod.rs`
- `AGENT_READY_HYPERTUNING_GUIDE.md`
- `VERIFIED_CONFIG_FLOW.md`

---

**Agent Type:** Specialized config wiring
**Priority:** HIGH - Enables full hyperparameter tuning
**Complexity:** MEDIUM - Repetitive pattern across phases
**Risk:** LOW - Backwards compatible, non-breaking changes
