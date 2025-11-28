# Warmstart CLI Usage Examples

This document provides practical examples for using the PRISM CLI warmstart flags.

## Overview

The warmstart system enhances graph coloring by providing probabilistic color priors derived from:
- **Flux Reservoir**: Phase 0 neuromorphic dynamics
- **Ensemble Methods**: Structural heuristics (greedy, DSATUR, etc.)
- **Random Exploration**: Uniform priors for entropy

## Basic Usage

### Enable Warmstart with Defaults

```bash
prism-cli --input graph.col --warmstart
```

**Default Configuration:**
- Max colors: 50
- Anchor fraction: 0.10 (10% of vertices)
- Flux weight: 0.40
- Ensemble weight: 0.40
- Random weight: 0.20

### Disable Warmstart (Default)

```bash
prism-cli --input graph.col
```

The pipeline runs without warmstart priors.

## Custom Weight Configurations

### High Flux Influence

Favor neuromorphic reservoir dynamics:

```bash
prism-cli --input graph.col --warmstart \
  --warmstart-flux-weight 0.6 \
  --warmstart-ensemble-weight 0.3 \
  --warmstart-random-weight 0.1
```

**Use Case:** Exploiting temporal dynamics and attractor states.

### High Ensemble Influence

Favor structural heuristics:

```bash
prism-cli --input graph.col --warmstart \
  --warmstart-flux-weight 0.2 \
  --warmstart-ensemble-weight 0.6 \
  --warmstart-random-weight 0.2
```

**Use Case:** Dense graphs where greedy/DSATUR perform well.

### High Exploration

Increase entropy for harder instances:

```bash
prism-cli --input graph.col --warmstart \
  --warmstart-flux-weight 0.3 \
  --warmstart-ensemble-weight 0.3 \
  --warmstart-random-weight 0.4
```

**Use Case:** DIMACS challenge instances with unknown structure.

### Balanced Configuration

Equal contribution from all sources:

```bash
prism-cli --input graph.col --warmstart \
  --warmstart-flux-weight 0.33 \
  --warmstart-ensemble-weight 0.33 \
  --warmstart-random-weight 0.34
```

## Anchor Configurations

### Low Anchor Density (Default)

Suitable for most graphs:

```bash
prism-cli --input DSJC250.5.col --warmstart \
  --warmstart-anchor-fraction 0.10
```

**Result:** ~25 anchors for DSJC250 (250 vertices × 0.10).

### High Anchor Density

For dense graphs requiring strong structural guidance:

```bash
prism-cli --input DSJC500.5.col --warmstart \
  --warmstart-anchor-fraction 0.20
```

**Result:** ~100 anchors for DSJC500 (500 vertices × 0.20).

### No Anchors

Pure probabilistic priors without deterministic assignments:

```bash
prism-cli --input graph.col --warmstart \
  --warmstart-anchor-fraction 0.0
```

**Use Case:** Maximum flexibility, highest entropy.

## Color Space Configuration

### Small Color Space

For graphs with known low chromatic number:

```bash
prism-cli --input myciel4.col --warmstart \
  --warmstart-max-colors 10
```

**Note:** χ(myciel4) = 5, so 10 colors is sufficient.

### Large Color Space

For dense graphs or unknown chromatic number:

```bash
prism-cli --input DSJC1000.9.col --warmstart \
  --warmstart-max-colors 250
```

**Tradeoff:** Higher memory usage, better coverage.

## Curriculum Profiles

### Default Heuristics

Automatic profile selection based on graph statistics:

```bash
prism-cli --input graph.col --warmstart
```

### Custom Catalog

Use a specific curriculum profile catalog:

```bash
prism-cli --input graph.col --warmstart \
  --warmstart-curriculum-path profiles/dimacs_catalog.json
```

**Catalog Format:**
```json
{
  "profiles": [
    {
      "name": "dense_hard",
      "density_min": 0.5,
      "flux_weight": 0.5,
      "ensemble_weight": 0.3,
      "random_weight": 0.2
    }
  ]
}
```

## Complete Example

Full configuration for a DIMACS challenge instance:

```bash
prism-cli \
  --input DSJC250.5.col \
  --warmstart \
  --warmstart-max-colors 50 \
  --warmstart-anchor-fraction 0.15 \
  --warmstart-flux-weight 0.5 \
  --warmstart-ensemble-weight 0.3 \
  --warmstart-random-weight 0.2 \
  --warmstart-curriculum-path profiles/dimacs_curriculum.json \
  --verbose
```

**Output:**
```
[INFO] PRISM v2 CLI - Starting
[INFO] Warmstart enabled:
[INFO]   Max colors: 50
[INFO]   Anchor fraction: 0.15
[INFO]   Flux weight: 0.50
[INFO]   Ensemble weight: 0.30
[INFO]   Random weight: 0.20
[INFO]   Curriculum catalog: profiles/dimacs_curriculum.json
[INFO] Running pipeline on graph with 250 vertices
[INFO] Warmstart plan created: mean_entropy=3.52, anchors=38
...
```

## Validation & Error Handling

### Invalid Weight Sum

```bash
prism-cli --input graph.col --warmstart \
  --warmstart-flux-weight 0.5 \
  --warmstart-ensemble-weight 0.5 \
  --warmstart-random-weight 0.3
```

**Error:**
```
Error: Warmstart weights must sum to 1.0 (got 1.300).
flux=0.50, ensemble=0.50, random=0.30
```

### Invalid Anchor Fraction

```bash
prism-cli --input graph.col --warmstart \
  --warmstart-anchor-fraction 1.5
```

**Error:**
```
Error: Warmstart anchor fraction must be in [0.0, 1.0] (got 1.500)
```

### Zero Max Colors

```bash
prism-cli --input graph.col --warmstart \
  --warmstart-max-colors 0
```

**Error:**
```
Error: Warmstart max_colors must be > 0
```

## Performance Guidelines

### Memory Usage

- Max colors = 50: ~200 bytes/vertex (50 × 4 bytes)
- Max colors = 100: ~400 bytes/vertex
- Max colors = 250: ~1KB/vertex

For 1000-vertex graphs:
- 50 colors: ~200KB prior storage
- 250 colors: ~1MB prior storage

### Anchor Fraction Impact

- **Low (0.05-0.10):** Fast warmstart, high flexibility
- **Medium (0.10-0.20):** Balanced guidance
- **High (0.20-0.30):** Strong constraints, may over-constrain

### Weight Configuration Impact

- **High flux (>0.5):** Slower Phase 0, richer priors
- **High ensemble (>0.5):** Fast warmstart, less diversity
- **High random (>0.3):** High entropy, slower convergence

## Recommended Configurations

### DIMACS Benchmark (DSJC series)

```bash
prism-cli --input DSJC250.5.col --warmstart \
  --warmstart-max-colors 50 \
  --warmstart-anchor-fraction 0.10 \
  --warmstart-flux-weight 0.5 \
  --warmstart-ensemble-weight 0.3 \
  --warmstart-random-weight 0.2
```

### Graph Coloring Competition (Sparse Graphs)

```bash
prism-cli --input sparse_graph.col --warmstart \
  --warmstart-max-colors 30 \
  --warmstart-anchor-fraction 0.05 \
  --warmstart-flux-weight 0.4 \
  --warmstart-ensemble-weight 0.4 \
  --warmstart-random-weight 0.2
```

### Random Graphs (Erdős-Rényi)

```bash
prism-cli --input random_n500_p0.5.col --warmstart \
  --warmstart-max-colors 60 \
  --warmstart-anchor-fraction 0.15 \
  --warmstart-flux-weight 0.4 \
  --warmstart-ensemble-weight 0.3 \
  --warmstart-random-weight 0.3
```

## References

- Warmstart Plan: `docs/spec/prism_gpu_plan.md` §6
- Phase 0 Implementation: `docs/phase0_ensemble_implementation.md`
- Pipeline Orchestrator: `prism-pipeline/src/orchestrator/mod.rs`
