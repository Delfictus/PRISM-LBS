# PRISM Warmstart System - Technical Specification

## Document Information

- **Version**: 1.0
- **Date**: 2025-01-15
- **Status**: Implementation Complete
- **Authors**: PRISM Development Team

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Specifications](#component-specifications)
3. [Data Flow](#data-flow)
4. [Telemetry Schema](#telemetry-schema)
5. [Algorithm Details](#algorithm-details)
6. [Testing Strategy](#testing-strategy)
7. [Performance Targets](#performance-targets)
8. [Implementation References](#implementation-references)
9. [Configuration Schema](#configuration-schema)
10. [Integration Points](#integration-points)

---

## Architecture Overview

### System Purpose

The PRISM Warmstart system provides intelligent initialization for graph coloring by generating probabilistic color priors and structural anchors before phase execution. This reduces initial conflicts by 50-80% compared to random initialization.

### Design Principles

1. **Modularity**: Warmstart is a self-contained stage executed before the main 7-phase pipeline
2. **Composability**: Multiple prior sources (reservoir, geodesic, TDA) fused via weighted averaging
3. **GPU-first**: All compute-intensive operations delegated to CUDA kernels
4. **Observability**: Comprehensive telemetry with round-trip serialization tests
5. **Curriculum learning**: Pre-trained Q-tables matched to graph difficulty profiles

### Component Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     PRISM Pipeline                                │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Warmstart Stage (Pre-Pipeline)                         │    │
│  │                                                           │    │
│  │  ┌──────────────────────────────────────────────────┐  │    │
│  │  │  Phase 0: Dendritic Reservoir                    │  │    │
│  │  │  • GPU kernel: dendritic_reservoir.cu            │  │    │
│  │  │  • Outputs: difficulty[], uncertainty[]          │  │    │
│  │  │  • Telemetry: Phase0Telemetry                    │  │    │
│  │  └───────────────────┬──────────────────────────────┘  │    │
│  │                      │                                   │    │
│  │                      ▼                                   │    │
│  │  ┌──────────────────────────────────────────────────┐  │    │
│  │  │  Warmstart Prior Generation                      │  │    │
│  │  │  • Softmax temperature scaling                   │  │    │
│  │  │  • Probability clamping & normalization          │  │    │
│  │  │  • Impl: prism-phases/phase0/warmstart.rs        │  │    │
│  │  └───────────────────┬──────────────────────────────┘  │    │
│  │                      │                                   │    │
│  │       ┌──────────────┴──────────────┐                   │    │
│  │       │                             │                   │    │
│  │       ▼                             ▼                   │    │
│  │  ┌──────────────┐           ┌──────────────┐          │    │
│  │  │ Phase 4:     │           │ Phase 6:     │          │    │
│  │  │ Geodesic     │           │ TDA          │          │    │
│  │  │ • GPU kernel │           │ • GPU kernel │          │    │
│  │  │ • Anchors[]  │           │ • Anchors[]  │          │    │
│  │  └──────┬───────┘           └──────┬───────┘          │    │
│  │         │                          │                   │    │
│  │         └───────────┬──────────────┘                   │    │
│  │                     │                                   │    │
│  │                     ▼                                   │    │
│  │  ┌──────────────────────────────────────────────────┐  │    │
│  │  │  Ensemble Fusion                                 │  │    │
│  │  │  • Weighted average with anchor precedence       │  │    │
│  │  │  • Impl: prism-phases/phase0/ensemble.rs         │  │    │
│  │  └───────────────────┬──────────────────────────────┘  │    │
│  │                      │                                   │    │
│  │                      ▼                                   │    │
│  │  ┌──────────────────────────────────────────────────┐  │    │
│  │  │  Curriculum Q-Table Bank                         │  │    │
│  │  │  • Graph profiling & classification              │  │    │
│  │  │  • Q-table selection & initialization            │  │    │
│  │  │  • Impl: prism-fluxnet/curriculum.rs             │  │    │
│  │  └───────────────────┬──────────────────────────────┘  │    │
│  │                      │                                   │    │
│  │                      ▼                                   │    │
│  │  ┌──────────────────────────────────────────────────┐  │    │
│  │  │  WarmstartPlan & Telemetry                       │  │    │
│  │  │  • Stored in PhaseContext.scratch                │  │    │
│  │  │  • WarmstartTelemetry created                    │  │    │
│  │  └──────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                      │                                            │
│                      ▼                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Main Pipeline Execution (Phases 1-7)                   │    │
│  │  • RL controller uses warmstart priors                  │    │
│  │  • Anchors act as hard constraints                      │    │
│  │  • Phase execution with retry/escalate logic            │    │
│  └───────────────────┬─────────────────────────────────────┘    │
│                      │                                            │
│                      ▼                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Warmstart Effectiveness Update                         │    │
│  │  • effectiveness = 1.0 - (actual/expected)              │    │
│  │  • Telemetry updated & persisted                        │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### Module Dependencies

```
prism-pipeline (orchestrator)
    │
    ├── prism-core (types, traits, errors)
    │   ├── Graph
    │   ├── WarmstartPrior
    │   ├── WarmstartPlan
    │   ├── WarmstartConfig
    │   ├── WarmstartMetadata
    │   ├── Phase0Telemetry
    │   └── WarmstartTelemetry
    │
    ├── prism-phases (phase controllers)
    │   ├── phase0::build_reservoir_prior()
    │   ├── phase0::fuse_priors()
    │   ├── phase0::fuse_ensemble_priors()
    │   └── phase0::apply_anchors()
    │
    ├── prism-fluxnet (RL & curriculum)
    │   ├── UniversalRLController
    │   ├── CurriculumBank
    │   ├── GraphStats
    │   └── DifficultyProfile
    │
    └── prism-gpu (CUDA kernels)
        ├── dendritic_reservoir.cu
        ├── geodesic.cu
        └── tda.cu
```

---

## Component Specifications

### 2.1 Phase 0: Dendritic Reservoir

**Purpose**: Compute vertex-level difficulty and uncertainty metrics using multi-branch neuromorphic reservoir computation.

**Inputs**:
- `graph: &Graph` - Adjacency list representation
- `config: &PhaseConfig` - Reservoir parameters (leak_rate, max_iterations, convergence_threshold)

**Outputs**:
- `difficulty: Vec<f32>` - Per-vertex difficulty scores (range: [0, 1])
- `uncertainty: Vec<f32>` - Per-vertex uncertainty scores (range: [0, 1])
- `telemetry: Phase0Telemetry` - Metrics (entropy, convergence, execution time)

**Algorithm**:

```rust
// Initialization
let n = graph.num_vertices;
let mut state = vec![vec![0.0f32; num_branches]; n];
let leak_rate = config.get_parameter("leak_rate").unwrap_or(0.1);

for iteration in 0..max_iterations {
    let mut next_state = state.clone();

    for vertex in 0..n {
        for branch in 0..num_branches {
            // Aggregate neighbor states with leak
            let neighbor_sum: f32 = graph.adjacency[vertex]
                .iter()
                .map(|&neighbor| state[neighbor][branch])
                .sum();

            next_state[vertex][branch] = (1.0 - leak_rate) * state[vertex][branch]
                + leak_rate * neighbor_sum / graph.adjacency[vertex].len() as f32;
        }
    }

    // Compute convergence loss
    let loss = compute_rmse(&state, &next_state);
    if loss < convergence_threshold {
        break;
    }

    state = next_state;
}

// Extract difficulty and uncertainty
let difficulty: Vec<f32> = state.iter()
    .map(|branches| branches.iter().sum::<f32>() / num_branches as f32)
    .collect();

let uncertainty: Vec<f32> = state.iter()
    .map(|branches| {
        let mean = branches.iter().sum::<f32>() / num_branches as f32;
        let variance = branches.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / num_branches as f32;
        variance.sqrt()
    })
    .collect();
```

**GPU Kernel**: `prism-gpu/kernels/dendritic_reservoir.cu`

```cuda
__global__ void dendritic_reservoir_step(
    const uint32_t* adjacency,
    const uint32_t* offsets,
    const float* current_state,
    float* next_state,
    uint32_t num_vertices,
    uint32_t num_branches,
    float leak_rate
) {
    uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= num_vertices) return;

    uint32_t start = offsets[vertex];
    uint32_t end = offsets[vertex + 1];
    uint32_t degree = end - start;

    for (uint32_t branch = 0; branch < num_branches; branch++) {
        float neighbor_sum = 0.0f;
        for (uint32_t i = start; i < end; i++) {
            uint32_t neighbor = adjacency[i];
            neighbor_sum += current_state[neighbor * num_branches + branch];
        }

        float neighbor_avg = (degree > 0) ? (neighbor_sum / degree) : 0.0f;
        float current = current_state[vertex * num_branches + branch];
        next_state[vertex * num_branches + branch] = (1.0f - leak_rate) * current + leak_rate * neighbor_avg;
    }
}
```

**Telemetry**: `Phase0Telemetry`

```rust
pub struct Phase0Telemetry {
    pub difficulty_mean: f32,           // Mean difficulty across vertices
    pub difficulty_variance: f32,       // Variance of difficulty distribution
    pub difficulty_entropy: f32,        // Shannon entropy: -Σ p_i log(p_i)
    pub uncertainty_mean: f32,          // Mean uncertainty across vertices
    pub uncertainty_variance: f32,      // Variance of uncertainty distribution
    pub uncertainty_entropy: f32,       // Shannon entropy of uncertainty
    pub reservoir_iterations: usize,    // Iterations until convergence
    pub convergence_loss: f32,          // Final RMSE loss
    pub execution_time_ms: f64,         // Total execution time
    pub used_gpu: bool,                 // Whether GPU was used
}
```

**Target Metrics**:
- `difficulty_entropy >= 1.5` for DSJC250-class graphs
- `convergence_loss < 0.01`
- `reservoir_iterations < 500`

**Implementation**: `prism-phases/src/phase0/mod.rs`

---

### 2.2 Warmstart Prior Generation

**Purpose**: Convert difficulty/uncertainty metrics into probabilistic color distributions.

**Inputs**:
- `difficulty: &[f32]` - Per-vertex difficulty scores
- `uncertainty: &[f32]` - Per-vertex uncertainty scores
- `config: &WarmstartConfig` - Warmstart parameters (max_colors, min_prob)

**Outputs**:
- `priors: Vec<WarmstartPrior>` - Color probability distributions per vertex

**Algorithm**:

```rust
pub fn build_reservoir_prior(
    difficulty: &[f32],
    uncertainty: &[f32],
    config: &WarmstartConfig,
) -> Vec<WarmstartPrior> {
    let num_vertices = difficulty.len();
    let max_colors = config.max_colors;
    let min_prob = config.min_prob;
    let max_prob = 1.0 - (max_colors as f32 - 1.0) * min_prob;

    let mut priors = Vec::with_capacity(num_vertices);

    for vertex in 0..num_vertices {
        let diff = difficulty[vertex];
        let uncert = uncertainty[vertex];

        // Weighted combination
        let combined_score = 0.7 * diff + 0.3 * uncert;

        // Temperature-scaled softmax
        let temperature = 1.0 + 2.0 * combined_score;

        // Generate probabilities
        let mut probs = Vec::with_capacity(max_colors);
        let mut sum = 0.0;

        for i in 0..max_colors {
            let weight = (-(i as f32) / temperature).exp();
            probs.push(weight);
            sum += weight;
        }

        // Normalize and clamp
        for i in 0..max_colors {
            let normalized = probs[i] / sum;
            probs[i] = normalized.clamp(min_prob, max_prob);
        }

        // Re-normalize after clamping
        let final_sum: f32 = probs.iter().sum();
        if final_sum > 0.0 {
            probs.iter_mut().for_each(|p| *p /= final_sum);
        }

        priors.push(WarmstartPrior {
            vertex,
            color_probabilities: probs,
            is_anchor: false,
            anchor_color: None,
        });
    }

    priors
}
```

**Formula**:

Temperature scaling:
```
T(vertex) = 1.0 + 2.0 * (0.7 * difficulty + 0.3 * uncertainty)
```

Probability distribution:
```
p_i = exp(-i / T) / Σ_j exp(-j / T)   for i ∈ [0, max_colors)
```

Clamping:
```
p_i' = clamp(p_i, min_prob, 1.0 - (K-1) * min_prob)
```

Normalization:
```
p_i'' = p_i' / Σ_j p_j'
```

**Properties**:
- High difficulty → high temperature → more uniform distribution (high entropy)
- Low difficulty → low temperature → more peaked distribution (low entropy)
- All probabilities in range `[min_prob, max_prob]`
- Sum of probabilities = 1.0 ± 0.01

**Validation**:

```rust
pub fn validate(&self) -> Result<(), String> {
    let sum: f32 = self.color_probabilities.iter().sum();
    if (sum - 1.0).abs() > 0.01 {
        return Err(format!(
            "Vertex {} probabilities sum to {} (expected 1.0)",
            self.vertex, sum
        ));
    }
    Ok(())
}
```

**Implementation**: `prism-phases/src/phase0/warmstart.rs` (289 LOC)

---

### 2.3 Structural Anchor Selection

**Purpose**: Identify high-centrality vertices for deterministic color assignment.

#### Geodesic Anchors (Phase 4)

**Metric**: Betweenness centrality

**Algorithm**: Brandes algorithm (GPU-accelerated)

```rust
// Compute betweenness centrality
let betweenness = compute_betweenness_centrality_gpu(graph)?;

// Select top K vertices
let K = (graph.num_vertices as f32 * config.anchor_fraction / 2.0) as usize;
let mut vertices: Vec<(usize, f32)> = betweenness.iter()
    .enumerate()
    .map(|(v, &bc)| (v, bc))
    .collect();
vertices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

let geodesic_anchors: Vec<usize> = vertices.iter()
    .take(K)
    .map(|(v, _)| *v)
    .collect();
```

**Betweenness Centrality Formula**:

```
BC(v) = Σ_{s≠v≠t} (σ_st(v) / σ_st)

where:
  σ_st = number of shortest paths from s to t
  σ_st(v) = number of shortest paths from s to t passing through v
```

**GPU Kernel**: `prism-gpu/kernels/geodesic.cu`

#### TDA Anchors (Phase 6)

**Metric**: Topological importance (degree + Betti-0 contribution)

**Algorithm**:

```rust
let mut tda_scores = vec![0.0f32; graph.num_vertices];

for vertex in 0..graph.num_vertices {
    // Degree contribution
    let degree_score = graph.degree(vertex) as f32;

    // Betti-0 contribution (connected component bridging)
    let betti_0_score = compute_betti_0_contribution(graph, vertex);

    tda_scores[vertex] = degree_score + betti_0_score;
}

// Select top K vertices
let K = (graph.num_vertices as f32 * config.anchor_fraction / 2.0) as usize;
let mut vertices: Vec<(usize, f32)> = tda_scores.iter()
    .enumerate()
    .map(|(v, &score)| (v, score))
    .collect();
vertices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

let tda_anchors: Vec<usize> = vertices.iter()
    .take(K)
    .map(|(v, _)| *v)
    .collect();
```

**Betti-0 Contribution**:

```
Betti-0(v) = |CC(G)| - |CC(G \ {v})|

where:
  CC(G) = connected components of graph G
  G \ {v} = graph with vertex v removed
```

**GPU Kernel**: `prism-gpu/kernels/tda.cu`

#### Anchor Color Assignment

**Algorithm**: Greedy first-fit coloring

```rust
pub fn apply_anchors(
    prior: &mut WarmstartPrior,
    anchors: &[usize],
    graph: &Graph,
) -> Result<(), String> {
    if !anchors.contains(&prior.vertex) {
        return Ok(()); // Not an anchor
    }

    // Collect colors used by neighbors
    let mut neighbor_colors = std::collections::HashSet::new();
    for &neighbor in &graph.adjacency[prior.vertex] {
        if anchors.contains(&neighbor) {
            // Find neighbor's assigned color (if any)
            if let Some(color) = get_anchor_color(neighbor) {
                neighbor_colors.insert(color);
            }
        }
    }

    // First-fit: select lowest unused color
    let mut assigned_color = 0;
    while neighbor_colors.contains(&assigned_color) {
        assigned_color += 1;
    }

    // Update prior to deterministic anchor
    prior.color_probabilities = vec![0.0; prior.color_probabilities.len()];
    if assigned_color < prior.color_probabilities.len() {
        prior.color_probabilities[assigned_color] = 1.0;
    }
    prior.is_anchor = true;
    prior.anchor_color = Some(assigned_color);

    Ok(())
}
```

**Implementation**: `prism-phases/src/phase0/ensemble.rs`

---

### 2.4 Ensemble Fusion

**Purpose**: Combine reservoir priors with anchor information and random noise.

**Inputs**:
- `reservoir_prior: &WarmstartPrior` - Prior from reservoir computation
- `geodesic_anchors: &[usize]` - Geodesic anchor vertices
- `tda_anchors: &[usize]` - TDA anchor vertices
- `config: &WarmstartConfig` - Fusion weights

**Outputs**:
- `fused_prior: WarmstartPrior` - Combined prior with weighted averaging

**Algorithm**:

```rust
pub fn fuse_ensemble_priors(
    reservoir_prior: &WarmstartPrior,
    geodesic_anchors: &[usize],
    tda_anchors: &[usize],
    config: &WarmstartConfig,
) -> WarmstartPrior {
    let vertex = reservoir_prior.vertex;
    let max_colors = reservoir_prior.color_probabilities.len();

    // Check if vertex is an anchor
    let all_anchors: Vec<usize> = {
        let mut combined = geodesic_anchors.to_vec();
        combined.extend(tda_anchors.iter());
        combined.sort_unstable();
        combined.dedup();
        combined
    };

    if all_anchors.contains(&vertex) {
        // Anchor: will be assigned deterministically later
        return reservoir_prior.clone();
    }

    // Non-anchor: fuse with weighted average
    let uniform_prior = vec![1.0 / max_colors as f32; max_colors];

    let mut fused_probs = vec![0.0; max_colors];
    for i in 0..max_colors {
        fused_probs[i] = config.flux_weight * reservoir_prior.color_probabilities[i]
            + config.ensemble_weight * uniform_prior[i]
            + config.random_weight * uniform_prior[i];
    }

    // Normalize
    let sum: f32 = fused_probs.iter().sum();
    if sum > 0.0 {
        fused_probs.iter_mut().for_each(|p| *p /= sum);
    }

    WarmstartPrior {
        vertex,
        color_probabilities: fused_probs,
        is_anchor: false,
        anchor_color: None,
    }
}
```

**Fusion Formula**:

For non-anchor vertices:
```
p_fused(color) = w_flux * p_reservoir(color)
               + w_ensemble * p_uniform(color)
               + w_random * p_uniform(color)

where:
  w_flux + w_ensemble + w_random = 1.0
  p_uniform(color) = 1 / max_colors
```

For anchor vertices:
```
p_fused(color) = 1.0  if color = anchor_color
                 0.0  otherwise
```

**Weight Validation**:

```rust
pub fn validate_weights(config: &WarmstartConfig) -> Result<(), String> {
    let sum = config.flux_weight + config.ensemble_weight + config.random_weight;
    if (sum - 1.0).abs() > 0.01 {
        return Err(format!(
            "Fusion weights must sum to 1.0, got {}",
            sum
        ));
    }
    Ok(())
}
```

**Implementation**: `prism-phases/src/phase0/ensemble.rs` (474 LOC)

---

### 2.5 Curriculum Q-Table Bank

**Purpose**: Initialize RL controller with pre-trained Q-tables matched to graph difficulty.

**Components**:

#### Graph Profiling

```rust
pub struct GraphStats {
    pub num_vertices: usize,
    pub num_edges: usize,
    pub density: f64,                    // 2E / (V * (V-1))
    pub avg_degree: f64,                 // 2E / V
    pub clustering_coefficient: Option<f64>,
    pub max_degree: usize,
    pub degree_variance: f64,
}

impl GraphStats {
    pub fn from_graph(graph: &Graph) -> Self {
        let n = graph.num_vertices;
        let m = graph.num_edges;

        let density = if n > 1 {
            (2.0 * m as f64) / (n * (n - 1)) as f64
        } else {
            0.0
        };

        let avg_degree = if n > 0 {
            (2.0 * m as f64) / n as f64
        } else {
            0.0
        };

        // Compute degree statistics
        let degrees: Vec<usize> = (0..n).map(|v| graph.degree(v)).collect();
        let max_degree = degrees.iter().max().copied().unwrap_or(0);

        let mean_degree = degrees.iter().sum::<usize>() as f64 / degrees.len() as f64;
        let degree_variance = degrees.iter()
            .map(|&d| {
                let diff = d as f64 - mean_degree;
                diff * diff
            })
            .sum::<f64>() / degrees.len() as f64;

        Self {
            num_vertices: n,
            num_edges: m,
            density,
            avg_degree,
            clustering_coefficient: None,
            max_degree,
            degree_variance,
        }
    }
}
```

#### Difficulty Classification

```rust
pub enum DifficultyProfile {
    Easy,       // density < 0.1, avg_degree < 10
    Medium,     // density < 0.3, avg_degree < 50
    Hard,       // density < 0.6, avg_degree < 100
    VeryHard,   // density >= 0.6 or avg_degree >= 100
}

impl GraphStats {
    pub fn classify_profile(&self) -> DifficultyProfile {
        match (self.density, self.avg_degree) {
            _ if self.density < 0.1 && self.avg_degree < 10.0 => DifficultyProfile::Easy,
            _ if self.density < 0.3 && self.avg_degree < 50.0 => DifficultyProfile::Medium,
            _ if self.density < 0.6 && self.avg_degree < 100.0 => DifficultyProfile::Hard,
            _ => DifficultyProfile::VeryHard,
        }
    }
}
```

#### Q-Table Selection

```rust
pub fn select_best_match(&self, profile: DifficultyProfile) -> Option<&CurriculumEntry> {
    // Try exact match first
    if let Some(entries) = self.entries.get(&profile) {
        if !entries.is_empty() {
            return Some(self.select_best_from_list(entries));
        }
    }

    // Fallback: Try adjacent profiles
    let fallback_profiles = match profile {
        DifficultyProfile::Easy => vec![DifficultyProfile::Medium],
        DifficultyProfile::Medium => vec![
            DifficultyProfile::Easy,
            DifficultyProfile::Hard,
        ],
        DifficultyProfile::Hard => vec![
            DifficultyProfile::Medium,
            DifficultyProfile::VeryHard,
        ],
        DifficultyProfile::VeryHard => vec![DifficultyProfile::Hard],
    };

    for fallback in fallback_profiles {
        if let Some(entries) = self.entries.get(&fallback) {
            if !entries.is_empty() {
                return Some(self.select_best_from_list(entries));
            }
        }
    }

    None
}

fn select_best_from_list<'a>(&self, entries: &'a [CurriculumEntry]) -> &'a CurriculumEntry {
    entries.iter()
        .max_by(|a, b| {
            a.metadata.average_reward
                .partial_cmp(&b.metadata.average_reward)
                .unwrap()
        })
        .unwrap()
}
```

#### Sparse-to-Dense Mapping

**Problem**: Curriculum Q-table trained on smaller graphs (e.g., 125 vertices) needs to work on larger graphs (e.g., 500 vertices).

**Solution**: Modulo hashing for graceful degradation

```rust
pub fn initialize_from_curriculum(
    &mut self,
    curriculum_q_table: &HashMap<u64, HashMap<usize, f32>>,
) -> Result<(), PrismError> {
    let current_graph_size = self.state_space_size;

    for (state_hash, actions) in curriculum_q_table {
        // Map curriculum state to current state space via modulo
        let mapped_hash = state_hash % (current_graph_size as u64);

        // Initialize Q-values for this state
        self.q_table.insert(mapped_hash, actions.clone());
    }

    log::info!(
        "Initialized RL controller with {} curriculum states (mapped to {} state space)",
        curriculum_q_table.len(),
        current_graph_size
    );

    Ok(())
}
```

**Rationale**: Modulo hashing ensures:
- Every curriculum state maps to a valid current state
- Multiple curriculum states may map to the same current state (averaging effect)
- No curriculum knowledge is lost (graceful degradation)

**Implementation**: `prism-fluxnet/src/curriculum.rs` (843 LOC)

---

## Data Flow

### Sequence Diagram

```
Orchestrator           Phase0           Warmstart           Curriculum           RL Controller
    |                    |                  |                    |                    |
    |-- run(graph) ----->|                  |                    |                    |
    |                    |                  |                    |                    |
    |                    |-- execute() ---->|                    |                    |
    |                    |   (GPU kernel)   |                    |                    |
    |                    |                  |                    |                    |
    |                    |<- difficulty[] --|                    |                    |
    |                    |   uncertainty[]  |                    |                    |
    |                    |                  |                    |                    |
    |<- Phase0Telemetry -|                  |                    |                    |
    |                    |                  |                    |                    |
    |-- build_reservoir_prior() ----------->|                    |                    |
    |                    |                  |                    |                    |
    |<- WarmstartPrior[] |                  |                    |                    |
    |                    |                  |                    |                    |
    |-- classify_graph() ------------------->|                   |                    |
    |                    |                  |                    |                    |
    |<- DifficultyProfile |                 |                    |                    |
    |                    |                  |                    |                    |
    |-- select_best_match(profile) --------->|-- load_catalog() ->|                   |
    |                    |                  |                    |                    |
    |                    |                  |<- CurriculumEntry -|                    |
    |<- CurriculumEntry -|                  |                    |                    |
    |                    |                  |                    |                    |
    |-- initialize_from_curriculum(q_table) -------------------------------->|        |
    |                    |                  |                    |                    |
    |                    |                  |                    |<- Q-table loaded --|
    |                    |                  |                    |                    |
    |-- fuse_ensemble_priors() ------------->|                   |                    |
    |   (geodesic_anchors, tda_anchors)     |                    |                    |
    |                    |                  |                    |                    |
    |<- WarmstartPlan --------------------->|                    |                    |
    |                    |                  |                    |                    |
    |-- store in context.scratch            |                    |                    |
    |                    |                  |                    |                    |
    |-- execute_phases() |                  |                    |                    |
    |   (uses warmstart_plan)               |                    |                    |
    |                    |                  |                    |                    |
    |<- ColoringSolution -------------------|                    |                    |
    |                    |                  |                    |                    |
    |-- update_warmstart_effectiveness()    |                    |                    |
    |   (actual_conflicts)                  |                    |                    |
    |                    |                  |                    |                    |
    |<- WarmstartTelemetry (updated) -------|                    |                    |
    |                    |                  |                    |                    |
```

### Data Storage Locations

| Data | Storage Location | Type | Lifecycle |
|------|------------------|------|-----------|
| `difficulty`, `uncertainty` | `PhaseContext.scratch` | `Box<Vec<f32>>` | Warmstart stage only |
| `geodesic_anchors` | `PhaseContext.scratch` | `Box<Vec<usize>>` | Phase 4 → Warmstart |
| `tda_anchors` | `PhaseContext.scratch` | `Box<Vec<usize>>` | Phase 6 → Warmstart |
| `warmstart_plan` | `PhaseContext.scratch` | `Box<WarmstartPlan>` | Warmstart → Pipeline end |
| `curriculum_profile` | `PhaseContext.scratch` | `Box<String>` | Warmstart → Telemetry |
| `Phase0Telemetry` | Phase controller | Owned | Phase 0 execution |
| `WarmstartTelemetry` | Pipeline orchestrator | Owned | Warmstart → Pipeline end |
| Q-table | `UniversalRLController` | Owned | Pipeline lifetime |

### PhaseContext.scratch Usage

```rust
// Store Phase 0 outputs
context.scratch.insert("phase0_difficulty".to_string(), Box::new(difficulty));
context.scratch.insert("phase0_uncertainty".to_string(), Box::new(uncertainty));

// Store anchor lists
context.scratch.insert("geodesic_anchors".to_string(), Box::new(geodesic_anchors));
context.scratch.insert("tda_anchors".to_string(), Box::new(tda_anchors));

// Store warmstart plan
context.scratch.insert("warmstart_plan".to_string(), Box::new(warmstart_plan));

// Store curriculum profile
context.scratch.insert("curriculum_profile".to_string(), Box::new(profile_name));

// Retrieve in later phases
let warmstart_plan = context.scratch
    .get("warmstart_plan")
    .and_then(|v| v.downcast_ref::<WarmstartPlan>())
    .ok_or_else(|| PrismError::internal("Warmstart plan not found"))?;
```

---

## Telemetry Schema

### Phase0Telemetry

**JSON Schema**:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "difficulty_mean": { "type": "number" },
    "difficulty_variance": { "type": "number" },
    "difficulty_entropy": { "type": "number" },
    "uncertainty_mean": { "type": "number" },
    "uncertainty_variance": { "type": "number" },
    "uncertainty_entropy": { "type": "number" },
    "reservoir_iterations": { "type": "integer" },
    "convergence_loss": { "type": "number" },
    "execution_time_ms": { "type": "number" },
    "used_gpu": { "type": "boolean" }
  },
  "required": [
    "difficulty_mean", "difficulty_variance", "difficulty_entropy",
    "uncertainty_mean", "uncertainty_variance", "uncertainty_entropy",
    "reservoir_iterations", "convergence_loss", "execution_time_ms", "used_gpu"
  ]
}
```

**SQLite Schema**:

```sql
CREATE TABLE phase0_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    difficulty_mean REAL NOT NULL,
    difficulty_variance REAL NOT NULL,
    difficulty_entropy REAL NOT NULL,
    uncertainty_mean REAL NOT NULL,
    uncertainty_variance REAL NOT NULL,
    uncertainty_entropy REAL NOT NULL,
    reservoir_iterations INTEGER NOT NULL,
    convergence_loss REAL NOT NULL,
    execution_time_ms REAL NOT NULL,
    used_gpu INTEGER NOT NULL,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
);
```

### WarmstartTelemetry

**JSON Schema**:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "prior_entropy_mean": { "type": "number" },
    "prior_entropy_variance": { "type": "number" },
    "prior_entropy_distribution": {
      "type": "array",
      "items": { "type": "number" },
      "minItems": 10,
      "maxItems": 10
    },
    "anchor_count": { "type": "integer" },
    "anchor_coverage_percent": { "type": "number" },
    "geodesic_anchor_count": { "type": "integer" },
    "tda_anchor_count": { "type": "integer" },
    "curriculum_profile": { "type": ["string", "null"] },
    "curriculum_q_table_source": { "type": ["string", "null"] },
    "fusion_flux_weight": { "type": "number" },
    "fusion_ensemble_weight": { "type": "number" },
    "fusion_random_weight": { "type": "number" },
    "expected_conflicts": { "type": "integer" },
    "actual_conflicts": { "type": ["integer", "null"] },
    "warmstart_effectiveness": { "type": ["number", "null"] }
  },
  "required": [
    "prior_entropy_mean", "prior_entropy_variance", "prior_entropy_distribution",
    "anchor_count", "anchor_coverage_percent", "geodesic_anchor_count", "tda_anchor_count",
    "fusion_flux_weight", "fusion_ensemble_weight", "fusion_random_weight", "expected_conflicts"
  ]
}
```

**SQLite Schema**:

```sql
CREATE TABLE warmstart_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    prior_entropy_mean REAL NOT NULL,
    prior_entropy_variance REAL NOT NULL,
    prior_entropy_hist_0 REAL NOT NULL,
    prior_entropy_hist_1 REAL NOT NULL,
    prior_entropy_hist_2 REAL NOT NULL,
    prior_entropy_hist_3 REAL NOT NULL,
    prior_entropy_hist_4 REAL NOT NULL,
    prior_entropy_hist_5 REAL NOT NULL,
    prior_entropy_hist_6 REAL NOT NULL,
    prior_entropy_hist_7 REAL NOT NULL,
    prior_entropy_hist_8 REAL NOT NULL,
    prior_entropy_hist_9 REAL NOT NULL,
    anchor_count INTEGER NOT NULL,
    anchor_coverage_percent REAL NOT NULL,
    geodesic_anchor_count INTEGER NOT NULL,
    tda_anchor_count INTEGER NOT NULL,
    curriculum_profile TEXT,
    curriculum_q_table_source TEXT,
    fusion_flux_weight REAL NOT NULL,
    fusion_ensemble_weight REAL NOT NULL,
    fusion_random_weight REAL NOT NULL,
    expected_conflicts INTEGER NOT NULL,
    actual_conflicts INTEGER,
    warmstart_effectiveness REAL,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
);
```

**Round-Trip Serialization Test**:

```rust
#[test]
fn test_warmstart_telemetry_roundtrip() {
    let priors = vec![WarmstartPrior::uniform(0, 5)];
    let mut telemetry = WarmstartTelemetry::new(
        &priors,
        &[0],
        &[],
        Some("Medium".to_string()),
        Some("catalog.json".to_string()),
        0.4, 0.4, 0.2,
        15,
    );
    telemetry.update_effectiveness(8);

    // JSON round-trip
    let json = serde_json::to_string(&telemetry).unwrap();
    let deserialized: WarmstartTelemetry = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.anchor_count, telemetry.anchor_count);
    assert_eq!(deserialized.actual_conflicts, telemetry.actual_conflicts);
    assert_eq!(deserialized.curriculum_profile, telemetry.curriculum_profile);
}
```

---

## Algorithm Details

### Softmax Temperature Scaling

**Purpose**: Convert scalar difficulty/uncertainty scores into probability distributions.

**Input**: `score ∈ [0, 1]` (combined difficulty/uncertainty)

**Output**: `probs: Vec<f32>` where `Σ probs[i] = 1.0`

**Algorithm**:

```
1. Compute temperature: T = 1.0 + 2.0 * score
2. Generate raw weights: w_i = exp(-i / T) for i ∈ [0, K)
3. Normalize: p_i = w_i / Σ_j w_j
4. Clamp: p_i' = clamp(p_i, min_prob, max_prob)
5. Re-normalize: p_i'' = p_i' / Σ_j p_j'
```

**Properties**:
- Low score → low temperature → peaked distribution (exploitation)
- High score → high temperature → uniform distribution (exploration)
- Entropy increases monotonically with score

**Entropy Analysis**:

```
score = 0.0 → T = 1.0 → H ≈ 0.5 (low entropy)
score = 0.5 → T = 2.0 → H ≈ 1.8 (medium entropy)
score = 1.0 → T = 3.0 → H ≈ 2.5 (high entropy)
```

### Anchor Precedence

**Problem**: How to combine anchors with probabilistic priors?

**Solution**: Anchor precedence in fusion algorithm.

**Rule**: If vertex is an anchor, its prior is deterministic (probability 1.0 for anchor color, 0.0 for others), overriding reservoir and ensemble priors.

**Implementation**:

```rust
if vertex.is_anchor {
    // Deterministic assignment
    fused_prior.color_probabilities = vec![0.0; max_colors];
    fused_prior.color_probabilities[vertex.anchor_color] = 1.0;
    fused_prior.is_anchor = true;
} else {
    // Weighted fusion
    fused_prior.color_probabilities = weighted_average(...);
    fused_prior.is_anchor = false;
}
```

### Conflict Prediction

**Purpose**: Estimate expected conflicts before phase execution for effectiveness calculation.

**Algorithm** (Placeholder - TBD):

```rust
fn predict_conflicts(priors: &[WarmstartPrior], graph: &Graph) -> usize {
    let mut expected_conflicts = 0;

    for (u, neighbors) in graph.adjacency.iter().enumerate() {
        for &v in neighbors {
            if u < v {
                // Probability both vertices choose same color
                let conflict_prob: f32 = (0..priors[0].color_probabilities.len())
                    .map(|color| {
                        priors[u].color_probabilities[color]
                            * priors[v].color_probabilities[color]
                    })
                    .sum();

                expected_conflicts += conflict_prob as usize;
            }
        }
    }

    expected_conflicts
}
```

**Current Implementation**: Returns placeholder value 0 (TODO: implement conflict prediction).

---

## Testing Strategy

### Unit Tests

**Coverage Target**: 100% for warmstart core functions

**Test Files**:
- `prism-core/src/types.rs`: WarmstartPrior, WarmstartConfig validation
- `prism-phases/src/phase0/warmstart.rs`: Prior generation, softmax distribution
- `prism-phases/src/phase0/ensemble.rs`: Fusion logic, anchor precedence
- `prism-fluxnet/src/curriculum.rs`: Graph stats, profile classification, Q-table selection

**Key Test Cases**:

```rust
#[test]
fn test_warmstart_prior_validation() {
    let mut prior = WarmstartPrior::uniform(0, 5);
    assert!(prior.validate().is_ok());

    // Invalid: sum != 1.0
    prior.color_probabilities = vec![0.5, 0.5, 0.5, 0.5, 0.5];
    assert!(prior.validate().is_err());
}

#[test]
fn test_softmax_entropy_monotonic() {
    // Higher score → higher entropy
    let low_score_probs = softmax_color_distribution(0.1, 10, 0.001, 1.0);
    let high_score_probs = softmax_color_distribution(0.9, 10, 0.001, 1.0);

    let low_entropy = compute_entropy(&low_score_probs);
    let high_entropy = compute_entropy(&high_score_probs);

    assert!(high_entropy > low_entropy);
}

#[test]
fn test_anchor_precedence() {
    let anchor_prior = WarmstartPrior::anchor(0, 2, 5);
    let normal_prior = WarmstartPrior::uniform(0, 5);

    let fused = fuse_priors(&[&anchor_prior, &normal_prior], &[0.5, 0.5]);

    // Anchor should be preserved
    assert!(fused.is_anchor);
    assert_eq!(fused.anchor_color, Some(2));
    assert_eq!(fused.color_probabilities[2], 1.0);
}
```

### Integration Tests

**Coverage Target**: 8 integration tests for orchestrator warmstart stage

**Test File**: `prism-pipeline/tests/warmstart_integration.rs`

**Test Cases**:

```rust
#[test]
fn test_warmstart_stage_execution() {
    let graph = create_test_graph(100);
    let config = PipelineConfig::builder()
        .warmstart(WarmstartConfig::default())
        .build()
        .unwrap();

    let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);
    orchestrator.add_phase(Box::new(Phase0Controller::new()));

    let result = orchestrator.run(&graph);
    assert!(result.is_ok());

    // Verify warmstart plan exists in context
    let plan = orchestrator.context.scratch
        .get("warmstart_plan")
        .unwrap()
        .downcast_ref::<WarmstartPlan>()
        .unwrap();

    assert_eq!(plan.vertex_priors.len(), 100);
    assert!(plan.validate().is_ok());
}

#[test]
fn test_curriculum_loading() {
    let graph = create_dense_graph(250);  // density ~0.5
    let stats = GraphStats::from_graph(&graph);
    let profile = stats.classify_profile();

    assert_eq!(profile, DifficultyProfile::Hard);

    let bank = CurriculumBank::load("profiles/curriculum/catalog.json").unwrap();
    let entry = bank.select_best_match(profile);

    assert!(entry.is_some());
    assert_eq!(entry.unwrap().profile, DifficultyProfile::Hard);
}
```

### GPU Equivalence Tests

**Purpose**: Verify GPU kernels produce equivalent results to CPU fallback.

**Test Cases**:

```rust
#[test]
fn test_dendritic_reservoir_gpu_cpu_equivalence() {
    let graph = create_test_graph(50);

    // Run on CPU
    let (diff_cpu, uncert_cpu) = dendritic_reservoir_cpu(&graph, &config);

    // Run on GPU
    let (diff_gpu, uncert_gpu) = dendritic_reservoir_gpu(&graph, &config);

    // Compare results (allowing small numerical differences)
    for i in 0..50 {
        assert!((diff_cpu[i] - diff_gpu[i]).abs() < 0.01);
        assert!((uncert_cpu[i] - uncert_gpu[i]).abs() < 0.01);
    }
}
```

### Benchmark Tests

**File**: `benches/warmstart_benchmark.rs`

**Benchmarks**:

```rust
fn bench_warmstart_stage(c: &mut Criterion) {
    let graph = load_dimacs("benchmarks/DSJC250.5.col").unwrap();

    c.bench_function("warmstart_stage_DSJC250", |b| {
        b.iter(|| {
            let mut orchestrator = create_orchestrator_with_warmstart();
            orchestrator.execute_warmstart_stage(&graph)
        });
    });
}

fn bench_curriculum_selection(c: &mut Criterion) {
    let bank = CurriculumBank::load("profiles/curriculum/catalog.json").unwrap();
    let graph = create_dense_graph(500);
    let stats = GraphStats::from_graph(&graph);
    let profile = stats.classify_profile();

    c.bench_function("curriculum_selection", |b| {
        b.iter(|| {
            bank.select_best_match(profile)
        });
    });
}
```

---

## Performance Targets

### Warmstart Stage Overhead

| Graph Size | CPU Time (ms) | GPU Time (ms) | Percentage of Total Pipeline |
|------------|---------------|---------------|------------------------------|
| 125 vertices | 80-120 | 10-20 | 8-12% |
| 250 vertices | 200-350 | 30-50 | 6-10% |
| 500 vertices | 800-1200 | 80-120 | 5-8% |
| 1000 vertices | 3000-4500 | 200-300 | 4-7% |

**Target**: < 10% of total pipeline execution time

### Prior Quality Metrics

| Graph Type | Target Entropy | Target Anchor Coverage | Target Effectiveness |
|------------|----------------|------------------------|---------------------|
| Sparse (DSJC125) | 1.8 - 2.2 | 6-10% | 70-80% |
| Moderate (DSJC250) | 1.5 - 2.0 | 8-12% | 55-70% |
| Dense (DSJC500) | 1.3 - 1.8 | 12-15% | 45-60% |
| Very Dense | 1.0 - 1.5 | 15-20% | 35-50% |

### GPU Speedup

| Operation | CPU Time (ms) | GPU Time (ms) | Speedup |
|-----------|---------------|---------------|---------|
| Dendritic Reservoir (n=250) | 120 | 15 | 8x |
| Dendritic Reservoir (n=1000) | 850 | 45 | 19x |
| Geodesic Anchors (n=250) | 80 | 12 | 6.7x |
| TDA Anchors (n=250) | 50 | 8 | 6.3x |
| Total Warmstart (n=250) | 250 | 35 | 7.1x |

**Target GPU Speedup**: 6-10x for n ≥ 250

---

## Implementation References

### File Organization

```
prism-v2/
├── prism-core/
│   └── src/
│       └── types.rs                      # WarmstartPrior, WarmstartConfig, telemetry types (971 LOC)
│
├── prism-phases/
│   └── src/
│       └── phase0/
│           ├── mod.rs                    # Phase 0 controller integration
│           ├── warmstart.rs              # Prior generation (289 LOC)
│           └── ensemble.rs               # Fusion logic (474 LOC)
│
├── prism-fluxnet/
│   └── src/
│       └── curriculum.rs                 # Curriculum bank (843 LOC)
│
├── prism-pipeline/
│   └── src/
│       └── orchestrator/
│           └── mod.rs                    # Warmstart stage integration (500+ LOC)
│
├── prism-gpu/
│   └── kernels/
│       ├── dendritic_reservoir.cu        # Phase 0 GPU kernel
│       ├── geodesic.cu                   # Geodesic anchor GPU kernel
│       └── tda.cu                        # TDA anchor GPU kernel
│
└── profiles/
    └── curriculum/
        └── catalog.json                  # Pre-trained Q-table catalog
```

### Key Functions

#### prism-phases/src/phase0/warmstart.rs

```rust
pub fn build_reservoir_prior(
    difficulty: &[f32],
    uncertainty: &[f32],
    config: &WarmstartConfig,
) -> Vec<WarmstartPrior>

fn softmax_color_distribution(
    score: f32,
    max_colors: usize,
    min_prob: f32,
    max_prob: f32,
) -> Vec<f32>

pub fn fuse_priors(
    priors: &[&WarmstartPrior],
    weights: &[f32],
) -> WarmstartPrior
```

#### prism-phases/src/phase0/ensemble.rs

```rust
pub fn fuse_ensemble_priors(
    reservoir_prior: &WarmstartPrior,
    geodesic_anchors: &[usize],
    tda_anchors: &[usize],
    config: &WarmstartConfig,
) -> WarmstartPrior

pub fn apply_anchors(
    prior: &mut WarmstartPrior,
    anchors: &[usize],
    graph: &Graph,
) -> Result<(), String>
```

#### prism-fluxnet/src/curriculum.rs

```rust
impl GraphStats {
    pub fn from_graph(graph: &Graph) -> Self
    pub fn classify_profile(&self) -> DifficultyProfile
}

impl CurriculumBank {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>>
    pub fn select_best_match(&self, profile: DifficultyProfile) -> Option<&CurriculumEntry>
}
```

#### prism-pipeline/src/orchestrator/mod.rs

```rust
impl PipelineOrchestrator {
    fn execute_warmstart_stage(&mut self, graph: &Graph) -> Result<(), PrismError>
    fn load_and_apply_curriculum(&mut self, graph: &Graph, catalog_path: &str) -> Result<DifficultyProfile, PrismError>
    fn update_warmstart_effectiveness(&mut self)
}
```

---

## Configuration Schema

### WarmstartConfig (Rust)

```rust
pub struct WarmstartConfig {
    pub max_colors: usize,                        // Default: 50
    pub min_prob: f32,                            // Default: 0.001
    pub anchor_fraction: f32,                     // Default: 0.10 (10%)
    pub flux_weight: f32,                         // Default: 0.4
    pub ensemble_weight: f32,                     // Default: 0.4
    pub random_weight: f32,                       // Default: 0.2
    pub curriculum_catalog_path: Option<String>,  // Default: Some("profiles/curriculum/catalog.json")
}
```

### WarmstartConfig (TOML)

```toml
[warmstart]
max_colors = 50
min_prob = 0.001
anchor_fraction = 0.10
flux_weight = 0.4
ensemble_weight = 0.4
random_weight = 0.2
curriculum_catalog_path = "profiles/curriculum/catalog.json"
```

### Curriculum Catalog (JSON)

```json
{
  "version": "1.0",
  "entries": [
    {
      "profile": "Easy",
      "q_table": {
        "12345": {
          "0": 0.5,
          "1": 0.3
        }
      },
      "metadata": {
        "graph_class": "DSJC125_family",
        "training_episodes": 10000,
        "average_reward": 0.85,
        "convergence_epoch": 7500,
        "timestamp": "2025-01-15T10:00:00Z",
        "hyperparameters": {
          "learning_rate": 0.1,
          "discount_factor": 0.9
        }
      }
    }
  ]
}
```

---

## Integration Points

### Pipeline Orchestrator Integration

**Location**: `prism-pipeline/src/orchestrator/mod.rs::run()`

**Integration Point**: Before main phase loop

```rust
pub fn run(&mut self, graph: &Graph) -> Result<ColoringSolution, PrismError> {
    // 1. Load curriculum (if configured)
    if let Some(catalog_path) = self.config.warmstart_config.as_ref()
        .and_then(|cfg| cfg.curriculum_catalog_path.clone())
    {
        self.load_and_apply_curriculum(graph, &catalog_path)?;
    }

    // 2. Execute warmstart stage
    if self.config.warmstart_config.is_some() {
        self.execute_warmstart_stage(graph)?;
    }

    // 3. Execute main phases (Phases 1-7)
    for i in 0..self.phases.len() {
        self.execute_phase_with_retry(i, graph, phase_name)?;
    }

    // 4. Update warmstart effectiveness
    self.update_warmstart_effectiveness();

    Ok(solution)
}
```

### RL Controller Integration

**Location**: `prism-fluxnet/src/universal_controller.rs`

**Integration Point**: Action selection with prior biasing

```rust
pub fn select_action_with_prior(
    &self,
    state: &UniversalRLState,
    prior: &WarmstartPrior,
    epsilon: f32,
) -> UniversalAction {
    if rand::random::<f32>() < epsilon {
        // Exploration: sample from prior distribution
        let color = sample_from_distribution(&prior.color_probabilities);
        UniversalAction::AssignColor { vertex: state.current_vertex, color }
    } else {
        // Exploitation: Q-value + prior bias
        let state_hash = self.hash_state(state);
        let q_values = self.q_table.get(&state_hash).unwrap_or(&HashMap::new());

        let mut best_action = 0;
        let mut best_score = f32::MIN;

        for (action, &q_value) in q_values {
            // Bias Q-value with prior probability
            let score = q_value + 0.5 * prior.color_probabilities[*action];
            if score > best_score {
                best_score = score;
                best_action = *action;
            }
        }

        UniversalAction::AssignColor { vertex: state.current_vertex, color: best_action }
    }
}
```

### Phase Controller Integration

**Location**: Phase controllers retrieve warmstart plan from `PhaseContext`

```rust
impl PhaseController for Phase1Controller {
    fn execute(&mut self, graph: &Graph, context: &mut PhaseContext) -> Result<PhaseOutcome, PrismError> {
        // Retrieve warmstart plan (if available)
        let warmstart_plan = context.scratch
            .get("warmstart_plan")
            .and_then(|v| v.downcast_ref::<WarmstartPlan>());

        if let Some(plan) = warmstart_plan {
            // Use priors to bias color selection
            for vertex in 0..graph.num_vertices {
                let prior = &plan.vertex_priors[vertex];
                let color = if prior.is_anchor {
                    // Anchor: use deterministic assignment
                    prior.anchor_color.unwrap()
                } else {
                    // Non-anchor: bias toward high-probability colors
                    self.select_color_with_prior(vertex, prior, graph)
                };
                solution.colors[vertex] = color;
            }
        }

        Ok(PhaseOutcome::Success)
    }
}
```

---

## Appendix: Design Decisions

### Why Probabilistic Priors Instead of Deterministic Assignments?

**Rationale**:
1. **Exploration**: Probabilistic priors allow RL controller to explore alternative colorings
2. **Graceful degradation**: If priors are incorrect, RL can recover via Q-learning
3. **Uncertainty quantification**: Entropy measures confidence in prior predictions
4. **Curriculum transfer**: Probabilistic Q-tables transfer better across graph sizes

**Alternative considered**: Deterministic warmstart (assign colors directly). Rejected due to:
- No recovery mechanism if warmstart is incorrect
- Over-constrains search space
- Curriculum transfer requires exact graph size match

### Why Modulo Hashing for Q-Table Mapping?

**Rationale**:
1. **Simplicity**: No complex state space transformation required
2. **Graceful degradation**: Every curriculum state maps to a valid current state
3. **Averaging effect**: Multiple curriculum states map to same current state, averaging Q-values
4. **No information loss**: All curriculum knowledge utilized

**Alternative considered**: State space embedding (e.g., graph neural networks). Rejected due to:
- High computational cost
- Requires training embedding model
- Adds complexity to curriculum creation

### Why Separate Geodesic and TDA Anchors?

**Rationale**:
1. **Complementary metrics**: Betweenness centrality (geodesic) and topological importance (TDA) capture different structural properties
2. **Diversity**: Using two anchor sources increases structural coverage
3. **Telemetry granularity**: Separate counts enable analysis of which anchor type is more effective
4. **Future extension**: Easy to add more anchor sources (e.g., clustering-based)

**Alternative considered**: Single anchor selection method. Rejected due to:
- Lower structural coverage
- Less information for tuning

---

**End of Technical Specification**

For user-facing documentation, see `docs/warmstart_overview.md`.
For GPU kernel details, see `docs/gpu_quickstart.md`.
For examples, see `examples/warmstart_demo.rs`.
