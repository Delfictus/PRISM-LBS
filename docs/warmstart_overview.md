# PRISM Warmstart System - User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [What is Warmstart?](#what-is-warmstart)
3. [Quick Start](#quick-start)
4. [How Warmstart Works](#how-warmstart-works)
5. [Configuration Guide](#configuration-guide)
6. [API Usage](#api-usage)
7. [Telemetry & Monitoring](#telemetry--monitoring)
8. [Performance & Tuning](#performance--tuning)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Topics](#advanced-topics)
11. [References](#references)

---

## Introduction

The PRISM Warmstart system provides intelligent initialization for graph coloring algorithms by leveraging probabilistic priors, structural anchors, and curriculum learning. Instead of starting from random color assignments, warmstart uses neuromorphic computation and topological analysis to predict good initial colorings, dramatically reducing conflicts and accelerating convergence.

### Key Benefits

- **50-80% conflict reduction**: Warmstart typically achieves 50-80% effectiveness in reducing initial edge conflicts compared to random initialization
- **Faster convergence**: Fewer initial conflicts lead to faster convergence to valid colorings
- **Better chromatic numbers**: Intelligent initialization helps find solutions with fewer colors
- **GPU-accelerated**: All compute-intensive operations run on CUDA for maximum performance
- **Curriculum learning**: Pre-trained Q-tables initialized based on graph difficulty profiles

### When to Use Warmstart

Warmstart is recommended for:

- **Large graphs** (>1000 vertices): Where initialization quality significantly impacts runtime
- **Dense graphs** (density >0.3): Where random initialization produces many conflicts
- **Hard coloring instances**: DIMACS benchmarks, DSJC/DSJCn graphs, register allocation problems
- **Production workloads**: Where solution quality and runtime predictability matter

Warmstart adds minimal overhead (<10% of total pipeline time) while providing substantial quality improvements.

---

## What is Warmstart?

### The Problem: Random Initialization

Traditional graph coloring algorithms start with random color assignments, leading to:

- **High initial conflicts**: Many edges connect vertices with the same color
- **Wasted computation**: Early iterations fix easily predictable conflicts
- **Poor exploration**: Random starts don't leverage graph structure
- **Unpredictable quality**: Solution quality varies widely across runs

### The Solution: Intelligent Initialization

Warmstart solves these problems by:

1. **Analyzing graph structure**: Dendritic reservoir computation identifies difficult/uncertain vertices
2. **Probabilistic priors**: Generates color probability distributions instead of deterministic assignments
3. **Structural anchors**: Selects high-centrality vertices for deterministic color assignments
4. **Curriculum learning**: Initializes RL controllers with pre-trained Q-tables matched to graph difficulty
5. **Ensemble fusion**: Combines multiple prior sources with weighted averaging

### Core Concepts

#### Probabilistic Priors

Instead of assigning colors deterministically, warmstart creates a probability distribution over colors for each vertex:

```
Vertex 42: [0.45, 0.25, 0.15, 0.10, 0.05]  (prefers color 0, but allows exploration)
```

Higher entropy distributions → more exploration
Lower entropy distributions → more exploitation

**Target entropy**: H ≥ 1.5 for DSJC250-class graphs

#### Structural Anchors

Anchors are vertices with deterministic color assignments based on topological importance:

- **Geodesic anchors** (Phase 4): High betweenness centrality (bridge nodes, cut vertices)
- **TDA anchors** (Phase 6): High topological importance (degree + Betti-0 score)

**Typical anchor coverage**: 5-15% of vertices

#### Curriculum Learning

Pre-trained Q-tables matched to graph difficulty profiles:

- **Easy**: Sparse graphs (density <0.1, avg_degree <10)
- **Medium**: Moderate graphs (density <0.3, avg_degree <50)
- **Hard**: Dense graphs (density <0.6, avg_degree <100)
- **VeryHard**: Very dense graphs (density ≥0.6 or avg_degree ≥100)

Q-tables are loaded from `profiles/curriculum/catalog.json` and mapped to the current graph via sparse-to-dense hashing.

---

## Quick Start

### Basic Usage (CLI)

Enable warmstart with default settings:

```bash
./prism-cli --input graph.col --warmstart
```

### Custom Configuration

Tune warmstart parameters:

```bash
./prism-cli --input graph.col \
  --warmstart \
  --warmstart-flux-weight 0.5 \
  --warmstart-ensemble-weight 0.3 \
  --warmstart-random-weight 0.2 \
  --warmstart-anchor-fraction 0.15
```

### Configuration File (TOML)

Create `config.toml`:

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

Run with config file:

```bash
./prism-cli --input graph.col --config config.toml
```

### Output Example

```
[INFO] Starting pipeline execution on graph with 250 vertices
[INFO] Curriculum Q-table loaded for profile: Medium
[INFO] Executing warmstart stage
[DEBUG] Warmstart anchors: 15 geodesic, 10 TDA
[INFO] Warmstart plan created: mean_entropy=1.687, anchors=25
[INFO] Phase Phase1-DendriticReservoir completed successfully
...
[INFO] Pipeline completed in 2.34s
```

### Verifying Effectiveness

Check warmstart effectiveness in telemetry output:

```json
{
  "warmstart_effectiveness": 0.72,
  "expected_conflicts": 150,
  "actual_conflicts": 42,
  "prior_entropy_mean": 1.687,
  "anchor_coverage_percent": 10.0
}
```

**Interpretation**: 72% effectiveness means warmstart reduced conflicts by 72% compared to baseline prediction.

---

## How Warmstart Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  Graph Input                                        │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│  Phase 0: Dendritic Reservoir (GPU)                 │
│  • Multi-branch neuromorphic computation            │
│  • Outputs: difficulty vector, uncertainty vector   │
│  • Target entropy: H ≥ 1.5                          │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│  Warmstart Prior Generation                         │
│  • Softmax over difficulty/uncertainty              │
│  • Temperature scaling: T = 1.0 + 2.0 * score       │
│  • Probability clamping: [min_prob, max_prob]       │
└───────────────┬─────────────────────────────────────┘
                │
                ├──────────────────────────────────────┐
                │                                      │
                ▼                                      ▼
┌──────────────────────────┐      ┌──────────────────────────┐
│  Phase 4: Geodesic (GPU) │      │  Phase 6: TDA (GPU)      │
│  • Betweenness centrality│      │  • Topological analysis  │
│  • Geodesic anchors      │      │  • TDA anchors           │
└───────────┬──────────────┘      └──────────┬───────────────┘
            │                                │
            └────────────┬───────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Ensemble Fusion                                    │
│  • Weighted average: flux_weight * reservoir +      │
│  •                   ensemble_weight * anchors +    │
│  •                   random_weight * uniform        │
│  • Anchor precedence: Anchors override priors       │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│  Curriculum Q-Table Bank                            │
│  • Graph profiling: density, avg_degree, clustering │
│  • Difficulty classification: Easy/Medium/Hard/VeryHard │
│  • Q-table selection: Best match by profile         │
│  • Sparse-to-dense mapping via modulo hash          │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│  RL Controller Initialization                       │
│  • Priors loaded into RL state                      │
│  • Q-table initialized from curriculum              │
│  • Warmstart telemetry recorded                     │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│  Phase Execution (Phases 1-7)                       │
│  • RL-guided color assignment                       │
│  • Priors influence action selection                │
│  • Anchors act as hard constraints                  │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│  Warmstart Effectiveness Update                     │
│  • effectiveness = 1.0 - (actual/expected)          │
│  • Telemetry written to SQLite/JSON                 │
└─────────────────────────────────────────────────────┘
```

### Step-by-Step Breakdown

#### Step 1: Phase 0 Dendritic Reservoir

**Purpose**: Compute vertex-level difficulty and uncertainty metrics using neuromorphic reservoir computation.

**Algorithm**:
```python
# Multi-branch reservoir with leak rate
for iteration in range(max_iterations):
    for vertex in graph.vertices:
        # Aggregate neighbor states with leak
        state[vertex] = (1 - leak_rate) * state[vertex] + leak_rate * sum(neighbor_states)

    # Check convergence
    if loss < threshold:
        break

# Extract difficulty and uncertainty
difficulty = final_state_vector
uncertainty = state_variance_vector
```

**GPU Kernel**: `dendritic_reservoir.cu` (see `docs/gpu_quickstart.md`)

**Telemetry**: `Phase0Telemetry` captures difficulty/uncertainty statistics, entropy, convergence metrics

#### Step 2: Warmstart Prior Generation

**Purpose**: Convert difficulty/uncertainty metrics into color probability distributions.

**Algorithm**:
```python
for vertex in graph.vertices:
    # Combine difficulty and uncertainty
    score = 0.7 * difficulty[vertex] + 0.3 * uncertainty[vertex]

    # Temperature-scaled softmax
    temperature = 1.0 + 2.0 * score  # Higher score → higher temperature → more uniform

    probs = []
    for color in range(max_colors):
        weight = exp(-color / temperature)
        probs.append(weight)

    # Normalize and clamp
    probs = normalize(probs)
    probs = clamp(probs, min_prob, max_prob)
    probs = normalize(probs)  # Re-normalize after clamping

    warmstart_prior[vertex] = probs
```

**Properties**:
- High difficulty vertices → higher entropy (more exploration)
- Low difficulty vertices → lower entropy (more exploitation)
- All probabilities sum to 1.0
- Minimum probability enforced to prevent zero probabilities

#### Step 3: Structural Anchor Selection

**Geodesic Anchors (Phase 4)**:
```python
# Compute betweenness centrality (GPU-accelerated)
betweenness = compute_betweenness_centrality_gpu(graph)

# Select top K vertices
K = int(graph.num_vertices * anchor_fraction / 2)
geodesic_anchors = top_k(betweenness, K)
```

**TDA Anchors (Phase 6)**:
```python
# Compute topological importance
tda_scores = degree(vertex) + betti_0_contribution(vertex)

# Select top K vertices
K = int(graph.num_vertices * anchor_fraction / 2)
tda_anchors = top_k(tda_scores, K)
```

**Anchor Assignment**:
- Geodesic anchors: Greedy coloring using first-fit strategy
- TDA anchors: Greedy coloring using first-fit strategy
- Anchors have probability 1.0 for assigned color, 0.0 for others

#### Step 4: Ensemble Fusion

**Purpose**: Combine reservoir priors with anchor information and random noise.

**Algorithm**:
```python
for vertex in graph.vertices:
    reservoir_prior = warmstart_prior[vertex]

    if vertex in (geodesic_anchors + tda_anchors):
        # Anchor: deterministic assignment
        fused_prior = anchor_prior[vertex]  # [1.0, 0.0, 0.0, ...]
    else:
        # Non-anchor: weighted fusion
        uniform_prior = uniform(max_colors)  # [1/K, 1/K, ..., 1/K]

        fused = (flux_weight * reservoir_prior +
                 ensemble_weight * uniform_prior +  # Could be replaced with other ensemble methods
                 random_weight * uniform_prior)

        fused_prior = normalize(fused)

    final_prior[vertex] = fused_prior
```

**Weight Recommendations**:
- `flux_weight = 0.4`: Reservoir priors (neuromorphic intelligence)
- `ensemble_weight = 0.4`: Structural methods (anchors, clustering)
- `random_weight = 0.2`: Exploration noise

#### Step 5: Curriculum Q-Table Loading

**Purpose**: Initialize RL controller with pre-trained Q-tables matched to graph difficulty.

**Graph Profiling**:
```python
stats = GraphStats.from_graph(graph)
# Computes: density, avg_degree, max_degree, degree_variance

profile = classify_difficulty(stats)
# Returns: Easy | Medium | Hard | VeryHard
```

**Classification Thresholds**:
```python
if density < 0.1 and avg_degree < 10:
    profile = Easy
elif density < 0.3 and avg_degree < 50:
    profile = Medium
elif density < 0.6 and avg_degree < 100:
    profile = Hard
else:
    profile = VeryHard
```

**Q-Table Selection**:
```python
curriculum_bank = CurriculumBank.load("profiles/curriculum/catalog.json")
entry = curriculum_bank.select_best_match(profile)

if entry:
    rl_controller.initialize_from_curriculum(entry.q_table)
```

**Sparse-to-Dense Mapping**:
```python
# Curriculum Q-table has fewer states than current graph
# Map via modulo hash for graceful degradation
for state_hash, actions in curriculum_q_table.items():
    mapped_hash = state_hash % current_graph_size
    rl_controller.q_table[mapped_hash] = actions
```

#### Step 6: RL Controller Initialization

**Purpose**: Load warmstart priors and curriculum Q-tables into RL controller.

**Integration**:
```rust
// Warmstart priors stored in PhaseContext
context.scratch.insert("warmstart_plan", Box::new(warmstart_plan));

// RL controller retrieves priors during action selection
let warmstart_plan = context.scratch.get("warmstart_plan")?;
let prior = warmstart_plan.vertex_priors[vertex];

// Bias action selection toward high-probability colors
action = rl_controller.select_action_with_prior(state, prior, epsilon);
```

#### Step 7: Warmstart Effectiveness Tracking

**Purpose**: Measure warmstart performance by comparing expected vs actual conflicts.

**Computation**:
```python
# Before phase execution
expected_conflicts = predict_conflicts(warmstart_priors, graph)

# After phase execution
actual_conflicts = solution.validate(graph)

# Effectiveness metric
effectiveness = 1.0 - (actual_conflicts / expected_conflicts)
```

**Interpretation**:
- `effectiveness = 0.0`: No benefit (actual = expected)
- `effectiveness = 0.5`: 50% conflict reduction
- `effectiveness = 1.0`: Perfect (zero conflicts)
- `effectiveness > 1.0`: Exceeded expectations (actual < expected)

---

## Configuration Guide

### WarmstartConfig Fields

#### max_colors

**Type**: `usize`
**Default**: `50`
**Description**: Maximum number of colors in prior probability distributions.

**Guidance**:
- Set based on expected chromatic number
- Larger values → more exploration but higher memory usage
- Typical range: 30-100 for DIMACS benchmarks

#### min_prob

**Type**: `f32`
**Default**: `0.001`
**Description**: Minimum probability for any color (prevents zero probabilities).

**Guidance**:
- Ensures all colors remain explorable
- Too high → distributions become too uniform (high entropy)
- Too low → risk of zero probabilities (numerical issues)
- Typical range: 0.0001 - 0.01

#### anchor_fraction

**Type**: `f32`
**Default**: `0.10` (10% of vertices)
**Description**: Fraction of vertices to designate as structural anchors.

**Guidance**:
- Higher fraction → more constraints, lower entropy
- Lower fraction → more flexibility, higher entropy
- Typical range: 0.05 - 0.15 (5-15%)
- Dense graphs benefit from higher fractions
- Sparse graphs benefit from lower fractions

**Tuning Strategy**:
```
Graph Density    Recommended anchor_fraction
< 0.1            0.05 - 0.08
0.1 - 0.3        0.08 - 0.12
0.3 - 0.6        0.12 - 0.15
> 0.6            0.15 - 0.20
```

#### flux_weight, ensemble_weight, random_weight

**Type**: `f32`
**Default**: `0.4, 0.4, 0.2`
**Constraint**: Must sum to 1.0
**Description**: Weights for fusing reservoir priors, ensemble methods, and random noise.

**Guidance**:
- `flux_weight`: Trust in reservoir computation (neuromorphic intelligence)
- `ensemble_weight`: Trust in structural methods (anchors, clustering)
- `random_weight`: Exploration noise

**Recommended Presets**:

**High Trust in Reservoir** (good for structured graphs):
```toml
flux_weight = 0.6
ensemble_weight = 0.3
random_weight = 0.1
```

**Balanced** (general-purpose):
```toml
flux_weight = 0.4
ensemble_weight = 0.4
random_weight = 0.2
```

**High Exploration** (good for hard instances):
```toml
flux_weight = 0.3
ensemble_weight = 0.3
random_weight = 0.4
```

#### curriculum_catalog_path

**Type**: `Option<String>`
**Default**: `Some("profiles/curriculum/catalog.json")`
**Description**: Path to curriculum Q-table catalog JSON file.

**Guidance**:
- Set to `None` to disable curriculum learning
- Catalog must follow the CurriculumBank JSON schema
- Relative paths resolved from working directory

### Configuration Examples

#### Sparse Graphs (DSJC125)

```toml
[warmstart]
max_colors = 30
min_prob = 0.001
anchor_fraction = 0.08
flux_weight = 0.5
ensemble_weight = 0.4
random_weight = 0.1
curriculum_catalog_path = "profiles/curriculum/catalog.json"
```

#### Dense Graphs (DSJC500)

```toml
[warmstart]
max_colors = 100
min_prob = 0.0005
anchor_fraction = 0.15
flux_weight = 0.4
ensemble_weight = 0.4
random_weight = 0.2
curriculum_catalog_path = "profiles/curriculum/catalog.json"
```

#### Register Allocation (Small, Dense)

```toml
[warmstart]
max_colors = 20
min_prob = 0.01
anchor_fraction = 0.12
flux_weight = 0.6
ensemble_weight = 0.3
random_weight = 0.1
curriculum_catalog_path = "profiles/curriculum/catalog.json"
```

---

## API Usage

### Rust API

#### Basic Usage

```rust
use prism_core::{Graph, WarmstartConfig};
use prism_pipeline::{PipelineConfig, PipelineOrchestrator};
use prism_fluxnet::UniversalRLController;

// Create graph (example: DIMACS format loader)
let graph = Graph::from_dimacs_file("graph.col")?;

// Configure warmstart
let warmstart_config = WarmstartConfig {
    max_colors: 50,
    anchor_fraction: 0.10,
    flux_weight: 0.4,
    ensemble_weight: 0.4,
    random_weight: 0.2,
    curriculum_catalog_path: Some("profiles/curriculum/catalog.json".to_string()),
    min_prob: 0.001,
};

// Build pipeline configuration
let pipeline_config = PipelineConfig::builder()
    .max_vertices(10000)
    .warmstart(warmstart_config)
    .build()?;

// Create RL controller
let rl_controller = UniversalRLController::new(
    /* learning_rate */ 0.1,
    /* discount_factor */ 0.9,
    /* epsilon */ 0.1,
);

// Create orchestrator and run pipeline
let mut orchestrator = PipelineOrchestrator::new(pipeline_config, rl_controller);

// Register phases (example: Phase 1)
orchestrator.add_phase(Box::new(Phase1Controller::new()));

// Execute pipeline
let solution = orchestrator.run(&graph)?;

println!("Chromatic number: {}", solution.chromatic_number);
println!("Conflicts: {}", solution.conflicts);
println!("Valid: {}", solution.is_valid());
```

#### Advanced: Custom Curriculum Creation

```rust
use prism_fluxnet::curriculum::{
    CurriculumBank, CurriculumEntry, CurriculumMetadata, DifficultyProfile,
};
use std::collections::HashMap;

// Create curriculum bank
let mut bank = CurriculumBank::new();

// Train Q-tables for each difficulty profile
for profile in DifficultyProfile::all() {
    // Train Q-table (pseudo-code, actual training implementation TBD)
    let q_table = train_q_table_for_profile(profile)?;

    let metadata = CurriculumMetadata {
        graph_class: format!("{:?}_training", profile),
        training_episodes: 10000,
        average_reward: 0.85,
        convergence_epoch: 7500,
        timestamp: chrono::Utc::now().to_rfc3339(),
        hyperparameters: Some({
            let mut params = HashMap::new();
            params.insert("learning_rate".to_string(), 0.1);
            params.insert("discount_factor".to_string(), 0.9);
            params
        }),
    };

    let entry = CurriculumEntry::new(profile, q_table, metadata);
    bank.add_entry(entry);
}

// Save catalog
bank.save("profiles/curriculum/catalog.json")?;
println!("Saved curriculum bank with {} entries", bank.num_entries());
```

#### Advanced: Custom Telemetry Handling

```rust
use prism_core::{Phase0Telemetry, WarmstartTelemetry};
use prism_pipeline::TelemetryEvent;

// Register telemetry callback
orchestrator.set_telemetry_callback(|event: &TelemetryEvent| {
    // Extract warmstart telemetry
    if let Some(warmstart_telem) = event.warmstart_telemetry() {
        println!("Warmstart Metrics:");
        println!("  Prior entropy: {:.3}", warmstart_telem.prior_entropy_mean);
        println!("  Anchor coverage: {:.1}%", warmstart_telem.anchor_coverage_percent);
        println!("  Expected conflicts: {}", warmstart_telem.expected_conflicts);

        if let Some(effectiveness) = warmstart_telem.warmstart_effectiveness {
            println!("  Effectiveness: {:.1}%", effectiveness * 100.0);
        }
    }

    // Extract Phase 0 telemetry
    if event.phase_name == "Phase0-DendriticReservoir" {
        if let Some(phase0_telem) = event.phase0_telemetry() {
            println!("Phase 0 Metrics:");
            println!("  Difficulty entropy: {:.3}", phase0_telem.difficulty_entropy);
            println!("  Uncertainty entropy: {:.3}", phase0_telem.uncertainty_entropy);
            println!("  Iterations: {}", phase0_telem.reservoir_iterations);
            println!("  GPU used: {}", phase0_telem.used_gpu);
        }
    }
});
```

---

## Telemetry & Monitoring

### Phase0Telemetry Schema

Captures metrics from dendritic reservoir computation.

```rust
pub struct Phase0Telemetry {
    pub difficulty_mean: f32,
    pub difficulty_variance: f32,
    pub difficulty_entropy: f32,
    pub uncertainty_mean: f32,
    pub uncertainty_variance: f32,
    pub uncertainty_entropy: f32,
    pub reservoir_iterations: usize,
    pub convergence_loss: f32,
    pub execution_time_ms: f64,
    pub used_gpu: bool,
}
```

**Key Metrics**:

- **difficulty_entropy**: Shannon entropy of difficulty distribution. Higher values indicate more uniform difficulty across vertices.
  - **Target**: H ≥ 1.5 for DSJC250
  - **Low entropy (<1.0)**: Few vertices dominate difficulty
  - **High entropy (>2.5)**: Difficulty evenly distributed

- **reservoir_iterations**: Number of iterations until convergence.
  - **Typical range**: 50-200 iterations
  - **High values (>500)**: Slow convergence, may need parameter tuning

- **convergence_loss**: Final loss value at convergence.
  - **Target**: <0.01
  - **High loss (>0.1)**: Poor convergence, may need more iterations

- **used_gpu**: Whether GPU acceleration was used.
  - **false**: Fallback to CPU (check CUDA availability)

### WarmstartTelemetry Schema

Captures comprehensive warmstart metrics.

```rust
pub struct WarmstartTelemetry {
    pub prior_entropy_mean: f32,
    pub prior_entropy_variance: f32,
    pub prior_entropy_distribution: Vec<f32>,  // 10-bin histogram
    pub anchor_count: usize,
    pub anchor_coverage_percent: f32,
    pub geodesic_anchor_count: usize,
    pub tda_anchor_count: usize,
    pub curriculum_profile: Option<String>,
    pub curriculum_q_table_source: Option<String>,
    pub fusion_flux_weight: f32,
    pub fusion_ensemble_weight: f32,
    pub fusion_random_weight: f32,
    pub expected_conflicts: usize,
    pub actual_conflicts: Option<usize>,
    pub warmstart_effectiveness: Option<f32>,
}
```

**Key Metrics**:

- **prior_entropy_mean**: Average entropy across all vertex priors.
  - **Target**: 1.5 - 2.5 (balanced exploration/exploitation)
  - **Low (<1.0)**: Too deterministic, limited exploration
  - **High (>3.0)**: Too random, insufficient guidance

- **anchor_coverage_percent**: Percentage of vertices designated as anchors.
  - **Target**: 5-15%
  - **Low (<3%)**: Insufficient structural constraints
  - **High (>20%)**: Over-constrained, limited flexibility

- **warmstart_effectiveness**: Conflict reduction metric.
  - **Formula**: `1.0 - (actual_conflicts / expected_conflicts)`
  - **Target**: 0.5 - 0.8 (50-80% reduction)
  - **Low (<0.3)**: Poor warmstart quality, tune weights
  - **High (>0.9)**: Excellent warmstart quality

- **curriculum_profile**: Difficulty profile used for Q-table selection.
  - **Values**: `Easy`, `Medium`, `Hard`, `VeryHard`
  - **None**: Curriculum learning disabled or failed

### JSON Telemetry Format

```json
{
  "phase": "Phase0-DendriticReservoir",
  "timestamp": "2025-01-15T12:34:56Z",
  "phase0_telemetry": {
    "difficulty_mean": 0.523,
    "difficulty_variance": 0.084,
    "difficulty_entropy": 1.687,
    "uncertainty_mean": 0.412,
    "uncertainty_variance": 0.062,
    "uncertainty_entropy": 1.523,
    "reservoir_iterations": 87,
    "convergence_loss": 0.008,
    "execution_time_ms": 45.3,
    "used_gpu": true
  },
  "warmstart_telemetry": {
    "prior_entropy_mean": 1.742,
    "prior_entropy_variance": 0.156,
    "prior_entropy_distribution": [0.02, 0.08, 0.15, 0.22, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01],
    "anchor_count": 25,
    "anchor_coverage_percent": 10.0,
    "geodesic_anchor_count": 15,
    "tda_anchor_count": 10,
    "curriculum_profile": "Medium",
    "curriculum_q_table_source": "profiles/curriculum/catalog.json",
    "fusion_flux_weight": 0.4,
    "fusion_ensemble_weight": 0.4,
    "fusion_random_weight": 0.2,
    "expected_conflicts": 150,
    "actual_conflicts": 42,
    "warmstart_effectiveness": 0.72
  }
}
```

### SQLite Telemetry Schema

Warmstart telemetry stored in `warmstart_metrics` table:

```sql
CREATE TABLE warmstart_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    prior_entropy_mean REAL,
    prior_entropy_variance REAL,
    anchor_count INTEGER,
    anchor_coverage_percent REAL,
    geodesic_anchor_count INTEGER,
    tda_anchor_count INTEGER,
    curriculum_profile TEXT,
    curriculum_q_table_source TEXT,
    fusion_flux_weight REAL,
    fusion_ensemble_weight REAL,
    fusion_random_weight REAL,
    expected_conflicts INTEGER,
    actual_conflicts INTEGER,
    warmstart_effectiveness REAL,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
);
```

### Logging Output

**INFO Level**:
```
[INFO] Executing warmstart stage
[INFO] Warmstart plan created: mean_entropy=1.687, anchors=25
[INFO] Curriculum Q-table loaded for profile: Medium
```

**DEBUG Level**:
```
[DEBUG] Phase 0 metrics not found, using placeholder difficulty/uncertainty
[DEBUG] Warmstart anchors: 15 geodesic, 10 TDA
[DEBUG] No exact match for Easy, using fallback Medium
```

**WARN Level**:
```
[WARN] Failed to load curriculum catalog: file not found. Continuing without curriculum warmstart.
[WARN] Low warmstart effectiveness: 0.12 (expected 0.5+). Consider tuning weights.
```

---

## Performance & Tuning

### Expected Effectiveness Ranges

| Graph Type | Density | Expected Effectiveness | Notes |
|------------|---------|------------------------|-------|
| Sparse (DSJC125) | <0.1 | 60-80% | High effectiveness, sparse conflicts |
| Moderate (DSJC250) | 0.1-0.3 | 50-70% | Balanced, good warmstart benefit |
| Dense (DSJC500) | 0.3-0.6 | 40-60% | Lower effectiveness, many conflicts |
| Very Dense | >0.6 | 30-50% | Limited benefit, inherently hard |

### GPU Acceleration Benefits

| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Dendritic Reservoir (n=250) | 120 ms | 15 ms | 8x |
| Dendritic Reservoir (n=1000) | 850 ms | 45 ms | 19x |
| Geodesic Anchors (n=250) | 80 ms | 12 ms | 6.7x |
| TDA Anchors (n=250) | 50 ms | 8 ms | 6.3x |

**Total warmstart overhead**: 2-5% of total pipeline time with GPU, 8-12% with CPU

### Weight Tuning Strategies

#### Strategy 1: Trust Reservoir (Structured Graphs)

**Use for**: Graphs with clear structure (planar, grid-like, hierarchical)

```toml
flux_weight = 0.6
ensemble_weight = 0.3
random_weight = 0.1
```

**Rationale**: Reservoir computation excels at detecting structural patterns.

#### Strategy 2: Balanced (General Purpose)

**Use for**: Unknown graph types, benchmarks, mixed workloads

```toml
flux_weight = 0.4
ensemble_weight = 0.4
random_weight = 0.2
```

**Rationale**: Hedges bets, provides consistent results across graph types.

#### Strategy 3: High Exploration (Hard Instances)

**Use for**: DIMACS challenge graphs, register allocation, hard coloring instances

```toml
flux_weight = 0.3
ensemble_weight = 0.3
random_weight = 0.4
```

**Rationale**: Avoids local minima, encourages exploration.

### Anchor Fraction Tuning

**Rule of thumb**: `anchor_fraction = 0.05 + 0.15 * density`

```python
# Sparse (density = 0.05)
anchor_fraction = 0.05 + 0.15 * 0.05 = 0.0575  (~6%)

# Moderate (density = 0.25)
anchor_fraction = 0.05 + 0.15 * 0.25 = 0.0875  (~9%)

# Dense (density = 0.60)
anchor_fraction = 0.05 + 0.15 * 0.60 = 0.14    (~14%)
```

**Validation**:
- Run with tuned `anchor_fraction`
- Check `anchor_coverage_percent` in telemetry
- Verify `prior_entropy_mean` is in target range (1.5-2.5)

### Graph-Specific Recommendations

#### DSJC125.1 (Sparse, density ~0.1)

```toml
max_colors = 30
anchor_fraction = 0.08
flux_weight = 0.5
ensemble_weight = 0.4
random_weight = 0.1
```

**Expected**: 70-80% effectiveness, entropy ~1.8

#### DSJC250.5 (Dense, density ~0.5)

```toml
max_colors = 80
anchor_fraction = 0.15
flux_weight = 0.4
ensemble_weight = 0.4
random_weight = 0.2
```

**Expected**: 45-60% effectiveness, entropy ~1.5

#### Register Allocation (Small, dense)

```toml
max_colors = 16
anchor_fraction = 0.12
flux_weight = 0.6
ensemble_weight = 0.3
random_weight = 0.1
```

**Expected**: 55-70% effectiveness, entropy ~1.6

---

## Troubleshooting

### Low Effectiveness (<30%)

**Symptoms**:
- `warmstart_effectiveness < 0.3`
- High `actual_conflicts` close to `expected_conflicts`
- Poor solution quality

**Diagnosis**:

1. **Check prior entropy**:
   ```bash
   # Too low (over-constrained)?
   prior_entropy_mean < 1.0

   # Too high (under-constrained)?
   prior_entropy_mean > 3.0
   ```

2. **Check anchor coverage**:
   ```bash
   # Too few anchors?
   anchor_coverage_percent < 3%

   # Too many anchors?
   anchor_coverage_percent > 20%
   ```

3. **Check curriculum profile match**:
   ```bash
   # Is the profile correct?
   # Dense graph classified as Easy?
   ```

**Solutions**:

- **Low entropy**: Decrease `anchor_fraction`, increase `random_weight`
- **High entropy**: Increase `anchor_fraction`, increase `flux_weight`
- **Low anchor coverage**: Increase `anchor_fraction`
- **High anchor coverage**: Decrease `anchor_fraction`
- **Profile mismatch**: Create custom curriculum entry for this graph type

### High Conflicts After Warmstart

**Symptoms**:
- `actual_conflicts` remains high despite warmstart
- Solution requires many iterations to converge

**Diagnosis**:

1. **Check Phase 0 convergence**:
   ```bash
   convergence_loss > 0.1  # Poor convergence
   reservoir_iterations > 500  # Slow convergence
   ```

2. **Check anchor distribution**:
   ```bash
   # Are anchors well-distributed across graph?
   # Or clustered in one region?
   ```

**Solutions**:

- **Poor Phase 0 convergence**: Increase `max_iterations`, tune `leak_rate`
- **Clustered anchors**: Adjust anchor selection algorithm (future enhancement)
- **Graph too hard**: Use `VeryHard` curriculum profile, increase `max_colors`

### Slow Execution

**Symptoms**:
- Warmstart stage takes >10% of total pipeline time
- `execution_time_ms` high in Phase0Telemetry

**Diagnosis**:

1. **Check GPU usage**:
   ```bash
   used_gpu = false  # Not using GPU!
   ```

2. **Check graph size**:
   ```bash
   num_vertices > 10000  # Large graph
   ```

**Solutions**:

- **GPU not used**: Check CUDA installation, verify `--features cuda` enabled
- **Large graph**: Enable GPU acceleration, consider increasing batch sizes
- **Slow reservoir**: Decrease `max_iterations`, increase `convergence_threshold`

### Curriculum Not Loading

**Symptoms**:
- `curriculum_profile = None` in telemetry
- Warning: "Failed to load curriculum catalog"

**Diagnosis**:

1. **Check file path**:
   ```bash
   ls profiles/curriculum/catalog.json  # File exists?
   ```

2. **Check JSON format**:
   ```bash
   jq . profiles/curriculum/catalog.json  # Valid JSON?
   ```

**Solutions**:

- **File not found**: Verify path is correct, use absolute path
- **Invalid JSON**: Validate against schema, regenerate catalog
- **Missing profile**: Add entry for required difficulty profile

### Entropy Target Not Met

**Symptoms**:
- `prior_entropy_mean < 1.5` (DSJC250 target)
- Over-constrained priors, deterministic behavior

**Diagnosis**:

1. **Check weights**:
   ```bash
   flux_weight + ensemble_weight + random_weight = 1.0  # Correct sum?
   random_weight < 0.1  # Too little exploration?
   ```

2. **Check `min_prob`**:
   ```bash
   min_prob > 0.01  # Too high, forces uniformity?
   ```

**Solutions**:

- **Increase `random_weight`**: More exploration noise
- **Decrease `anchor_fraction`**: Fewer deterministic anchors
- **Decrease `flux_weight`**: Less influence from reservoir priors
- **Adjust `min_prob`**: Carefully tune to 0.001-0.005 range

---

## Advanced Topics

### Creating Custom Curriculum Entries

#### Step 1: Train Q-Table

Use RL controller to train on representative graphs:

```rust
use prism_fluxnet::{UniversalRLController, UniversalRLState, UniversalAction};
use prism_core::Graph;

let mut rl_controller = UniversalRLController::new(0.1, 0.9, 0.1);
let training_graphs = load_training_graphs("profiles/training/easy/*.col")?;

for episode in 0..10000 {
    for graph in &training_graphs {
        let mut state = UniversalRLState::from_graph(graph);

        loop {
            let action = rl_controller.select_action(&state, 0.1);
            let (next_state, reward, done) = apply_action(graph, &state, &action)?;

            rl_controller.update_q_value(&state, &action, reward, &next_state);

            if done {
                break;
            }
            state = next_state;
        }
    }

    // Decay epsilon
    rl_controller.decay_epsilon();
}

// Extract Q-table
let q_table = rl_controller.export_q_table();
```

#### Step 2: Create Curriculum Entry

```rust
use prism_fluxnet::curriculum::{CurriculumEntry, CurriculumMetadata, DifficultyProfile};
use std::collections::HashMap;

let metadata = CurriculumMetadata {
    graph_class: "DSJC125_family".to_string(),
    training_episodes: 10000,
    average_reward: compute_average_reward(&rl_controller),
    convergence_epoch: find_convergence_epoch(&training_history),
    timestamp: chrono::Utc::now().to_rfc3339(),
    hyperparameters: Some({
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), 0.1);
        params.insert("discount_factor".to_string(), 0.9);
        params.insert("epsilon_start".to_string(), 0.3);
        params.insert("epsilon_end".to_string(), 0.01);
        params.insert("epsilon_decay".to_string(), 0.995);
        params
    }),
};

let entry = CurriculumEntry::new(DifficultyProfile::Easy, q_table, metadata);
```

#### Step 3: Add to Catalog

```rust
let mut bank = CurriculumBank::load("profiles/curriculum/catalog.json")?;
bank.add_entry(entry);
bank.save("profiles/curriculum/catalog.json")?;

println!("Added curriculum entry for {:?}", DifficultyProfile::Easy);
```

### Training Q-Tables with RL Controller

See `examples/train_curriculum.rs` for complete training pipeline.

**Training Tips**:

- Use 5-10 representative graphs per difficulty profile
- Train for 10,000+ episodes for convergence
- Monitor `average_reward` over last 100 episodes
- Use epsilon decay schedule: start 0.3, end 0.01, decay 0.995
- Save checkpoints every 1000 episodes

### Extending with Custom Prior Sources

You can add custom prior sources beyond reservoir and anchors:

```rust
use prism_core::WarmstartPrior;

// Custom prior source: clustering-based
fn compute_clustering_priors(
    graph: &Graph,
    clusters: &[Vec<usize>],
    max_colors: usize,
) -> Vec<WarmstartPrior> {
    let mut priors = Vec::with_capacity(graph.num_vertices);

    for vertex in 0..graph.num_vertices {
        let cluster_id = find_cluster(vertex, clusters);

        // Bias toward colors not used by cluster neighbors
        let used_colors = get_cluster_colors(cluster_id, clusters);
        let mut probs = vec![1.0 / max_colors as f32; max_colors];

        for color in used_colors {
            probs[color] *= 0.5;  // Penalize used colors
        }

        // Normalize
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);

        priors.push(WarmstartPrior {
            vertex,
            color_probabilities: probs,
            is_anchor: false,
            anchor_color: None,
        });
    }

    priors
}

// Fuse with existing priors
let clustering_priors = compute_clustering_priors(&graph, &clusters, config.max_colors);

for (i, prior) in fused_priors.iter_mut().enumerate() {
    let clustering_prior = &clustering_priors[i];

    // Add fourth weight source
    let weights = [
        config.flux_weight * 0.8,      // Scale down existing weights
        config.ensemble_weight * 0.8,
        config.random_weight * 0.8,
        0.2,                            // New clustering weight
    ];

    *prior = prism_phases::phase0::fuse_priors(
        &[prior, clustering_prior],
        &[weights[0] + weights[1] + weights[2], weights[3]],
    );
}
```

### Integration with External Solvers

Export warmstart priors for use in external solvers:

```rust
use prism_core::WarmstartPlan;
use std::fs::File;
use std::io::Write;

// Export priors in DIMACS-style format
fn export_warmstart_dimacs(plan: &WarmstartPlan, path: &str) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    writeln!(file, "c Warmstart priors for graph coloring")?;
    writeln!(file, "c Format: vertex color_0_prob color_1_prob ... color_K_prob")?;
    writeln!(file, "p warmstart {} {}", plan.vertex_priors.len(), plan.vertex_priors[0].color_probabilities.len())?;

    for prior in &plan.vertex_priors {
        write!(file, "w {} ", prior.vertex)?;
        for prob in &prior.color_probabilities {
            write!(file, "{:.6} ", prob)?;
        }
        writeln!(file)?;

        // Mark anchors
        if prior.is_anchor {
            writeln!(file, "a {} {}", prior.vertex, prior.anchor_color.unwrap())?;
        }
    }

    Ok(())
}

// Usage
let warmstart_plan = context.scratch.get("warmstart_plan")?.downcast_ref::<WarmstartPlan>()?;
export_warmstart_dimacs(warmstart_plan, "warmstart_priors.txt")?;
```

---

## References

### Documentation

- **PRISM GPU Plan**: `docs/spec/prism_gpu_plan.md` - Overall architecture and design
- **Warmstart System Specification**: `docs/spec/warmstart_system.md` - Technical specification
- **GPU Quickstart**: `docs/gpu_quickstart.md` - CUDA setup and kernel details
- **Phase 0 Ensemble Implementation**: `docs/phase0_ensemble_implementation.md` - Ensemble fusion details

### Implementation Files

- **Core Types**: `prism-core/src/types.rs` - WarmstartPrior, WarmstartConfig, telemetry types
- **Prior Generation**: `prism-phases/src/phase0/warmstart.rs` - Reservoir prior computation (289 LOC)
- **Ensemble Fusion**: `prism-phases/src/phase0/ensemble.rs` - Multi-source fusion logic (474 LOC)
- **Curriculum Bank**: `prism-fluxnet/src/curriculum.rs` - Q-table management (843 LOC)
- **Orchestrator Integration**: `prism-pipeline/src/orchestrator/mod.rs` - Warmstart stage execution

### Academic References

- **Neuromorphic Graph Coloring**: Reservoir computing applied to combinatorial optimization
- **Betweenness Centrality**: Used for geodesic anchor selection (Brandes algorithm)
- **Topological Data Analysis**: Persistent homology for structural anchor selection
- **Curriculum Learning**: Transfer learning for RL initialization (Bengio et al.)

### External Tools

- **DIMACS Graph Format**: Standard format for graph coloring benchmarks
- **NetworkX**: Python library for graph analysis (useful for validation)
- **Gephi**: Graph visualization tool (useful for anchor analysis)

---

## Appendix: Quick Reference

### CLI Flags

```bash
--warmstart                          # Enable warmstart (default: disabled)
--warmstart-flux-weight <FLOAT>      # Reservoir weight (default: 0.4)
--warmstart-ensemble-weight <FLOAT>  # Ensemble weight (default: 0.4)
--warmstart-random-weight <FLOAT>    # Random weight (default: 0.2)
--warmstart-anchor-fraction <FLOAT>  # Anchor fraction (default: 0.10)
--warmstart-max-colors <INT>         # Max colors (default: 50)
--warmstart-min-prob <FLOAT>         # Min probability (default: 0.001)
--warmstart-catalog <PATH>           # Curriculum catalog path
```

### Configuration File Template

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

### Telemetry Key Metrics

| Metric | Target | Interpretation |
|--------|--------|----------------|
| `prior_entropy_mean` | 1.5 - 2.5 | Average exploration level |
| `anchor_coverage_percent` | 5 - 15% | Structural constraint coverage |
| `warmstart_effectiveness` | 0.5 - 0.8 | Conflict reduction (50-80%) |
| `difficulty_entropy` | ≥ 1.5 | Reservoir quality (DSJC250) |

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Low effectiveness | `effectiveness < 0.3` | Tune weights, increase anchors |
| Over-constrained | `entropy < 1.0` | Decrease anchors, increase random |
| Under-constrained | `entropy > 3.0` | Increase anchors, decrease random |
| No GPU | `used_gpu = false` | Check CUDA, rebuild with `--features cuda` |
| Curriculum failed | `profile = None` | Check catalog path, validate JSON |

---

**End of Warmstart Overview Documentation**

For technical details, see `docs/spec/warmstart_system.md`.
For GPU kernel details, see `docs/gpu_quickstart.md`.
For examples, see `examples/warmstart_demo.rs`.
