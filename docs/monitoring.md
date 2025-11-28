# PRISM v2 Monitoring Guide

Comprehensive guide for monitoring PRISM GPU-accelerated graph coloring pipeline using Prometheus and Grafana.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Prometheus Metrics](#prometheus-metrics)
- [Grafana Dashboard Setup](#grafana-dashboard-setup)
- [Performance Profiling](#performance-profiling)
- [Multi-GPU Monitoring](#multi-gpu-monitoring)
- [Troubleshooting](#troubleshooting)

---

## Overview

PRISM v2 provides comprehensive telemetry infrastructure for production monitoring:

- **Prometheus Metrics**: Real-time metrics exposed via HTTP endpoint
- **Grafana Dashboards**: Pre-configured visualizations for GPU, RL, and phase metrics
- **Performance Profiler**: Detailed timing analysis for offline optimization
- **Multi-GPU Coordination**: Load balancing and device utilization tracking

### Architecture

```
┌─────────────────┐
│  PRISM Pipeline │
│   (prism-cli)   │
└────────┬────────┘
         │
         ├─> Prometheus Metrics (HTTP :9090/metrics)
         │   └─> Scraped by Prometheus Server
         │       └─> Visualized in Grafana
         │
         └─> Performance Profiler
             └─> Exported to JSON/CSV
```

---

## Quick Start

### 1. Enable Metrics Server

Run PRISM with metrics server enabled:

```bash
prism-cli \
  --input graph.txt \
  --gpu \
  --enable-metrics \
  --metrics-port 9090
```

### 2. Verify Metrics Endpoint

```bash
curl http://localhost:9090/metrics
```

Expected output:
```
# HELP prism_phase_iteration_total Total iterations executed in each phase
# TYPE prism_phase_iteration_total counter
prism_phase_iteration_total{phase="Phase0"} 150
prism_phase_iteration_total{phase="Phase2"} 500

# HELP prism_gpu_utilization GPU utilization as fraction [0.0, 1.0] per device
# TYPE prism_gpu_utilization gauge
prism_gpu_utilization{device="0"} 0.82
```

### 3. Configure Prometheus

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'prism'
    scrape_interval: 5s
    static_configs:
      - targets: ['localhost:9090']
```

Start Prometheus:

```bash
prometheus --config.file=prometheus.yml
```

### 4. Import Grafana Dashboard

1. Open Grafana (default: http://localhost:3000)
2. Navigate to **Dashboards** → **Import**
3. Upload `dashboards/prism-gpu-performance.json`
4. Select Prometheus data source
5. Click **Import**

---

## Prometheus Metrics

### Phase Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `prism_phase_iteration_total` | Counter | `phase` | Total iterations executed in each phase |
| `prism_phase_duration_seconds` | Histogram | `phase` | Phase execution duration in seconds |
| `prism_phase_temperature` | Gauge | `phase` | Current temperature value for thermodynamic phases |
| `prism_phase_compaction_ratio` | Gauge | `phase` | Compaction ratio for Phase 2 thermodynamic search |
| `prism_phase_chromatic_best` | Gauge | - | Best chromatic number observed in current phase |
| `prism_phase_conflicts` | Gauge | - | Number of conflicts in current solution |

**Example Queries:**

```promql
# Average phase duration over 5 minutes
rate(prism_phase_duration_seconds_sum[5m]) / rate(prism_phase_duration_seconds_count[5m])

# Phase iteration rate (iterations per second)
rate(prism_phase_iteration_total[1m])

# Current temperature by phase
prism_phase_temperature
```

### RL Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `prism_rl_reward` | Gauge | `phase` | Reward signal received by RL controller per phase |
| `prism_rl_q_value` | Gauge | `phase`, `action` | Q-value for (phase, action) pairs in RL controller |
| `prism_rl_epsilon` | Gauge | - | Current epsilon value for RL exploration |
| `prism_rl_action_count` | Counter | `phase`, `action` | Number of times each RL action was taken per phase |

**Example Queries:**

```promql
# RL reward trend by phase
prism_rl_reward

# Action distribution (rate over 5m)
sum(rate(prism_rl_action_count[5m])) by (action)

# Epsilon decay over time
prism_rl_epsilon
```

### GPU Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `prism_gpu_utilization` | Gauge | `device` | GPU utilization as fraction [0.0, 1.0] per device |
| `prism_gpu_memory_used_bytes` | Gauge | `device` | GPU memory used in bytes per device |
| `prism_gpu_memory_total_bytes` | Gauge | `device` | GPU memory total in bytes per device |
| `prism_gpu_kernel_duration_seconds` | Histogram | `kernel` | GPU kernel execution duration in seconds |
| `prism_gpu_kernel_launches_total` | Counter | `kernel` | Total GPU kernel launches per kernel type |

**Example Queries:**

```promql
# GPU memory utilization percentage
100 * (prism_gpu_memory_used_bytes / prism_gpu_memory_total_bytes)

# Average kernel execution time
rate(prism_gpu_kernel_duration_seconds_sum[5m]) / rate(prism_gpu_kernel_launches_total[5m])

# Kernel launch rate (launches per second)
rate(prism_gpu_kernel_launches_total[1m])
```

### Pipeline Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `prism_pipeline_solutions_found` | Counter | - | Total solutions found across all pipeline runs |
| `prism_pipeline_best_chromatic` | Gauge | - | Best chromatic number found across all runs |
| `prism_pipeline_runtime_seconds` | Histogram | - | Pipeline total runtime in seconds |

**Example Queries:**

```promql
# Solutions found over time
prism_pipeline_solutions_found

# Best chromatic number
prism_pipeline_best_chromatic

# Average pipeline runtime
rate(prism_pipeline_runtime_seconds_sum[1h]) / rate(prism_pipeline_runtime_seconds_count[1h])
```

---

## Grafana Dashboard Setup

### Dashboard Overview

The `prism-gpu-performance.json` dashboard provides 16 panels across 4 categories:

#### 1. GPU Monitoring (Panels 1-3)
- **GPU Utilization**: Time series of GPU usage per device
- **GPU Memory Usage**: Gauge showing memory utilization
- **Current Phase**: Stat panel showing active phase

#### 2. Phase Progress (Panels 4-6)
- **Best Chromatic Number**: Real-time solution quality
- **Phase Duration Heatmap**: Execution time distribution
- **Chromatic & Conflicts Trend**: Solution convergence

#### 3. RL Training (Panels 7-13)
- **Reward Signal**: Multi-phase reward trends
- **Epsilon Decay**: Exploration rate over time
- **Action Distribution**: Pie chart of action frequencies
- **Q-Values**: State-action value evolution

#### 4. Performance (Panels 8-16)
- **Kernel Performance**: Bar chart of average kernel duration
- **Kernel Launch Count**: Kernel execution frequency
- **Pipeline Runtime**: Total execution time
- **Solution Quality Trend**: Long-term improvement

### Custom Dashboards

To create custom dashboards:

1. **Navigate to Grafana** → **Dashboards** → **New Dashboard**
2. **Add Panel** → Select visualization type
3. **Configure Query**:
   - Data source: Prometheus
   - Metric: Select from `prism_*` namespace
   - Labels: Filter by phase, device, kernel, etc.
4. **Set Panel Options**:
   - Title, description, unit
   - Thresholds (green/yellow/red)
   - Legend, tooltip
5. **Save Dashboard**

---

## Performance Profiling

The performance profiler captures detailed timing and resource usage for offline analysis.

### Enable Profiling

```bash
prism-cli \
  --input graph.txt \
  --gpu \
  --enable-profiler \
  --profiler-output profile_report.json
```

### Profiler Output Formats

#### JSON Report

Comprehensive report with all statistics:

```json
{
  "total_duration_secs": 125.5,
  "phase_timings": {
    "Phase0": {
      "count": 150,
      "total_duration_secs": 22.5,
      "min_duration_secs": 0.120,
      "max_duration_secs": 0.180,
      "avg_duration_secs": 0.150,
      "std_dev_secs": 0.012
    }
  },
  "kernel_timings": {
    "floyd_warshall": {
      "count": 1000,
      "total_duration_secs": 2.5,
      "avg_duration_secs": 0.0025
    }
  },
  "memory_samples": [
    {
      "timestamp_secs": 0.0,
      "device": 0,
      "used_bytes": 2048000000,
      "total_bytes": 8192000000,
      "utilization": 0.25
    }
  ],
  "peak_memory_bytes": {
    "0": 6144000000
  },
  "total_kernel_launches": 1000,
  "total_phase_iterations": 650
}
```

#### CSV Exports

**Timing CSV** (`--profiler-output timings.csv`):

```csv
category,name,count,total_secs,min_secs,max_secs,avg_secs,std_dev_secs
phase,Phase0,150,22.5,0.120,0.180,0.150,0.012
phase,Phase2,500,75.0,0.145,0.165,0.150,0.008
kernel,floyd_warshall,1000,2.5,0.002,0.003,0.0025,0.0001
```

**Memory CSV**:

```csv
timestamp_secs,device,used_bytes,total_bytes,utilization
0.0,0,2048000000,8192000000,0.25
1.5,0,4096000000,8192000000,0.50
```

### Analyzing Profiles

#### Identify Bottlenecks

```python
import json
import pandas as pd

# Load profile report
with open('profile_report.json') as f:
    report = json.load(f)

# Find slowest phase
phase_times = {k: v['avg_duration_secs'] for k, v in report['phase_timings'].items()}
slowest_phase = max(phase_times, key=phase_times.get)
print(f"Slowest phase: {slowest_phase} ({phase_times[slowest_phase]:.3f}s avg)")

# Find slowest kernel
kernel_times = {k: v['avg_duration_secs'] for k, v in report['kernel_timings'].items()}
slowest_kernel = max(kernel_times, key=kernel_times.get)
print(f"Slowest kernel: {slowest_kernel} ({kernel_times[slowest_kernel]:.6f}s avg)")
```

#### Memory Analysis

```python
# Load memory samples
df = pd.read_csv('memory_samples.csv')

# Plot memory over time
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for device in df['device'].unique():
    device_df = df[df['device'] == device]
    plt.plot(device_df['timestamp_secs'], device_df['utilization'], label=f'Device {device}')

plt.xlabel('Time (s)')
plt.ylabel('Memory Utilization')
plt.title('GPU Memory Usage Over Time')
plt.legend()
plt.grid(True)
plt.savefig('memory_profile.png')
```

---

## Multi-GPU Monitoring

PRISM supports multi-GPU workloads with intelligent load balancing.

### Enable Multi-GPU

```bash
prism-cli \
  --input graph.txt \
  --gpu \
  --gpu-devices 0,1,2 \
  --enable-metrics \
  --metrics-port 9090
```

### Multi-GPU Metrics

All GPU metrics are labeled by device ID:

```promql
# Utilization per device
prism_gpu_utilization{device="0"}
prism_gpu_utilization{device="1"}
prism_gpu_utilization{device="2"}

# Total memory used across all devices
sum(prism_gpu_memory_used_bytes)

# Average utilization
avg(prism_gpu_utilization)

# Device imbalance (std dev of utilization)
stddev(prism_gpu_utilization)
```

### Load Balancing Policies

Configure via CLI:

```bash
# Round-robin (default)
--gpu-scheduling-policy round-robin

# Least-loaded (requires NVML)
--gpu-scheduling-policy least-loaded

# Memory-aware
--gpu-scheduling-policy memory-aware
```

**Policy Comparison:**

| Policy | Use Case | Pros | Cons |
|--------|----------|------|------|
| Round-Robin | Uniform workloads | Simple, fair, low overhead | Ignores device load |
| Least-Loaded | Variable compute intensity | Balances compute dynamically | Requires NVML polling |
| Memory-Aware | Large memory allocations | Prevents OOM | May not balance compute |

### Multi-GPU Dashboard

Create custom Grafana panel with multi-GPU comparison:

```promql
# Query for panel
prism_gpu_utilization

# Legend format
Device {{device}}
```

Panel settings:
- Visualization: Time series
- Display mode: Lines
- Legend: Table with mean, max, current
- Y-axis: 0-1 (percentage)

---

## Troubleshooting

### Metrics Endpoint Not Responding

**Symptom:** `curl http://localhost:9090/metrics` fails

**Solutions:**

1. **Check server is running:**
   ```bash
   ps aux | grep prism-cli
   ```

2. **Verify port not in use:**
   ```bash
   netstat -tuln | grep 9090
   ```

3. **Check firewall:**
   ```bash
   sudo ufw allow 9090
   ```

4. **Enable verbose logging:**
   ```bash
   prism-cli --verbose --enable-metrics --metrics-port 9090
   ```

### Missing GPU Metrics

**Symptom:** `prism_gpu_utilization` returns 0.0 or NaN

**Solutions:**

1. **Verify CUDA installation:**
   ```bash
   nvidia-smi
   ```

2. **Check NVML availability:**
   - NVML metrics require `nvidia-ml-py3` or direct NVML integration
   - Graceful degradation: returns 0.0 if unavailable

3. **Enable GPU telemetry:**
   ```bash
   prism-cli --gpu --gpu-nvml-interval 1000  # Poll every 1s
   ```

### Prometheus Not Scraping

**Symptom:** Grafana shows "No data" for all panels

**Solutions:**

1. **Verify Prometheus targets:**
   - Navigate to http://localhost:9090/targets
   - Check `prism` job status (should be "UP")

2. **Check scrape interval:**
   ```yaml
   scrape_configs:
     - job_name: 'prism'
       scrape_interval: 5s  # Increase if too aggressive
   ```

3. **Test query in Prometheus:**
   - Navigate to http://localhost:9090/graph
   - Execute: `prism_phase_iteration_total`
   - Should return results

### High Metrics Overhead

**Symptom:** Pipeline runs slower with metrics enabled

**Solutions:**

1. **Reduce scrape frequency:**
   ```yaml
   scrape_interval: 30s  # Default is 15s
   ```

2. **Disable histogram metrics:**
   - Comment out histogram metrics in `prometheus.rs`
   - Use counters/gauges only

3. **Increase NVML polling interval:**
   ```bash
   --gpu-nvml-interval 5000  # Poll every 5s instead of 1s
   ```

### Memory Profiler OOM

**Symptom:** Profiler causes out-of-memory during long runs

**Solutions:**

1. **Limit memory samples:**
   - Modify `profiler.rs` to cap `memory_samples` vector size
   - Use circular buffer or periodic flushing

2. **Export incrementally:**
   ```rust
   // In profiler loop
   if profiler.memory_sample_count() > 10000 {
       profiler.export_memory_csv("memory_batch.csv")?;
       profiler.clear_memory_samples();
   }
   ```

3. **Disable memory profiling:**
   - Only track phase/kernel timings
   - Skip `record_memory()` calls

---

## Performance Optimization Workflows

### 1. Identify Bottleneck Phase

```bash
# Run with profiler
prism-cli --input graph.txt --enable-profiler --profiler-output profile.json

# Analyze
python -c "
import json
with open('profile.json') as f:
    report = json.load(f)
    phases = report['phase_timings']
    for phase, stats in sorted(phases.items(), key=lambda x: x[1]['avg_duration_secs'], reverse=True):
        print(f'{phase}: {stats[\"avg_duration_secs\"]:.3f}s avg ({stats[\"count\"]} iterations)')
"
```

### 2. Optimize Slow Kernels

```promql
# Query kernel performance in Grafana
rate(prism_gpu_kernel_duration_seconds_sum[5m]) / rate(prism_gpu_kernel_launches_total[5m])
```

**Actions:**
- Profile kernel with `nvprof` or `nsys`
- Optimize memory access patterns
- Increase block size / occupancy
- Reduce synchronization overhead

### 3. Balance Multi-GPU Load

```promql
# Check device imbalance
stddev(prism_gpu_utilization)

# Target: < 0.1 for balanced workload
```

**Actions:**
- Switch to `least-loaded` policy
- Adjust workload granularity
- Monitor memory-bound kernels with `memory-aware` policy

### 4. Tune RL Hyperparameters

```promql
# Monitor reward signal
prism_rl_reward

# Check epsilon decay
prism_rl_epsilon
```

**Actions:**
- Adjust epsilon decay rate if exploration too low
- Increase learning rate if reward plateaus
- Analyze Q-values to identify suboptimal actions

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: PRISM Performance Regression

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v2

      - name: Build PRISM
        run: cargo build --release --features gpu

      - name: Run benchmark with profiler
        run: |
          target/release/prism-cli \
            --input benchmarks/DSJC250.5.col \
            --enable-profiler \
            --profiler-output profile_${{ github.sha }}.json

      - name: Check performance regression
        run: |
          python scripts/check_regression.py \
            profile_${{ github.sha }}.json \
            baseline_profile.json \
            --max-regression 10%

      - name: Upload profile artifact
        uses: actions/upload-artifact@v2
        with:
          name: performance-profile
          path: profile_${{ github.sha }}.json
```

---

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)
- [PRISM GPU Plan](../spec/prism_gpu_plan.md)
- [CUDA Profiling Tools](https://docs.nvidia.com/cuda/profiler-users-guide/)
