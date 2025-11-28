# PRISM v2 Cleanup Inventory Report

**Date**: 2025-11-18
**Branch**: cleanup-playground
**Backup**: /mnt/c/Users/Predator/Desktop/PRISM-v2-full-backup-20251118.tar.gz (25MB)

## Executive Summary

- **Total Repository Size**: 4.0 GB
- **Target Build Artifacts**: 3.9 GB (97.5% of repo)
- **Root Markdown Files**: 177 files
- **Legacy Status Reports**: 80+ candidate files
- **Binary Artifacts**: 8 files (23.8 MB)
- **Large Files (>5MB)**: 4 files

## 1. Large Files (>5MB)

| File Path | Size | Category | Action |
|-----------|------|----------|--------|
| `./baseline-v1.0/bin/protein_structure_benchmark` | 6.2 MB | Binary | ARCHIVE |
| `./baseline-v1.0/bin/simple_dimacs_benchmark` | 6.2 MB | Binary | ARCHIVE |
| `./prism-ai-baseline-v1.0.tar.gz` | 8.3 MB | Tarball | ARCHIVE |
| `./python/gnn_training/gnn_model.onnx.data` | 5.3 MB | ML Model | KEEP (research) |

**Total Large Files Size**: 25.8 MB

## 2. Binary Artifacts

| File Path | Size | Purpose | Action |
|-----------|------|---------|--------|
| `./cuda-keyring_1.1-1_all.deb` | Unknown | CUDA Repo Key | DELETE (external) |
| `./cuda-keyring_1.1-1_all.deb.1` | Unknown | Duplicate | DELETE |
| `./baseline-v1.0/bin/federated-sim` | 1.3 MB | Old Binary | ARCHIVE |
| `./baseline-v1.0/bin/meta-flagsctl` | 4.4 MB | Old Binary | ARCHIVE |
| `./baseline-v1.0/bin/meta-ontologyctl` | 1.3 MB | Old Binary | ARCHIVE |
| `./baseline-v1.0/bin/meta-reflexive-snapshot` | 4.0 MB | Old Binary | ARCHIVE |
| `./baseline-v1.0/bin/protein_structure_benchmark` | 6.5 MB | Old Binary | ARCHIVE |
| `./baseline-v1.0/bin/simple_dimacs_benchmark` | 6.4 MB | Old Binary | ARCHIVE |

**Total Binary Size**: 23.9 MB

## 3. Legacy Documentation (Root Directory)

### Status Reports & Completion Files (HIGH PRIORITY FOR ARCHIVAL)

```
ACTIVE_INFERENCE_STATUS.md
ACTUAL_STATUS_AND_NEXT_STEPS.md
ADAPTER_IMPLEMENTATION_COMPLETE.md
AGGRESSIVE_FIXES_SUMMARY.md
BASELINE_RELEASE_SUMMARY.md
BATCH1_MIGRATION_COMPLETE.md
COMPILATION_COMPLETE_REPORT.md
COMPILATION_FIXES_SUMMARY.md
COMPLETE_CUDA_FIX_PROMPT.md
COMPLETE_CUDA_KERNEL_INVENTORY.md
COMPLETE_GPU_VERIFICATION_REPORT.md
COMPLETE_MEC_STACK_STATUS.md
CONFIG_V1.1_VERIFICATION_REPORT.md
CRITICAL_FILES_STATUS.md
CUDARC_09_MIGRATION_STATUS.md
CUDA_FINAL_MIGRATION_GUIDE.md
CUDA_MIGRATION_STATUS.md
CUDA_MIGRATION_SUMMARY.md
DIMACS_TEST_SUMMARY.md
ENHANCEMENT_SUMMARY.md
FILE_VERIFICATION_REPORT.md
FINAL_INTEGRATION_REPORT_OCT25.md
FINAL_STATUS_GPU_IMPLEMENTATION.md
FINAL_STATUS_REPORT.md
FULL_GPU_VERIFICATION_COMPLETE.md
FULL_PIPELINE_IMPLEMENTATION_SUMMARY.md
GEODESIC_VERIFICATION_COMPLETE.md
GPU_ACCELERATION_COMPLETE.md
GPU_COMPLETION_STATUS.md
GPU_FINAL_STATUS.md
GPU_KERNEL_SUMMARY.md
GPU_MILESTONE_COMPLETE.md
GPU_PHASE_COMPLETION_REPORT.md
GPU_PIPELINE_COMPLETION_STATUS.md
GPU_STATUS_SUMMARY.md
GPU_TASKS_CHECKLIST.md
GPU_VERIFICATION_COMPLETE.md
GPU_WORK_SUMMARY.md
IMMEDIATE_FIXES_SUMMARY.md
IMPLEMENTATION_ANALYSIS_ROADMAP.md
IMPLEMENTATION_STATUS.md
IMPLEMENTATION_SUMMARY.md
INTEGRATION_LAYER_COMPLETE.md
KERNEL_IMPLEMENTATION_COMPLETE.md
KERNEL_REGISTRY_COMPLETE.md
METADATA_INTEGRATION_COMPLETE.md
MIGRATION_STATUS.md
MIGRATION_SUMMARY.md
MIGRATION_VERIFICATION_REPORT.md
NEUROMORPHIC_IMPLEMENTATION_STATUS.md
NEUROMORPHIC_STATUS.md
NEXT_STEPS_IMPLEMENTATION.md
OBSERVABILITY_COMPLETE.md
PHASE1_IMPLEMENTATION_SUMMARY.md
PHASE2_COMPLETE.md
PHASE2_IMPLEMENTATION_SUMMARY.md
PHASE3_COMPLETE.md
PHASE3_IMPLEMENTATION_SUMMARY.md
PHASE4_COMPLETE.md
PHASE4_IMPLEMENTATION_SUMMARY.md
PHASE5_COMPLETE.md
PHASE5_IMPLEMENTATION_SUMMARY.md
PHASE6_COMPLETE.md
PHASE6_IMPLEMENTATION_SUMMARY.md
PHASE7_COMPLETE.md
PHASE7_IMPLEMENTATION_SUMMARY.md
PIPELINE_COMPLETION_STATUS.md
PRCT_PHASES_STATUS.md
PRISM_WORKSPACE_INTEGRATION_COMPLETE.md
PRODUCTION_READINESS_STATUS.md
QUANTUM_KERNEL_IMPLEMENTATION_COMPLETE.md
REALTIME_STRATEGY_STATUS.md
RELEASE_READY_REPORT.md
RIDGE_COMPLETE.md
TASK_VERIFICATION_COMPLETE.md
TELEMETRY_COMPLETE.md
TEST_PASS_REPORT.md
UNIVERSAL_FLUXNET_COMPLETE.md
WARMSTART_COMPLETE.md
WORKSPACE_COMPLETION_REPORT.md
WORKSPACE_INTEGRATION_COMPLETE.md
```

**Count**: 80+ files
**Estimated Total Size**: ~500-800 KB
**Action**: ARCHIVE ALL to `staging_cleanup/docs/legacy_status/`

### Implementation Plans & Strategies (MEDIUM PRIORITY)

```
BEATING_WORLD_RECORD_STRATEGY.md
COMPLETE-PRCT-GPU-IMPLEMENTATION-PLAN.md
CLEANUP_ACTION_PLAN.md
IMPLEMENTATION_ANALYSIS_ROADMAP.md
NEXT_STEPS_IMPLEMENTATION.md
REALTIME_STRATEGY_STATUS.md
```

**Action**: ARCHIVE to `staging_cleanup/docs/legacy_plans/`

### Architecture & Design Docs (REVIEW REQUIRED)

```
ACTUAL_GPU_ARCHITECTURE.md
ARCHITECTURE_MAP.md
BUILD_ORDER_VISUAL.md
CHROMATIC_BINDING_THEORY.md
CHROMATIC_DESIGN_SUMMARY.md
```

**Action**: REVIEW - May contain useful info to merge into docs/spec/

### Essential Documentation (KEEP)

```
README-UNIVERSAL.md (candidate for main README)
README_PROFILING.md (useful reference)
README_QUICK_START.md (useful reference)
```

**Action**: KEEP or consolidate into single README.md

## 4. Non-Source Files (Top 3 Levels)

| File | Category | Action |
|------|----------|--------|
| `.dockerignore` | Config | KEEP |
| `Dockerfile` | Config | KEEP |
| `Dockerfile.fix` | Legacy | ARCHIVE |
| `./cuda-keyring_1.1-1_all.deb*` | External | DELETE |
| `./benchmarks/dimacs/*.col` | Test Data | KEEP |
| `./data/nipah/*.mtx, *.pdb` | Research Data | KEEP |
| `./foundation/cma/transfer_entropy_gpu.rs.launcher-backup` | Backup | DELETE |
| `./foundation/mlir_runtime.cpp` | Source (C++) | KEEP |

## 5. Generated Artifacts

### Target Directory
- **Size**: 3.9 GB
- **Contents**: Cargo build artifacts, dependencies, PTX kernels
- **Action**: DELETE (regenerated by `cargo build`)

### Output Directory
- **Status**: Not found
- **Action**: N/A

## 6. Foundation Module Usage

Foundation modules are REFERENCED in workspace:

```toml
# Cargo.toml workspace members
"foundation/shared-types",
"foundation/prct-core",
"foundation/quantum",
"foundation/neuromorphic",
```

**Action**: KEEP - These are actively used by the workspace

## 7. Workspace Structure (CURRENT STATE)

### Active Crates
- `prism-core/` - Core types and traits
- `prism-gpu/` - GPU abstractions and kernels
- `prism-fluxnet/` - RL controller
- `prism-phases/` - Phase implementations
- `prism-pipeline/` - Orchestrator
- `prism-cli/` - CLI interface

### Supporting Directories
- `foundation/` - Legacy modules (still referenced)
- `benchmarks/` - DIMACS test graphs
- `data/` - Research datasets
- `docs/` - Current documentation
- `scripts/` - Build and deployment scripts
- `dashboards/` - Grafana configs
- `.github/workflows/` - CI/CD

## 8. Untracked Files

**Status**: None detected via `git status --porcelain`

## 9. Risk Assessment

### Safe to Archive (LOW RISK)
- All *_STATUS.md, *_COMPLETE.md files (80+ files)
- baseline-v1.0/ binaries (8 files, 23.8 MB)
- prism-ai-baseline-v1.0.tar.gz (8.3 MB)
- cuda-keyring_*.deb files (external packages)

### Safe to Delete (LOW RISK)
- target/ directory (3.9 GB, regenerated)
- *.launcher-backup files
- Duplicate .deb files

### Needs Review (MEDIUM RISK)
- Architecture docs (may have unique insights)
- Some README-*.md files (consolidation candidate)

### Must Keep (HIGH RISK)
- prism-*/ workspace crates
- foundation/ (actively referenced)
- benchmarks/ (test data)
- docs/ (current documentation)
- scripts/ (deployment tools)
- dashboards/ (monitoring)
- .github/workflows/ (CI/CD)

## 10. Recommended Actions

### Phase 1: Archive Legacy Docs (80+ files, ~1 MB)
Move all *_STATUS.md, *_COMPLETE.md, *_SUMMARY.md to `staging_cleanup/docs/legacy_status/`

### Phase 2: Archive Binaries (8 files, 32 MB)
Move baseline-v1.0/ and *.tar.gz to `staging_cleanup/artifacts/`

### Phase 3: Delete External Packages
Remove cuda-keyring_*.deb (re-downloadable)

### Phase 4: Remove Build Artifacts (3.9 GB)
Delete target/ directory entirely

### Phase 5: Consolidate READMEs
Merge README-UNIVERSAL.md, README_PROFILING.md, README_QUICK_START.md into single README.md

## 11. Expected Savings

| Category | Current Size | After Cleanup | Savings |
|----------|--------------|---------------|---------|
| target/ | 3.9 GB | 0 GB | 3.9 GB |
| Binaries | 32 MB | 0 MB | 32 MB |
| Legacy Docs | 1 MB | 0 MB | 1 MB |
| **TOTAL** | **4.0 GB** | **~60 MB** | **~3.94 GB (98.5%)** |

## 12. Next Steps

1. Create `staging_cleanup/` directory structure
2. Move files in batches with git mv
3. Run verification suite after each batch
4. Update documentation
5. Commit changes incrementally
6. Generate final cleanup summary

---

**Generated**: 2025-11-18
**Agent**: prism-architect
**Status**: Ready for execution
