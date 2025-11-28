# Staging Cleanup Archive

This directory contains files removed from the active PRISM v2 tree during the cleanup-playground branch cleanup operation on 2025-11-18.

These files are preserved for historical reference but are not required for development, building, or running PRISM.

## Contents

### docs/legacy_status/
Historical status reports, completion documents, and milestone tracking files. These documented the development process but are no longer needed for active development.

**Examples**: GPU_MILESTONE_COMPLETE.md, PHASE*_COMPLETE.md, *_STATUS.md files

### docs/legacy_plans/
Old implementation plans, strategies, and roadmaps that have been superseded by the current architecture in docs/spec/.

**Examples**: COMPLETE-PRCT-GPU-IMPLEMENTATION-PLAN.md, CLEANUP_ACTION_PLAN.md

### docs/architecture_review/
Architecture and design documents that may contain useful historical insights but are not part of the current documentation set.

**Examples**: ACTUAL_GPU_ARCHITECTURE.md, ARCHITECTURE_MAP.md

### artifacts/
Generated binaries, tarballs, and compiled artifacts from old releases. These can be regenerated from source.

**Examples**: baseline-v1.0/ binaries, *.tar.gz archives

### backup_files/
Miscellaneous backup files created during development.

**Examples**: *.launcher-backup, Dockerfile.fix

## Retrieval

All files in this directory remain in git history. If you need any of these files:

```bash
# Find when a file was moved
git log --follow -- staging_cleanup/docs/legacy_status/<filename>

# View file contents at a specific commit
git show <commit>:path/to/original/file

# Restore a file to the working tree
git checkout <commit> -- path/to/original/file
```

## Size Summary

- **Total archived**: ~35 MB (excluding target/)
- **Legacy docs**: ~1 MB (80+ markdown files)
- **Binary artifacts**: ~32 MB (8 binaries + 1 tarball)
- **Backup files**: ~2 MB

## Cleanup Date

**Date**: 2025-11-18
**Branch**: cleanup-playground
**Commit**: (See git log for commit hashes)
**Full backup**: /mnt/c/Users/Predator/Desktop/PRISM-v2-full-backup-20251118.tar.gz

---

**Note**: This staging area will be committed to git but excluded from active documentation indexes. Consider periodic review for permanent removal if not accessed.
