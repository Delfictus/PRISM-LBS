# Claude Code Auto-Accept Configuration

## Overview
This configuration enables automatic acceptance of file edits and other operations without user prompts.

## Configuration Location
- **File**: `.claude/settings.local.json`
- **Type**: Local settings (not committed to version control)

## Auto-Accepted Operations

The following tools are configured to run without user confirmation:

### Build & Test Operations
- `Bash(cargo test:*)` - All cargo test commands
- `Bash(cargo build:*)` - All cargo build commands

### File System Operations
- `Bash(ls:*)` - List directory contents
- `Bash(du:*)` - Check disk usage
- `Bash(awk:*)` - AWK text processing
- `Bash(wc:*)` - Word/line counting

### File Editing Operations
- `Edit` - Modify existing files
- `Write` - Create/overwrite files
- `NotebookEdit` - Edit Jupyter notebooks
- `TodoWrite` - Manage todo lists

## How It Works

Claude Code checks the `permissions` section in settings:
- **allow**: Operations that execute automatically without prompts
- **deny**: Operations that are blocked
- **ask**: Operations that require user confirmation (default for unlisted)

## Modifying Configuration

To add more auto-accepted operations:
1. Edit `.claude/settings.local.json`
2. Add tool names to the `"allow"` array
3. Use patterns like `"Bash(command:*)"` for specific bash commands
4. Save the file - changes take effect immediately

## Safety Considerations

While auto-accept improves workflow speed, consider:
- Review changes with version control before committing
- Use `deny` list for sensitive operations
- Keep backups of important files
- Monitor Claude's actions in the terminal

## Task Completion Requirements

- **Do not mark a task complete** until the feature is actually implemented, tested, and demonstrated. That means:
  - Real code (no TODOs/stubs) executing the required algorithm.
  - A CLI command or test run showing the feature working end-to-end, with command output pasted in the summary.
  - Telemetry or log snippets proving real metrics were emitted.
  - `PhaseContext` setters and telemetry emitters invoked with real values.
- If a feature cannot be finished, stop and report the blocker—do not leave scaffolding and call it done.

## Focus Guard

- Work on **one subsystem at a time** (e.g., CMA GPU kernels, MEC MD simulation). Do not switch until that subsystem is fully functional—code, telemetry, CLI mode, docs, tests.
- Each time you switch tasks, state what evidence you produced (telemetry snippet, test log). Without proof, do not move on.

## Self-Check (Answer before coding each session)

1. Did I finish the previous task with real code, telemetry, and tests? If not, why not?
2. What proof will I provide when this task is complete (CLI command output, telemetry file, test results)?
3. Am I about to mark something done without evidence? If yes, stop and fix it.

In your first message each session, explicitly answer these questions before doing any work.

## Reverting Changes

To disable auto-accept for any operation:
1. Remove the tool name from the `"allow"` array
2. Optionally add to `"ask"` array for explicit prompting
3. Or add to `"deny"` array to block completely

## Example: Disable Auto-Edit
```json
{
  "permissions": {
    "allow": [
      // Remove "Edit" from here
    ],
    "ask": [
      "Edit"  // Add here to require confirmation
    ]
  }
}
```
