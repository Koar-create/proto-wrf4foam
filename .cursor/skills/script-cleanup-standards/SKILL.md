---
name: script-cleanup-standards
description: Cleans up Python scripts by removing redundant statements, wrapping execution in __main__, preserving existing comments, and avoiding Chinese text in print output. Use when the user asks to refactor/clean a script, remove redundant code, or "套__main__".
disable-model-invocation: true
---

# Script cleanup standards (Python)

## Scope

Apply this skill when cleaning a Python script (especially notebooks exported to .py) and the user cares about minimal diffs and readability.

## User requirements (verbatim)

- 冗余语句识别并删除
- 套__main__（如果还没套）
- 别删注释
- 避免print中文
- 任务不需要记录到@AUTO-CHECKPOINT.md

## Workflow

### 1) Read-first

- Read the whole target file before editing.
- Identify **top-level side effects** (code that runs on import): data loading, plotting, directory creation, `print(...)`, DataFrame display expressions, etc.

### 2) Redundant statements removal

Remove only what is redundant or unused, without changing intent:

- **Unused imports**: any import not referenced after cleanup.
- **Dead variables**: assigned but never used (e.g., `OUTPUT_DIR = ...` without writes).
- **Notebook display leftovers**: bare expressions like `df.head()`, standalone `metrics`, etc.
- **Duplicate computation**: repeated calculations that can be computed once (prefer keep the first, remove later duplicates).
- **Debug-only output**: excessive prints that do not serve final output (keep essential, concise English summaries).

Keep data/logic correctness as the highest priority; do not "optimize" if it changes behavior.

### 3) Wrap execution with `__main__`

- If the file has top-level execution, move it into a `main()` function and add:

```python
if __name__ == "__main__":
    main()
```

- Leave reusable functions at module level.
- Ensure importing the module produces **no side effects** (no reads/writes/prints).

### 4) Preserve comments

- Do **not** delete existing comments.
- Do **not** rewrite comments unless required to keep them accurate after code moves.
- Keep comment positions reasonably close to the code they describe.

### 5) Avoid Chinese in `print`

- All runtime `print(...)` strings must be English only.
- If Chinese is needed for documentation, keep it in comments (allowed), not in printed output.

### 6) Validate

- Search for remaining top-level execution statements after refactor.
- Search for `print(...)` containing Chinese characters.
- Run lints/diagnostics for the edited file; fix issues introduced by the cleanup (unused imports, syntax errors, etc.).

## Quick checks (copy/paste)

- **No top-level execution**: only constants, function/class defs, and the `if __name__ == "__main__":` guard at bottom.
- **No Chinese prints**: regex search for `[\u4e00-\u9fff]` within `print(...)` arguments.
- **Comments preserved**: verify comment count/blocks are retained.

