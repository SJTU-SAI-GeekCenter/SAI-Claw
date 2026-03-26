---
description: Claude Code style workflow for searching, reading, and applying precise multi-step file edits
---

Use these tools when making code changes:

1. Use `search_files` first to locate filenames, symbols, or snippets quickly.
2. Use `read_file` to inspect the exact local context before editing.
3. Use `edit_file` for one precise replacement.
4. Use `multi_edit_file` when several related edits must be applied to the same file in one step.
5. Prefer exact snippets from `read_file` when editing. If a match is ambiguous, read more context and make the edit more specific.

Recommended workflow:
- search_files → read_file → edit_file / multi_edit_file → read_file again to verify
