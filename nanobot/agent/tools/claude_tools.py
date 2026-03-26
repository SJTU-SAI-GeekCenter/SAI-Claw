"""Claude Code style developer tools."""

from __future__ import annotations

import fnmatch
from typing import Any

from nanobot.agent.tools.filesystem import _FsTool, _find_match


class SearchFilesTool(_FsTool):
    """Search filenames or file contents under a directory."""

    _DEFAULT_MAX_RESULTS = 50
    _IGNORE_DIRS = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".coverage",
        "htmlcov",
    }

    @property
    def name(self) -> str:
        return "search_files"

    @property
    def description(self) -> str:
        return (
            "Search for a query inside filenames or file contents under a directory. "
            "Useful before reading or editing files. "
            "Use mode='filename' for name-only search, mode='content' for text search, "
            "or mode='auto' to do both."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to search in",
                },
                "query": {
                    "type": "string",
                    "description": "Text query to search for",
                },
                "mode": {
                    "type": "string",
                    "enum": ["auto", "filename", "content"],
                    "description": "Search mode (default: auto)",
                },
                "file_glob": {
                    "type": "string",
                    "description": "Optional glob filter like '*.py' or 'nanobot/**/*.py'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matched files to return (default 50)",
                    "minimum": 1,
                },
            },
            "required": ["path", "query"],
        }

    async def execute(
        self,
        path: str,
        query: str,
        mode: str = "auto",
        file_glob: str = "*",
        max_results: int | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            dp = self._resolve(path)
            if not dp.exists():
                return f"Error: Directory not found: {path}"
            if not dp.is_dir():
                return f"Error: Not a directory: {path}"

            cap = max_results or self._DEFAULT_MAX_RESULTS
            query_lower = query.lower()
            results: list[str] = []
            total_matches = 0

            for item in sorted(dp.rglob("*")):
                if any(part in self._IGNORE_DIRS for part in item.parts):
                    continue
                if not item.is_file():
                    continue

                rel = item.relative_to(dp)
                rel_str = str(rel)

                if file_glob not in ("", "*"):
                    if not fnmatch.fnmatch(item.name, file_glob) and not fnmatch.fnmatch(rel_str, file_glob):
                        continue

                matched = False

                if mode in ("auto", "filename") and query_lower in item.name.lower():
                    matched = True
                    total_matches += 1
                    if len(results) < cap:
                        results.append(f"📄 {rel_str} (filename match)")

                if matched or mode == "filename":
                    continue

                if mode in ("auto", "content"):
                    try:
                        raw = item.read_bytes()
                        if b"\x00" in raw:
                            continue
                        text = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        continue

                    snippets: list[str] = []
                    for line_no, line in enumerate(text.splitlines(), 1):
                        if query_lower in line.lower():
                            snippet = line.strip()
                            if len(snippet) > 180:
                                snippet = snippet[:177] + "..."
                            snippets.append(f"  L{line_no}: {snippet}")
                            if len(snippets) >= 3:
                                break

                    if snippets:
                        total_matches += 1
                        if len(results) < cap:
                            results.append(f"📄 {rel_str}\n" + "\n".join(snippets))

            if total_matches == 0:
                return (
                    f"No matches found for query {query!r} under {dp}. "
                    "Try broadening the query or changing mode/file_glob."
                )

            out = "\n\n".join(results)
            if total_matches > cap:
                out += f"\n\n(truncated, showing first {cap} of {total_matches} matched files)"
            else:
                out += f"\n\n(total matched files: {total_matches})"
            return out

        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error searching files: {e}"


class MultiEditFileTool(_FsTool):
    """Apply multiple edits to one file in a single tool call."""

    @property
    def name(self) -> str:
        return "multi_edit_file"

    @property
    def description(self) -> str:
        return (
            "Apply multiple text replacements to a single file in order. "
            "Useful for Claude Code style refactors where several related edits "
            "must succeed together. Each edit uses the same matching behavior as edit_file."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to edit",
                },
                "edits": {
                    "type": "array",
                    "description": "Ordered list of text edits to apply",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_text": {
                                "type": "string",
                                "description": "The text to find",
                            },
                            "new_text": {
                                "type": "string",
                                "description": "The replacement text",
                            },
                            "replace_all": {
                                "type": "boolean",
                                "description": "Replace all exact occurrences of the matched fragment",
                            },
                        },
                        "required": ["old_text", "new_text"],
                    },
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Validate and preview edits without writing the file",
                },
            },
            "required": ["path", "edits"],
        }

    async def execute(
        self,
        path: str,
        edits: list[dict[str, Any]],
        dry_run: bool = False,
        **kwargs: Any,
    ) -> str:
        try:
            if not edits:
                return "Error: edits must contain at least one edit"

            fp = self._resolve(path)
            if not fp.exists():
                return f"Error: File not found: {path}"
            if not fp.is_file():
                return f"Error: Not a file: {path}"

            raw = fp.read_bytes()
            uses_crlf = b"\r\n" in raw
            content = raw.decode("utf-8").replace("\r\n", "\n")

            summaries: list[str] = []

            for idx, edit in enumerate(edits, 1):
                old_text = str(edit.get("old_text", "")).replace("\r\n", "\n")
                new_text = str(edit.get("new_text", "")).replace("\r\n", "\n")
                replace_all = bool(edit.get("replace_all", False))

                if not old_text:
                    return f"Error: edit #{idx} old_text cannot be empty"

                match, count = _find_match(content, old_text)
                if match is None:
                    return (
                        f"Error: edit #{idx} old_text not found in {path}. "
                        "Read the file again and provide a more exact snippet."
                    )

                if count > 1 and not replace_all:
                    return (
                        f"Warning: edit #{idx} old_text appears {count} times in {path}. "
                        "Provide a more specific snippet or set replace_all=true."
                    )

                if replace_all:
                    content = content.replace(match, new_text)
                    summaries.append(f"edit #{idx}: replaced all occurrences of the matched fragment")
                else:
                    content = content.replace(match, new_text, 1)
                    summaries.append(f"edit #{idx}: replaced 1 occurrence")

            output = content.replace("\n", "\r\n") if uses_crlf else content

            if not dry_run:
                fp.write_bytes(output.encode("utf-8"))

            prefix = "Dry run successful." if dry_run else f"Successfully edited {fp}"
            return prefix + "\n" + "\n".join(f"- {s}" for s in summaries)

        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error editing file: {e}"
