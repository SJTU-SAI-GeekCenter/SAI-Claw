"""Skill management tool for creating, patching, and deleting agent skills."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.tools.base import Tool

_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


class SkillManageTool(Tool):
    """Tool for creating, updating (patching), and deleting agent skills.

    Workspace skills live under workspace/skills/. Built-in skills cannot be deleted.
    """

    name = "skill_manage"
    description = (
        "Manage agent skills. Use to create, update (patch), or delete workspace skills. "
        "Use this after completing a complex task, fixing a tricky error, or discovering "
        "a reusable workflow."
    )

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "patch", "delete"],
                "description": "Action to perform: 'create' a new skill, 'patch' an existing one, or 'delete' a skill.",
            },
            "skill_name": {
                "type": "string",
                "description": "Name of the skill (directory name, lowercase with hyphens, e.g. 'pdf-editor').",
            },
            "content": {
                "type": "string",
                "description": (
                    "For 'create': full SKILL.md content including YAML frontmatter (name + description required). "
                    "For 'patch': the replacement text block."
                ),
            },
            "patch_target": {
                "type": "string",
                "description": "For 'patch': a unique substring in the existing SKILL.md to locate and replace.",
            },
        },
        "required": ["action", "skill_name"],
    }

    def __init__(self, workspace: Path, usage_tracker: Any | None = None):
        self.workspace = workspace
        self.skills_dir = workspace / "skills"
        self.usage_tracker = usage_tracker

    async def execute(
        self,
        action: str,
        skill_name: str,
        content: str = "",
        patch_target: str = "",
    ) -> str:
        if action == "create":
            return await self._create(skill_name, content)
        if action == "patch":
            return await self._patch(skill_name, content, patch_target)
        if action == "delete":
            return await self._delete(skill_name)
        return f"Error: Unknown action '{action}'."

    # ── create ──────────────────────────────────────────────

    async def _create(self, name: str, content: str) -> str:
        if not content.strip():
            return "Error: 'content' is required for create action."
        if not _NAME_RE.match(name):
            return (
                f"Error: Invalid skill name '{name}'. "
                "Use lowercase letters, digits, and hyphens only (e.g. 'pdf-editor')."
            )

        skill_dir = self.skills_dir / name
        if skill_dir.exists():
            return f"Error: Skill '{name}' already exists in workspace."

        # Validate YAML frontmatter
        fm_match = _FRONTMATTER_RE.match(content)
        if not fm_match:
            return "Error: SKILL.md must have YAML frontmatter (--- ... ---) at the top."
        frontmatter = fm_match.group(1)
        required_fields = ["name:", "description:"]
        missing = [f for f in required_fields if f not in frontmatter]
        if missing:
            return f"Error: SKILL.md frontmatter missing required field(s): {', '.join(missing)}"

        self.skills_dir.mkdir(parents=True, exist_ok=True)
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

        if self.usage_tracker:
            self.usage_tracker.record_create(name)

        return f"Skill '{name}' created successfully at workspace/skills/{name}/SKILL.md."

    # ── patch ───────────────────────────────────────────────

    async def _patch(self, name: str, content: str, target: str) -> str:
        if not content or not target:
            return "Error: Both 'content' and 'patch_target' are required for patch action."

        skill_file = self._find_skill_file(name)
        if skill_file is None:
            return f"Error: Skill '{name}' not found. Check workspace/skills/ and built-in skills."
        if self._is_builtin_path(skill_file):
            return f"Error: Cannot patch built-in skill '{name}'. Create a workspace skill instead."

        current = skill_file.read_text(encoding="utf-8")
        count = current.count(target)
        if count == 0:
            return f"Error: Patch target not found in '{name}/SKILL.md'."
        if count > 1:
            return (
                f"Error: Patch target appears {count} times in '{name}/SKILL.md'. "
                "Use a longer, more unique string to identify the location."
            )

        updated = current.replace(target, content, 1)
        skill_file.write_text(updated, encoding="utf-8")

        if self.usage_tracker:
            self.usage_tracker.record_patch(name)

        return f"Skill '{name}' patched successfully."

    # ── delete ──────────────────────────────────────────────

    async def _delete(self, name: str) -> str:
        builtin_path = BUILTIN_SKILLS_DIR / name / "SKILL.md"
        if builtin_path.exists():
            return f"Error: Cannot delete built-in skill '{name}'. Only workspace skills can be deleted."

        skill_dir = self.skills_dir / name
        if not skill_dir.exists():
            return f"Error: Skill '{name}' not found in workspace."

        try:
            shutil.rmtree(skill_dir)
        except OSError as e:
            return f"Error: Failed to delete skill '{name}': {e}"

        return f"Skill '{name}' deleted successfully."

    # ── helpers ─────────────────────────────────────────────

    def _find_skill_file(self, name: str) -> Path | None:
        """Find a SKILL.md for the given skill name, checking workspace first then builtin."""
        workspace_path = self.skills_dir / name / "SKILL.md"
        if workspace_path.exists():
            return workspace_path
        builtin_path = BUILTIN_SKILLS_DIR / name / "SKILL.md"
        if builtin_path.exists():
            return builtin_path
        return None

    def _is_builtin_path(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(BUILTIN_SKILLS_DIR.resolve())
            return True
        except ValueError:
            return False
