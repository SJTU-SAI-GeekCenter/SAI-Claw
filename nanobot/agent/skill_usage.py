"""Skill usage tracking via .usage.json sidecar files."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class SkillUsageRecord:
    """Usage statistics for a single skill."""

    name: str
    use_count: int = 0
    view_count: int = 0
    patch_count: int = 0
    last_used_at: str | None = None
    last_viewed_at: str | None = None
    last_patched_at: str | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "use_count": self.use_count,
            "view_count": self.view_count,
            "patch_count": self.patch_count,
            "last_used_at": self.last_used_at,
            "last_viewed_at": self.last_viewed_at,
            "last_patched_at": self.last_patched_at,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillUsageRecord:
        return cls(
            name=data.get("name", ""),
            use_count=data.get("use_count", 0),
            view_count=data.get("view_count", 0),
            patch_count=data.get("patch_count", 0),
            last_used_at=data.get("last_used_at"),
            last_viewed_at=data.get("last_viewed_at"),
            last_patched_at=data.get("last_patched_at"),
            created_at=data.get("created_at"),
        )


class SkillUsageTracker:
    """Tracks skill usage statistics via .usage.json sidecar files.

    Each skill's usage data is stored in workspace/skills/.usage/{name}.json,
    separate from the SKILL.md file. Atomic writes prevent corruption.
    """

    def __init__(self, workspace: Path):
        self.usage_dir = workspace / "skills" / ".usage"
        self.usage_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, skill_name: str) -> Path:
        return self.usage_dir / f"{skill_name}.json"

    def get(self, skill_name: str) -> SkillUsageRecord | None:
        path = self._path_for(skill_name)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return SkillUsageRecord.from_dict(data)
        except (json.JSONDecodeError, OSError):
            return None

    def _atomic_write(self, path: Path, record: SkillUsageRecord) -> None:
        """Write record to a temp file then atomically replace."""
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json", prefix=".usage_", dir=str(self.usage_dir)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _now(self) -> str:
        return datetime.now().isoformat()

    def record_use(self, skill_name: str) -> None:
        record = self.get(skill_name) or SkillUsageRecord(name=skill_name)
        record.use_count += 1
        record.last_used_at = self._now()
        self._atomic_write(self._path_for(skill_name), record)

    def record_view(self, skill_name: str) -> None:
        record = self.get(skill_name) or SkillUsageRecord(name=skill_name)
        record.view_count += 1
        record.last_viewed_at = self._now()
        self._atomic_write(self._path_for(skill_name), record)

    def record_patch(self, skill_name: str) -> None:
        record = self.get(skill_name) or SkillUsageRecord(name=skill_name)
        record.patch_count += 1
        record.last_patched_at = self._now()
        self._atomic_write(self._path_for(skill_name), record)

    def record_create(self, skill_name: str) -> None:
        record = SkillUsageRecord(
            name=skill_name,
            use_count=0,
            view_count=0,
            patch_count=0,
            created_at=self._now(),
        )
        self._atomic_write(self._path_for(skill_name), record)

    def list_all(self) -> list[SkillUsageRecord]:
        records: list[SkillUsageRecord] = []
        if not self.usage_dir.exists():
            return records
        for path in sorted(self.usage_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                records.append(SkillUsageRecord.from_dict(data))
            except (json.JSONDecodeError, OSError):
                continue
        return records
