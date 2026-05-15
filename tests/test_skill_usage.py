"""Tests for SkillUsageTracker."""

import json
import tempfile
from pathlib import Path

import pytest

from nanobot.agent.skill_usage import SkillUsageRecord, SkillUsageTracker


@pytest.fixture
def tracker():
    tmp = Path(tempfile.mkdtemp())
    t = SkillUsageTracker(tmp)
    yield t
    import shutil
    shutil.rmtree(str(tmp))


class TestSkillUsageTracker:
    def test_record_create(self, tracker):
        tracker.record_create("test-skill")
        record = tracker.get("test-skill")
        assert record is not None
        assert record.name == "test-skill"
        assert record.use_count == 0
        assert record.view_count == 0
        assert record.patch_count == 0
        assert record.created_at is not None

    def test_record_view(self, tracker):
        tracker.record_create("test-skill")
        tracker.record_view("test-skill")
        record = tracker.get("test-skill")
        assert record is not None
        assert record.view_count == 1
        assert record.last_viewed_at is not None

    def test_record_use(self, tracker):
        tracker.record_create("test-skill")
        tracker.record_use("test-skill")
        tracker.record_use("test-skill")
        record = tracker.get("test-skill")
        assert record is not None
        assert record.use_count == 2
        assert record.last_used_at is not None

    def test_record_patch(self, tracker):
        tracker.record_create("test-skill")
        tracker.record_patch("test-skill")
        record = tracker.get("test-skill")
        assert record is not None
        assert record.patch_count == 1
        assert record.last_patched_at is not None

    def test_get_nonexistent(self, tracker):
        assert tracker.get("nonexistent") is None

    def test_list_all(self, tracker):
        tracker.record_create("skill-a")
        tracker.record_create("skill-b")
        tracker.record_create("skill-c")
        records = tracker.list_all()
        names = {r.name for r in records}
        assert names == {"skill-a", "skill-b", "skill-c"}

    def test_auto_create_on_record(self, tracker):
        """record_view/use/patch should auto-create a record if none exists."""
        tracker.record_view("new-skill")
        record = tracker.get("new-skill")
        assert record is not None
        assert record.name == "new-skill"
        assert record.view_count == 1

    def test_atomic_write(self, tracker):
        """No .tmp files should be left after writes."""
        tracker.record_create("test-skill")
        tracker.record_use("test-skill")
        tmp_files = list(tracker.usage_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_persists_to_disk(self, tracker):
        tracker.record_create("test-skill")
        path = tracker._path_for("test-skill")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["name"] == "test-skill"
        assert data["use_count"] == 0

    def test_non_existent_usage_dir(self):
        """Should handle non-existent usage dir gracefully."""
        tmp = Path(tempfile.mkdtemp())
        # Don't create skills directory — tracker should handle it
        t = SkillUsageTracker(tmp)
        records = t.list_all()
        assert records == []
        import shutil
        shutil.rmtree(str(tmp))
