"""Tests for SkillManageTool."""

import tempfile
from pathlib import Path

import pytest

from nanobot.agent.skill_usage import SkillUsageTracker
from nanobot.agent.tools.skill_manage import SkillManageTool


@pytest.fixture
def tool():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "skills").mkdir()
    tracker = SkillUsageTracker(tmp)
    t = SkillManageTool(workspace=tmp, usage_tracker=tracker)
    yield t
    import shutil
    shutil.rmtree(str(tmp))


def _create_skill_content(name="test-skill"):
    return (
        f"---\nname: {name}\ndescription: A test skill\n---\n\n"
        f"# Test Skill\n\nThis is a test.\n\n## Section A\n\nHello World.\n"
    )


class TestSkillManageCreate:
    @pytest.mark.asyncio
    async def test_create_simple(self, tool):
        content = _create_skill_content()
        r = await tool.execute("create", "test-skill", content)
        assert "created successfully" in r
        skill_file = tool.workspace / "skills" / "test-skill" / "SKILL.md"
        assert skill_file.exists()

    @pytest.mark.asyncio
    async def test_create_invalid_name(self, tool):
        r = await tool.execute("create", "Bad Name", "---\nname: x\n---\n")
        assert "Invalid skill name" in r

    @pytest.mark.asyncio
    async def test_create_empty_name(self, tool):
        r = await tool.execute("create", "-bad", _create_skill_content())
        assert "Invalid skill name" in r

    @pytest.mark.asyncio
    async def test_create_missing_frontmatter(self, tool):
        r = await tool.execute("create", "test-skill", "# No frontmatter\n\nJust body.")
        assert "YAML frontmatter" in r

    @pytest.mark.asyncio
    async def test_create_missing_description(self, tool):
        r = await tool.execute("create", "test-skill", "---\nname: test-skill\n---\nbody")
        assert "missing required field" in r

    @pytest.mark.asyncio
    async def test_create_duplicate(self, tool):
        content = _create_skill_content()
        await tool.execute("create", "test-skill", content)
        r = await tool.execute("create", "test-skill", content)
        assert "already exists" in r

    @pytest.mark.asyncio
    async def test_create_tracks_usage(self, tool):
        content = _create_skill_content()
        await tool.execute("create", "test-skill", content)
        rec = tool.usage_tracker.get("test-skill")
        assert rec is not None
        assert rec.created_at is not None


class TestSkillManagePatch:
    @pytest.mark.asyncio
    async def test_patch_simple(self, tool):
        content = _create_skill_content()
        await tool.execute("create", "test-skill", content)
        # execute(action, skill_name, content=replacement, patch_target=find_this)
        r = await tool.execute("patch", "test-skill",
                               content="World.", patch_target="Hello World.")
        assert "patched successfully" in r

        skill_file = tool.workspace / "skills" / "test-skill" / "SKILL.md"
        patched = skill_file.read_text(encoding="utf-8")
        assert "World." in patched
        assert "Hello World." not in patched

    @pytest.mark.asyncio
    async def test_patch_skill_not_found(self, tool):
        r = await tool.execute("patch", "nonexistent",
                               content="X", patch_target="Y")
        assert "not found" in r

    @pytest.mark.asyncio
    async def test_patch_target_not_found(self, tool):
        content = _create_skill_content()
        await tool.execute("create", "test-skill", content)
        r = await tool.execute("patch", "test-skill",
                               content="replacement", patch_target="ZZZ_not_there_ZZZ")
        assert "not found" in r

    @pytest.mark.asyncio
    async def test_patch_non_unique_target(self, tool):
        content = _create_skill_content() + "\n\n## Notes\n\nNOTE-MARKER\n\nNOTE-MARKER\n"
        await tool.execute("create", "test-skill", content)
        r = await tool.execute("patch", "test-skill",
                               content="UPDATED", patch_target="NOTE-MARKER")
        assert "appears 2 times" in r

    @pytest.mark.asyncio
    async def test_patch_missing_args(self, tool):
        r = await tool.execute("patch", "test-skill",
                               content="", patch_target="")
        assert "Error" in r


class TestSkillManageDelete:
    @pytest.mark.asyncio
    async def test_delete_simple(self, tool):
        content = _create_skill_content()
        await tool.execute("create", "test-skill", content)
        r = await tool.execute("delete", "test-skill")
        assert "deleted successfully" in r
        skill_dir = tool.workspace / "skills" / "test-skill"
        assert not skill_dir.exists()

    @pytest.mark.asyncio
    async def test_delete_not_found(self, tool):
        r = await tool.execute("delete", "nonexistent")
        assert "not found" in r

    @pytest.mark.asyncio
    async def test_delete_builtin(self, tool):
        r = await tool.execute("delete", "memory")
        assert "Cannot delete built-in" in r

    @pytest.mark.asyncio
    async def test_cannot_patch_builtin(self, tool):
        r = await tool.execute("patch", "memory",
                               content="X", patch_target="Y")
        assert "Cannot patch built-in" in r


class TestSkillManageOther:
    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        r = await tool.execute("rename", "test-skill", "content")
        assert "Unknown action" in r
