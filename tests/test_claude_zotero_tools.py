import sqlite3
from pathlib import Path

import pytest

from nanobot.agent.tools.claude_tools import MultiEditFileTool, SearchFilesTool
from nanobot.agent.tools.zotero_tools import (
    ListZoteroCollectionsTool,
    SearchZoteroLibraryTool,
)


@pytest.mark.asyncio
async def test_search_files_finds_filename_and_content(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("alpha beta gamma\n", encoding="utf-8")
    (tmp_path / "paper_notes.md").write_text("contains transformer keyword\n", encoding="utf-8")

    tool = SearchFilesTool(workspace=tmp_path, allowed_dir=tmp_path)

    result1 = await tool.execute(path=".", query="paper", mode="filename")
    assert "paper_notes.md" in result1

    result2 = await tool.execute(path=".", query="transformer", mode="content")
    assert "paper_notes.md" in result2
    assert "transformer" in result2


@pytest.mark.asyncio
async def test_multi_edit_file_applies_multiple_edits(tmp_path: Path):
    file_path = tmp_path / "sample.py"
    file_path.write_text(
        "def greet():\n"
        "    message = 'hello'\n"
        "    return message\n",
        encoding="utf-8",
    )

    tool = MultiEditFileTool(workspace=tmp_path, allowed_dir=tmp_path)
    result = await tool.execute(
        path="sample.py",
        edits=[
            {"old_text": "message = 'hello'", "new_text": "message = 'hi'"},
            {"old_text": "return message", "new_text": "return message.upper()"},
        ],
    )

    assert "Successfully edited" in result
    updated = file_path.read_text(encoding="utf-8")
    assert "message = 'hi'" in updated
    assert "return message.upper()" in updated


def _create_minimal_zotero_db(root: Path) -> None:
    db = root / "zotero.sqlite"
    storage_dir = root / "storage" / "ATTACHKEY1"
    storage_dir.mkdir(parents=True, exist_ok=True)
    (storage_dir / "paper.pdf").write_text("fake pdf placeholder", encoding="utf-8")

    conn = sqlite3.connect(str(db))
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE itemTypes (
            itemTypeID INTEGER PRIMARY KEY,
            typeName TEXT
        );

        CREATE TABLE items (
            itemID INTEGER PRIMARY KEY,
            key TEXT,
            itemTypeID INTEGER
        );

        CREATE TABLE fieldsCombined (
            fieldID INTEGER PRIMARY KEY,
            fieldName TEXT
        );

        CREATE TABLE itemDataValues (
            valueID INTEGER PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE itemData (
            itemID INTEGER,
            fieldID INTEGER,
            valueID INTEGER
        );

        CREATE TABLE creatorData (
            creatorDataID INTEGER PRIMARY KEY,
            firstName TEXT,
            lastName TEXT
        );

        CREATE TABLE creators (
            creatorID INTEGER PRIMARY KEY,
            creatorDataID INTEGER
        );

        CREATE TABLE itemCreators (
            itemID INTEGER,
            creatorID INTEGER
        );

        CREATE TABLE collections (
            collectionID INTEGER PRIMARY KEY,
            key TEXT,
            collectionName TEXT
        );

        CREATE TABLE collectionItems (
            collectionID INTEGER,
            itemID INTEGER
        );

        CREATE TABLE itemAttachments (
            itemID INTEGER PRIMARY KEY,
            parentItemID INTEGER,
            path TEXT,
            contentType TEXT
        );

        CREATE TABLE deletedItems (
            itemID INTEGER
        );
        """
    )

    cur.executemany(
        "INSERT INTO itemTypes (itemTypeID, typeName) VALUES (?, ?)",
        [
            (1, "journalArticle"),
            (2, "attachment"),
        ],
    )

    cur.executemany(
        "INSERT INTO items (itemID, key, itemTypeID) VALUES (?, ?, ?)",
        [
            (1, "ITEMKEY1", 1),
            (2, "ATTACHKEY1", 2),
        ],
    )

    cur.executemany(
        "INSERT INTO fieldsCombined (fieldID, fieldName) VALUES (?, ?)",
        [
            (1, "title"),
            (2, "date"),
            (3, "publicationTitle"),
        ],
    )

    cur.executemany(
        "INSERT INTO itemDataValues (valueID, value) VALUES (?, ?)",
        [
            (1, "Attention Is All You Need"),
            (2, "2017"),
            (3, "NeurIPS"),
        ],
    )

    cur.executemany(
        "INSERT INTO itemData (itemID, fieldID, valueID) VALUES (?, ?, ?)",
        [
            (1, 1, 1),
            (1, 2, 2),
            (1, 3, 3),
        ],
    )

    cur.execute(
        "INSERT INTO creatorData (creatorDataID, firstName, lastName) VALUES (?, ?, ?)",
        (1, "Ashish", "Vaswani"),
    )
    cur.execute(
        "INSERT INTO creators (creatorID, creatorDataID) VALUES (?, ?)",
        (1, 1),
    )
    cur.execute(
        "INSERT INTO itemCreators (itemID, creatorID) VALUES (?, ?)",
        (1, 1),
    )

    cur.execute(
        "INSERT INTO collections (collectionID, key, collectionName) VALUES (?, ?, ?)",
        (1, "COLLKEY1", "Transformers"),
    )
    cur.execute(
        "INSERT INTO collectionItems (collectionID, itemID) VALUES (?, ?)",
        (1, 1),
    )

    cur.execute(
        "INSERT INTO itemAttachments (itemID, parentItemID, path, contentType) VALUES (?, ?, ?, ?)",
        (2, 1, "storage:paper.pdf", "application/pdf"),
    )

    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_list_zotero_collections(tmp_path: Path):
    _create_minimal_zotero_db(tmp_path)

    tool = ListZoteroCollectionsTool()
    result = await tool.execute(library_path=str(tmp_path))

    assert "Transformers" in result
    assert "COLLKEY1" in result


@pytest.mark.asyncio
async def test_search_zotero_library_returns_item_and_attachment(tmp_path: Path):
    _create_minimal_zotero_db(tmp_path)

    tool = SearchZoteroLibraryTool()
    result = await tool.execute(
        library_path=str(tmp_path),
        query="attention",
        include_attachments=True,
    )

    assert "Attention Is All You Need" in result
    assert "Ashish Vaswani" in result
    assert "NeurIPS" in result
    assert "paper.pdf" in result
