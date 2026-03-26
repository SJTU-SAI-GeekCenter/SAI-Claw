"""Zotero local library tools."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


def _resolve_library_root(library_path: str) -> Path:
    p = Path(library_path).expanduser().resolve()
    if p.is_file():
        if p.name != "zotero.sqlite":
            raise ValueError("library_path must be a Zotero data directory or zotero.sqlite file")
        return p.parent
    return p


def _connect_db(library_root: Path) -> sqlite3.Connection:
    db_path = library_root / "zotero.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"zotero.sqlite not found under: {library_root}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r["name"] == column for r in rows)


def _collection_name_column(conn: sqlite3.Connection) -> str:
    if _has_column(conn, "collections", "collectionName"):
        return "collectionName"
    if _has_column(conn, "collections", "name"):
        return "name"
    return "collectionName"


def _resolve_attachment_path(library_root: Path, attachment_key: str, raw_path: str | None) -> str | None:
    if not raw_path:
        return None

    if raw_path.startswith("storage:"):
        filename = raw_path.split(":", 1)[1]
        return str((library_root / "storage" / attachment_key / filename).resolve())

    p = Path(raw_path).expanduser()
    if p.is_absolute():
        return str(p)

    return str((library_root / p).resolve())


class SearchZoteroLibraryTool(Tool):
    """Search items in a local Zotero library."""

    @property
    def name(self) -> str:
        return "search_zotero_library"

    @property
    def description(self) -> str:
        return (
            "Search a local Zotero library by title, author, venue, or year. "
            "Returns matching items and attachment file paths when available. "
            "Pass the Zotero data directory (the folder containing zotero.sqlite)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "library_path": {
                    "type": "string",
                    "description": "Path to the Zotero data directory or zotero.sqlite file",
                },
                "query": {
                    "type": "string",
                    "description": "Search query such as a title keyword, author name, venue, or year",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of items to return (default 10)",
                    "minimum": 1,
                },
                "include_attachments": {
                    "type": "boolean",
                    "description": "Whether to include resolved attachment file paths",
                },
            },
            "required": ["library_path", "query"],
        }

    async def execute(
        self,
        library_path: str,
        query: str,
        limit: int = 10,
        include_attachments: bool = True,
        **kwargs: Any,
    ) -> str:
        try:
            library_root = _resolve_library_root(library_path)
            conn = _connect_db(library_root)

            deleted_clause = ""
            if _has_table(conn, "deletedItems"):
                deleted_clause = "AND i.itemID NOT IN (SELECT itemID FROM deletedItems)"

            sql = f"""
            WITH creator_names AS (
                SELECT
                    ic.itemID AS itemID,
                    GROUP_CONCAT(
                        TRIM(COALESCE(cd.firstName, '') || ' ' || COALESCE(cd.lastName, '')),
                        '; '
                    ) AS creators
                FROM itemCreators ic
                JOIN creators c ON c.creatorID = ic.creatorID
                JOIN creatorData cd ON cd.creatorDataID = c.creatorDataID
                GROUP BY ic.itemID
            ),
            base AS (
                SELECT
                    i.itemID AS itemID,
                    i.key AS itemKey,
                    it.typeName AS itemType,
                    COALESCE(MAX(CASE WHEN f.fieldName = 'title' THEN v.value END), '') AS title,
                    COALESCE(MAX(CASE WHEN f.fieldName IN ('date', 'year') THEN v.value END), '') AS date,
                    COALESCE(MAX(CASE
                        WHEN f.fieldName IN (
                            'publicationTitle',
                            'bookTitle',
                            'proceedingsTitle',
                            'seriesTitle',
                            'university',
                            'institution'
                        ) THEN v.value
                    END), '') AS venue
                FROM items i
                JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
                LEFT JOIN itemData d ON d.itemID = i.itemID
                LEFT JOIN fieldsCombined f ON f.fieldID = d.fieldID
                LEFT JOIN itemDataValues v ON v.valueID = d.valueID
                WHERE it.typeName NOT IN ('attachment', 'note', 'annotation')
                {deleted_clause}
                GROUP BY i.itemID, i.key, it.typeName
            )
            SELECT
                base.itemID,
                base.itemKey,
                base.itemType,
                base.title,
                base.date,
                base.venue,
                COALESCE(creator_names.creators, '') AS creators
            FROM base
            LEFT JOIN creator_names ON creator_names.itemID = base.itemID
            WHERE
                lower(base.title) LIKE ?
                OR lower(base.venue) LIKE ?
                OR lower(base.date) LIKE ?
                OR lower(COALESCE(creator_names.creators, '')) LIKE ?
            ORDER BY
                CASE WHEN lower(base.title) LIKE ? THEN 0 ELSE 1 END,
                base.title ASC
            LIMIT ?
            """

            q = f"%{query.lower()}%"
            rows = conn.execute(sql, (q, q, q, q, q, limit)).fetchall()

            if not rows:
                conn.close()
                return (
                    f"No Zotero items matched query {query!r} in {library_root}. "
                    "Try a broader title keyword, author name, or year."
                )

            attachments_by_parent: dict[int, list[str]] = {}
            if include_attachments:
                item_ids = [int(r["itemID"]) for r in rows]
                placeholders = ",".join("?" for _ in item_ids)
                attach_sql = f"""
                SELECT
                    ia.parentItemID AS parentItemID,
                    it.key AS attachmentKey,
                    ia.path AS attachmentPath,
                    ia.contentType AS contentType
                FROM itemAttachments ia
                JOIN items it ON it.itemID = ia.itemID
                WHERE ia.parentItemID IN ({placeholders})
                """
                for arow in conn.execute(attach_sql, item_ids).fetchall():
                    resolved = _resolve_attachment_path(
                        library_root,
                        str(arow["attachmentKey"]),
                        arow["attachmentPath"],
                    )
                    if resolved:
                        label = arow["contentType"] or "attachment"
                        attachments_by_parent.setdefault(int(arow["parentItemID"]), []).append(
                            f"{label}: {resolved}"
                        )

            conn.close()

            lines = [
                f"Zotero library: {library_root}",
                f"Query: {query}",
                f"Matched items: {len(rows)}",
                "",
            ]

            for idx, row in enumerate(rows, 1):
                lines.append(f"{idx}. {row['title'] or '(untitled)'}")
                lines.append(f"   - item_key: {row['itemKey']}")
                lines.append(f"   - item_type: {row['itemType']}")
                if row["creators"]:
                    lines.append(f"   - creators: {row['creators']}")
                if row["date"]:
                    lines.append(f"   - date: {row['date']}")
                if row["venue"]:
                    lines.append(f"   - venue: {row['venue']}")

                attachments = attachments_by_parent.get(int(row["itemID"]), [])
                if attachments:
                    lines.append("   - attachments:")
                    for item in attachments[:5]:
                        lines.append(f"       * {item}")

                lines.append("")

            return "\n".join(lines).rstrip()

        except FileNotFoundError as e:
            return f"Error: {e}"
        except ValueError as e:
            return f"Error: {e}"
        except sqlite3.Error as e:
            return f"Error reading Zotero database: {e}"
        except Exception as e:
            return f"Error searching Zotero library: {e}"


class ListZoteroCollectionsTool(Tool):
    """List collections in a local Zotero library."""

    @property
    def name(self) -> str:
        return "list_zotero_collections"

    @property
    def description(self) -> str:
        return (
            "List collections in a local Zotero library, optionally filtering by a query. "
            "Useful for understanding the structure of a Zotero library before searching specific papers."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "library_path": {
                    "type": "string",
                    "description": "Path to the Zotero data directory or zotero.sqlite file",
                },
                "query": {
                    "type": "string",
                    "description": "Optional collection-name filter",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of collections to return (default 50)",
                    "minimum": 1,
                },
            },
            "required": ["library_path"],
        }

    async def execute(
        self,
        library_path: str,
        query: str | None = None,
        limit: int = 50,
        **kwargs: Any,
    ) -> str:
        try:
            library_root = _resolve_library_root(library_path)
            conn = _connect_db(library_root)

            name_col = _collection_name_column(conn)
            sql = f"""
            SELECT
                c.collectionID AS collectionID,
                c.key AS collectionKey,
                c.{name_col} AS collectionName,
                COUNT(ci.itemID) AS itemCount
            FROM collections c
            LEFT JOIN collectionItems ci ON ci.collectionID = c.collectionID
            GROUP BY c.collectionID, c.key, c.{name_col}
            ORDER BY c.{name_col} COLLATE NOCASE ASC
            """
            rows = conn.execute(sql).fetchall()
            conn.close()

            if query:
                q = query.lower()
                rows = [r for r in rows if q in str(r["collectionName"] or "").lower()]

            rows = rows[:limit]

            if not rows:
                if query:
                    return f"No Zotero collections matched query {query!r} in {library_root}"
                return f"No Zotero collections found in {library_root}"

            lines = [
                f"Zotero library: {library_root}",
                f"Collections returned: {len(rows)}",
                "",
            ]
            for idx, row in enumerate(rows, 1):
                lines.append(
                    f"{idx}. {row['collectionName']} "
                    f"(key={row['collectionKey']}, items={row['itemCount']})"
                )

            return "\n".join(lines)

        except FileNotFoundError as e:
            return f"Error: {e}"
        except ValueError as e:
            return f"Error: {e}"
        except sqlite3.Error as e:
            return f"Error reading Zotero database: {e}"
        except Exception as e:
            return f"Error listing Zotero collections: {e}"
