"""Core data layer for vocabulary storage using SQLite."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class WordEntry:
    """A vocabulary word entry."""

    word: str
    meaning: str
    phonetic: str = ""
    examples: list[str] | None = None
    etymology: str = ""
    synonyms: list[str] | None = None
    antonyms: list[str] | None = None
    created_at: str | None = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []
        if self.synonyms is None:
            self.synonyms = []
        if self.antonyms is None:
            self.antonyms = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "word": self.word,
            "meaning": self.meaning,
            "phonetic": self.phonetic,
            "examples": self.examples,
            "etymology": self.etymology,
            "synonyms": self.synonyms,
            "antonyms": self.antonyms,
        }


@dataclass
class LearningStats:
    """User learning statistics."""

    total_words: int
    today_count: int


class VocabularyStore:
    """SQLite-based vocabulary storage.

    Handles CRUD operations for vocabulary words with user isolation.
    Each user has their own vocabulary space in the same database.
    """

    def __init__(self, db_path: str | Path = "./data/vocabulary.db"):
        """Initialize the vocabulary store.

        Args:
            db_path: Path to SQLite database file. Will be created if not exists.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vocabulary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                word TEXT NOT NULL,
                meaning TEXT NOT NULL,
                phonetic TEXT DEFAULT '',
                examples TEXT DEFAULT '[]',
                etymology TEXT DEFAULT '',
                synonyms TEXT DEFAULT '[]',
                antonyms TEXT DEFAULT '[]',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(user_id, word)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_word
            ON vocabulary(user_id, word)
        """)
        conn.commit()

    def add_word(self, entry: WordEntry, user_id: str) -> bool:
        """Add or update a word entry.

        Args:
            entry: The word entry to store.
            user_id: User identifier for isolation.

        Returns:
            True if added, False if updated existing.
        """
        import json

        conn = self._get_conn()
        is_new = True

        try:
            conn.execute(
                """
                INSERT INTO vocabulary
                (user_id, word, meaning, phonetic, examples, etymology, synonyms, antonyms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    entry.word.lower(),
                    entry.meaning,
                    entry.phonetic,
                    json.dumps(entry.examples),
                    entry.etymology,
                    json.dumps(entry.synonyms),
                    json.dumps(entry.antonyms),
                ),
            )
        except sqlite3.IntegrityError:
            # Word exists for this user, update it
            is_new = False
            conn.execute(
                """
                UPDATE vocabulary
                SET meaning = ?, phonetic = ?, examples = ?, etymology = ?,
                    synonyms = ?, antonyms = ?, updated_at = datetime('now')
                WHERE user_id = ? AND word = ?
                """,
                (
                    entry.meaning,
                    entry.phonetic,
                    json.dumps(entry.examples),
                    entry.etymology,
                    json.dumps(entry.synonyms),
                    json.dumps(entry.antonyms),
                    user_id,
                    entry.word.lower(),
                ),
            )
        finally:
            conn.commit()

        return is_new

    def get_word(self, word: str, user_id: str) -> Optional[WordEntry]:
        """Retrieve a word entry.

        Args:
            word: The word to look up.
            user_id: User identifier.

        Returns:
            WordEntry if found, None otherwise.
        """
        import json

        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM vocabulary WHERE user_id = ? AND word = ?",
            (user_id, word.lower()),
        ).fetchone()

        if row is None:
            return None

        return WordEntry(
            word=row["word"],
            meaning=row["meaning"],
            phonetic=row["phonetic"],
            examples=json.loads(row["examples"]),
            etymology=row["etymology"],
            synonyms=json.loads(row["synonyms"]),
            antonyms=json.loads(row["antonyms"]),
            created_at=row["created_at"],
        )

    def list_words(self, user_id: str, limit: int = 10) -> list[WordEntry]:
        """List recently added words.

        Args:
            user_id: User identifier.
            limit: Maximum number of words to return.

        Returns:
            List of WordEntry, most recent first.
        """
        import json

        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM vocabulary
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

        return [
            WordEntry(
                word=row["word"],
                meaning=row["meaning"],
                phonetic=row["phonetic"],
                examples=json.loads(row["examples"]),
                etymology=row["etymology"],
                synonyms=json.loads(row["synonyms"]),
                antonyms=json.loads(row["antonyms"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_stats(self, user_id: str) -> LearningStats:
        """Get learning statistics for a user.

        Args:
            user_id: User identifier.

        Returns:
            LearningStats with total word count and today's count.
        """
        conn = self._get_conn()

        total = conn.execute(
            "SELECT COUNT(*) FROM vocabulary WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]

        today = conn.execute(
            """
            SELECT COUNT(*) FROM vocabulary
            WHERE user_id = ? AND date(created_at) = date('now')
            """,
            (user_id,),
        ).fetchone()[0]

        return LearningStats(total_words=total, today_count=today)

    def search_words(self, prefix: str, user_id: str, limit: int = 20) -> list[WordEntry]:
        """Search for words by prefix.

        Args:
            prefix: Word prefix to search for.
            user_id: User identifier.
            limit: Maximum results.

        Returns:
            List of matching WordEntry.
        """
        import json

        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM vocabulary
            WHERE user_id = ? AND word LIKE ?
            ORDER BY word ASC
            LIMIT ?
            """,
            (user_id, f"{prefix.lower()}%", limit),
        ).fetchall()

        return [
            WordEntry(
                word=row["word"],
                meaning=row["meaning"],
                phonetic=row["phonetic"],
                examples=json.loads(row["examples"]),
                etymology=row["etymology"],
                synonyms=json.loads(row["synonyms"]),
                antonyms=json.loads(row["antonyms"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
