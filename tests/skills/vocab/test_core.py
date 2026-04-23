"""Tests for VocabularyStore core data layer."""

import pytest
from pathlib import Path
from nanobot.skills.vocab.core import VocabularyStore, WordEntry


@pytest.fixture
def temp_db(tmp_path: Path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_vocab.db"
    store = VocabularyStore(db_path=str(db_path))
    yield store
    store.close()


class TestVocabularyStore:
    """Tests for VocabularyStore class."""

    def test_add_and_get_word(self, temp_db: VocabularyStore):
        """Test adding a word and retrieving it."""
        entry = WordEntry(
            word="resilience",
            meaning="the ability to recover quickly from difficulties",
            phonetic="/rɪˈzɪliəns/",
            examples=["She showed great resilience in overcoming the setback."],
            etymology="from Latin resiliens 'leaping back'",
            synonyms=["toughness", "character"],
            antonyms=["fragility", "weakness"]
        )
        temp_db.add_word(entry, user_id="user123")

        result = temp_db.get_word("resilience", user_id="user123")
        assert result is not None
        assert result.word == "resilience"
        assert result.meaning == "the ability to recover quickly from difficulties"
        assert "She showed" in result.examples[0]

    def test_add_duplicate_word_updates_existing(self, temp_db: VocabularyStore):
        """Test that adding a duplicate word updates the existing entry."""
        entry1 = WordEntry(word="test", meaning="first meaning", examples=[])
        temp_db.add_word(entry1, user_id="user123")

        entry2 = WordEntry(word="test", meaning="updated meaning", examples=["new example"])
        temp_db.add_word(entry2, user_id="user123")

        result = temp_db.get_word("test", user_id="user123")
        assert result.meaning == "updated meaning"
        assert "new example" in result.examples[0]

    def test_get_nonexistent_word_returns_none(self, temp_db: VocabularyStore):
        """Test retrieving a word that doesn't exist."""
        result = temp_db.get_word("nonexistent", user_id="user123")
        assert result is None

    def test_list_words_returns_recent_entries(self, temp_db: VocabularyStore):
        """Test listing recently added words."""
        for i in range(5):
            entry = WordEntry(word=f"word{i}", meaning=f"meaning {i}", examples=[])
            temp_db.add_word(entry, user_id="user123")

        words = temp_db.list_words(user_id="user123", limit=3)
        assert len(words) == 3
        # Should return most recent first
        assert words[0].word == "word4"

    def test_get_stats_returns_correct_counts(self, temp_db: VocabularyStore):
        """Test getting learning statistics."""
        # Add 5 words for user123
        for i in range(5):
            entry = WordEntry(word=f"word{i}", meaning=f"meaning {i}", examples=[])
            temp_db.add_word(entry, user_id="user123")

        # Add 2 words for user456
        for i in range(2):
            entry = WordEntry(word=f"other{i}", meaning=f"meaning {i}", examples=[])
            temp_db.add_word(entry, user_id="user456")

        stats = temp_db.get_stats(user_id="user123")
        assert stats.total_words == 5

    def test_search_words_finds_matches(self, temp_db: VocabularyStore):
        """Test searching for words by prefix."""
        entry1 = WordEntry(word="apple", meaning="fruit", examples=[])
        entry2 = WordEntry(word="application", meaning="software", examples=[])
        entry3 = WordEntry(word="banana", meaning="fruit", examples=[])

        temp_db.add_word(entry1, user_id="user123")
        temp_db.add_word(entry2, user_id="user123")
        temp_db.add_word(entry3, user_id="user123")

        results = temp_db.search_words("app", user_id="user123")
        assert len(results) == 2
        word_list = [w.word for w in results]
        assert "apple" in word_list
        assert "application" in word_list
