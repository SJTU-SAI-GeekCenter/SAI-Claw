"""Command handler for vocabulary learning feature."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.skills.vocab.core import VocabularyStore
    from nanobot.skills.vocab.generator import VocabGenerator


def parse_command(content: str) -> tuple[str, list[str]]:
    """Parse command and arguments from message content.

    Examples:
        "/vocab apple banana" -> ("/vocab", ["apple", "banana"])
        "/word resilience" -> ("/word", ["resilience"])
        "/paragraph" -> ("/paragraph", [])

    Args:
        content: Message content to parse.

    Returns:
        Tuple of (command, args).
    """
    parts = content.strip().split()
    if not parts or not parts[0].startswith("/"):
        return ("", [])

    command = parts[0].lower()
    args = parts[1:]
    return (command, args)


class VocabHandler:
    """Handles vocabulary-related commands.

    Coordinates between VocabularyStore (data) and VocabGenerator (LLM)
    to process user commands and return formatted responses.
    """

    def __init__(self, store: "VocabularyStore", generator: "VocabGenerator"):
        """Initialize the handler.

        Args:
            store: Vocabulary data store instance.
            generator: LLM content generator instance.
        """
        self.store = store
        self.generator = generator

    async def handle_vocab(
        self,
        words: list[str],
        user_id: str,
    ) -> dict:
        """Handle /vocab command to store words.

        Args:
            words: List of words to store.
            user_id: User identifier.

        Returns:
            Dict with success, stored_count, duplicate_count.
        """
        if not words:
            return {
                "success": False,
                "error": "请提供要存储的单词，例如：/vocab apple banana cherry"
            }

        from nanobot.skills.vocab.core import WordEntry

        # Clean words
        cleaned_words = [
            w.strip(".,!?;:\"'").lower()
            for w in words
            if w.strip()
        ]

        if not cleaned_words:
            return {
                "success": False,
                "error": "没有有效的单词"
            }

        stored_count = 0
        duplicate_count = 0

        for word in cleaned_words:
            # Generate basic entry
            entry = WordEntry(
                word=word,
                meaning="",  # Will be enriched if user provides meaning
            )

            is_new = self.store.add_word(entry, user_id)
            if is_new:
                stored_count += 1
            else:
                duplicate_count += 1

        return {
            "success": True,
            "stored_count": stored_count,
            "duplicate_count": duplicate_count,
        }

    async def handle_word(
        self,
        word: str,
        user_id: str,
        meaning: str | None = None,
    ) -> dict:
        """Handle /word command to query word details.

        Args:
            word: Word to look up.
            user_id: User identifier.
            meaning: Optional meaning to store if word not found.

        Returns:
            Dict with success, word entry, or error.
        """
        if not word:
            return {
                "success": False,
                "error": "请提供要查询的单词，例如：/word resilience"
            }

        word_clean = word.strip(".,!?;:\"'").lower()
        if not word_clean:
            return {
                "success": False,
                "error": "无效的单词"
            }

        from nanobot.skills.vocab.core import WordEntry

        # Check if word exists in database
        entry = self.store.get_word(word_clean, user_id)

        if entry is None and meaning:
            # Word not found, generate and store it
            enrichment = await self.generator.enrich_word_entry(word_clean, meaning)

            entry = WordEntry(
                word=word_clean,
                meaning=meaning,
                phonetic=enrichment.get("phonetic", ""),
                examples=enrichment.get("examples", []),
                etymology=enrichment.get("etymology", ""),
                synonyms=enrichment.get("synonyms", []),
                antonyms=enrichment.get("antonyms", []),
            )

            self.store.add_word(entry, user_id)

        if entry is None:
            return {
                "success": False,
                "error": f"单词 '{word_clean}' 未找到。请提供释义来创建词条，例如：/word resilience means 'the ability to recover quickly'"
            }

        return {
            "success": True,
            "word": entry.to_dict(),
        }

    async def handle_paragraph(
        self,
        words: list[str] | None = None,
        level: str = "intermediate",
    ) -> dict:
        """Handle /paragraph command to generate contextual text.

        Args:
            words: List of words to include. If None, uses recent words.
            level: Difficulty level.

        Returns:
            Dict with success and generated content.
        """
        # If no words specified, could fetch from recent words
        if not words:
            return {
                "success": False,
                "error": "请提供要使用的单词，例如：/paragraph resilience community"
            }

        content = await self.generator.generate_paragraph(words, level)

        return {
            "success": True,
            "content": content,
        }

    async def handle_stats(self, user_id: str) -> dict:
        """Handle /stats command to show learning statistics.

        Args:
            user_id: User identifier.

        Returns:
            Dict with success and statistics.
        """
        stats = self.store.get_stats(user_id)

        return {
            "success": True,
            "total_words": stats.total_words,
            "today_count": stats.today_count,
        }

    async def handle_review(self, user_id: str) -> dict:
        """Handle /review command (placeholder for future SRS).

        Args:
            user_id: User identifier.

        Returns:
            Dict with success and review queue.
        """
        # MVP: just return recent words
        words = self.store.list_words(user_id, limit=5)

        return {
            "success": True,
            "review_words": [w.to_dict() for w in words],
            "message": "复习功能即将升级，现在显示最近学习的单词"
        }


def format_word_result(result: dict) -> str:
    """Format word query result as Markdown.

    Args:
        result: Word entry dict from handler.

    Returns:
        Formatted Markdown string.
    """
    word = result.get("word", "Unknown")
    phonetic = result.get("phonetic", "")
    lines = [f"📖 **{word}** {phonetic}\n"]

    meaning = result.get("meaning", "")
    if meaning:
        lines.append(f"**释义：** {meaning}\n")

    examples = result.get("examples", [])
    if examples:
        lines.append("**例句：**")
        for example in examples:
            lines.append(f"- {example}")
        lines.append("")

    etymology = result.get("etymology", "")
    if etymology:
        lines.append(f"**词根词缀：**\n{etymology}\n")

    synonyms = result.get("synonyms", [])
    if synonyms:
        lines.append(f"**同义词：** {', '.join(synonyms)}")

    antonyms = result.get("antonyms", [])
    if antonyms:
        lines.append(f"**反义词：** {', '.join(antonyms)}")

    return "\n".join(lines)


def format_paragraph_result(result: dict) -> str:
    """Format paragraph result as Markdown.

    Args:
        result: Paragraph content dict from handler.

    Returns:
        Formatted Markdown string.
    """
    lines = ["📚 **语境短文**\n"]

    paragraph = result.get("paragraph", "")
    if paragraph:
        lines.append(paragraph)
        lines.append("")

    translation = result.get("translation", "")
    if translation:
        lines.append(f"**中文翻译：**\n{translation}")
        lines.append("")

    words_used = result.get("words_used", [])
    if words_used:
        lines.append(f"**使用的单词：** {', '.join(words_used)}")

    return "\n".join(lines)
