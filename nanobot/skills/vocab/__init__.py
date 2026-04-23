"""Vocabulary learning skill for SAI-Claw."""

# Lazy imports to handle modules being implemented in stages
# This allows the vocab skill to be incrementally developed without import errors

def __getattr__(name: str):
    """Lazy import handler for vocab skill components."""
    if name == "VocabularyStore":
        from nanobot.skills.vocab.core import VocabularyStore
        return VocabularyStore
    elif name == "VocabGenerator":
        from nanobot.skills.vocab.generator import VocabGenerator
        return VocabGenerator
    elif name == "VocabHandler":
        from nanobot.skills.vocab.handler import VocabHandler
        return VocabHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["VocabularyStore", "VocabGenerator", "VocabHandler"]
