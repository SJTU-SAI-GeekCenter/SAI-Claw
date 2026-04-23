"""LLM-based content generation for vocabulary learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


class VocabGenerator:
    """Generates vocabulary learning content using LLM.

    Handles LLM calls for generating examples, paragraphs, etymology,
    and other educational content.
    """

    def __init__(self, provider: "LLMProvider", model: str):
        """Initialize the generator.

        Args:
            provider: LLM provider instance.
            model: Model name to use for generation.
        """
        self.provider = provider
        self.model = model

    async def generate_examples(self, word: str, count: int = 3) -> list[str]:
        """Generate example sentences for a word.

        Args:
            word: The target word.
            count: Number of examples to generate.

        Returns:
            List of example sentences.
        """
        prompt = f"""Generate {count} distinct, natural English sentences using the word "{word}".

Requirements:
- Each sentence should demonstrate a different context/meaning
- Sentences should be appropriate for intermediate learners
- Number each sentence on a new line

Output only the numbered sentences, nothing else."""

        response = await self.provider.chat(
            messages=[
                {"role": "system", "content": "You are an English language teaching assistant."},
                {"role": "user", "content": prompt},
            ],
            tools=None,
            model=self.model,
            max_tokens=500,
            temperature=0.7,
        )

        content = response.content or ""
        examples = []
        for line in content.strip().split("\n"):
            line = line.strip()
            # Remove numbering like "1." or "1) "
            for prefix in (f"{i}." for i in range(1, count + 1)):
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if line and not line.startswith(tuple(str(i) for i in range(10))):
                examples.append(line)

        return examples[:count]

    async def generate_paragraph(
        self,
        words: list[str],
        level: str = "intermediate",
    ) -> dict:
        """Generate a contextual paragraph using given words.

        Args:
            words: List of words to include in the paragraph.
            level: Difficulty level (beginner, intermediate, advanced).

        Returns:
            Dict with 'paragraph', 'translation', and 'words_used' keys.
        """
        word_list = ", ".join(f'"{w}"' for w in words)
        prompt = f"""Write a natural, coherent English paragraph (100-150 words) at {level} level.
Use these words: {word_list}.

After the paragraph, provide a Chinese translation.

Format your response as:
[English paragraph]

[Chinese translation]"""

        response = await self.provider.chat(
            messages=[
                {"role": "system", "content": "You are an English language teaching assistant."},
                {"role": "user", "content": prompt},
            ],
            tools=None,
            model=self.model,
            max_tokens=1000,
            temperature=0.7,
        )

        content = response.content or ""

        # Split by paragraph breaks to separate English and Chinese
        parts = content.split("\n\n")
        paragraph = parts[0] if parts else ""
        translation = parts[1] if len(parts) > 1 else ""

        return {
            "paragraph": paragraph,
            "translation": translation,
            "words_used": words,
        }

    async def generate_etymology(self, word: str) -> str:
        """Generate etymology information for a word.

        Args:
            word: The word to analyze.

        Returns:
            Etymology description.
        """
        prompt = f"""Provide a concise etymology for the English word "{word}".
Include:
- Language of origin
- Root words and their meanings
- How the meaning evolved

Keep it to 1-2 sentences. Focus on interesting historical connections."""

        response = await self.provider.chat(
            messages=[
                {"role": "system", "content": "You are an etymology expert."},
                {"role": "user", "content": prompt},
            ],
            tools=None,
            model=self.model,
            max_tokens=300,
            temperature=0.3,
        )

        return response.content or ""

    async def enrich_word_entry(self, word: str, meaning: str) -> dict:
        """Enrich a word entry with LLM-generated content.

        Args:
            word: The target word.
            meaning: Basic meaning/definition.

        Returns:
            Dict with phonetic, etymology, examples, synonyms, antonyms.
        """
        prompt = f"""For the English word "{word}" (meaning: {meaning}), provide:

1. Phonetic transcription (IPA)
2. Brief etymology (1 sentence)
3. 2 example sentences
4. 2 synonyms
5. 1 antonym

Format as JSON:
{{
  "phonetic": "...",
  "etymology": "...",
  "examples": ["...", "..."],
  "synonyms": ["...", "..."],
  "antonyms": ["..."]
}}"""

        response = await self.provider.chat(
            messages=[
                {"role": "system", "content": "You are a dictionary API. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            tools=None,
            model=self.model,
            max_tokens=500,
            temperature=0.3,
        )

        import json

        content = response.content or "{}"
        try:
            # Try to extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback if LLM didn't return valid JSON
            return {
                "phonetic": "",
                "etymology": "",
                "examples": [],
                "synonyms": [],
                "antonyms": [],
            }
