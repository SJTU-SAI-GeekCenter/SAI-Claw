"""Semantic memory: vector-based retrieval over conversation history.

Stores embeddings via litellm (already a project dependency — no new hard deps).
Falls back silently if the embedding model is unconfigured or unavailable.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger


def resolve_embedding_model(main_model: str) -> str | None:
    """Derive a compatible embedding model from the main LLM model string.

    Returns None for providers that have no embedding API (Anthropic, Groq, etc.)
    so callers can silently disable semantic memory rather than crashing.

    Mapping rationale
    -----------------
    openai / gpt / aihubmix / siliconflow
        → text-embedding-3-small  (cheapest OpenAI embedding, ~$0.02/M tokens)
    ollama
        → ollama/nomic-embed-text  (local, free, 768-dim, strong quality)
    gemini
        → gemini/text-embedding-004  (free-tier quota, 768-dim)
    deepseek
        → deepseek/deepseek-embedding  (same key, 1536-dim)
    dashscope / qwen
        → dashscope/text-embedding-v3  (same key)
    zhipu / glm / zai
        → zai/embedding-3  (same key)
    anything else
        → None  (disabled, no crash)
    """
    m = main_model.lower()
    if any(k in m for k in ("openai", "gpt", "aihubmix", "siliconflow")):
        return "text-embedding-3-small"
    if "ollama" in m:
        return "ollama/nomic-embed-text"
    if "gemini" in m:
        return "gemini/text-embedding-004"
    if "deepseek" in m:
        return "deepseek/deepseek-embedding"
    if any(k in m for k in ("qwen", "dashscope")):
        return "dashscope/text-embedding-v3"
    if any(k in m for k in ("zhipu", "glm", "zai")):
        return "zai/embedding-3"
    # Anthropic, Groq, Moonshot, MiniMax, etc. have no embedding endpoint
    return None


class SemanticMemoryStore:
    """Numpy-backed cosine-similarity index for history entries.

    Layout on disk::

        memory/semantic/
            embeddings.npy   — float32 array, shape (N, D)
            metadata.json    — list of {text, ts} dicts

    All public methods are async-safe; the internal numpy array is
    protected by an asyncio.Lock so concurrent embeds don't race.
    """

    _SIMILARITY_THRESHOLD = 0.45
    _SNIPPET_MAX_CHARS = 600

    def __init__(self, memory_dir: Path, embedding_model: str):
        self.semantic_dir = memory_dir / "semantic"
        self.index_file = self.semantic_dir / "embeddings.npy"
        self.meta_file = self.semantic_dir / "metadata.json"
        self.embedding_model = embedding_model
        self._lock = asyncio.Lock()
        self._embeddings: Any = None   # np.ndarray | None
        self._metadata: list[dict] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add_entry(self, text: str) -> None:
        """Embed *text* and append it to the index.  Never raises."""
        try:
            vec = await self._embed(text)
            async with self._lock:
                self._ensure_loaded()
                self._append_vector(vec, text)
                self._save()
        except Exception as exc:
            logger.warning("Semantic indexing skipped ({}): {}", type(exc).__name__, exc)

    async def search(self, query: str, k: int = 5) -> list[str]:
        """Return up to *k* history snippets most relevant to *query*.

        Returns an empty list on any error so callers never need to guard.
        """
        try:
            async with self._lock:
                self._ensure_loaded()
                n = len(self._metadata)
                if n == 0:
                    return []

            q_vec = await self._embed(query)

            async with self._lock:
                import numpy as np
                emb = self._embeddings  # (N, D)
                if emb is None or emb.shape[0] == 0:
                    return []

                # Cosine similarity: (N,)
                q_norm = np.linalg.norm(q_vec)
                if q_norm < 1e-9:
                    return []
                row_norms = np.linalg.norm(emb, axis=1)
                row_norms = np.where(row_norms < 1e-9, 1e-9, row_norms)
                scores = (emb @ q_vec) / (row_norms * q_norm)

                top_idx = int(min(k, len(scores)))
                ranked = scores.argsort()[::-1][:top_idx]
                return [
                    self._metadata[i]["text"]
                    for i in ranked
                    if scores[i] >= self._SIMILARITY_THRESHOLD
                ]
        except Exception as exc:
            logger.warning("Semantic search skipped ({}): {}", type(exc).__name__, exc)
            return []

    def entry_count(self) -> int:
        """Return the number of indexed entries (0 if not yet loaded)."""
        return len(self._metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load index from disk on first access (call inside lock)."""
        if self._loaded:
            return
        self._loaded = True
        if self.index_file.exists() and self.meta_file.exists():
            try:
                import numpy as np
                self._embeddings = np.load(str(self.index_file))
                self._metadata = json.loads(self.meta_file.read_text(encoding="utf-8"))
                logger.debug(
                    "Semantic index loaded: {} entries, dim={}",
                    len(self._metadata),
                    self._embeddings.shape[1] if self._embeddings.ndim == 2 else "?",
                )
            except Exception as exc:
                logger.warning("Semantic index corrupted, starting fresh: {}", exc)
                self._reset()
        else:
            self._reset()

    def _reset(self) -> None:
        import numpy as np
        self._embeddings = np.zeros((0, 0), dtype=np.float32)
        self._metadata = []

    def _append_vector(self, vec: Any, text: str) -> None:
        import numpy as np
        row = vec.reshape(1, -1).astype(np.float32)
        if self._embeddings is None or self._embeddings.shape[0] == 0:
            self._embeddings = row
        else:
            self._embeddings = np.vstack([self._embeddings, row])
        self._metadata.append({"text": text[: self._SNIPPET_MAX_CHARS]})

    def _save(self) -> None:
        import numpy as np
        self.semantic_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(self.index_file), self._embeddings)
        self.meta_file.write_text(
            json.dumps(self._metadata, ensure_ascii=False), encoding="utf-8"
        )

    async def _embed(self, text: str) -> Any:
        """Call litellm.aembedding and return a float32 numpy vector."""
        import numpy as np
        import litellm

        response = await litellm.aembedding(
            model=self.embedding_model,
            input=[text],
        )
        return np.array(response.data[0]["embedding"], dtype=np.float32)
