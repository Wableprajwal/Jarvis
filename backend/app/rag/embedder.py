"""Text embedding for RAG.

Design note: embeddings are deliberately INDEPENDENT of the chat `ModelProvider`. Retrieval
runs on its own local model (config.embedding_model, default BAAI/bge-small-en-v1.5), so RAG
works fully offline no matter which chat backend is active, and swapping the chat model
(Ollama today, something else tomorrow, a mock in tests) never invalidates stored vectors.
The flip side: vectors are only comparable within one embedder, so every stored chunk is
tagged with `Embedder.name` and searches are scoped to it.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import re
from abc import ABC, abstractmethod

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_KNOWN_DIMS = {"BAAI/bge-small-en-v1.5": 384, "BAAI/bge-base-en-v1.5": 768, "BAAI/bge-large-en-v1.5": 1024}


class Embedder(ABC):
    name: str  # identifies the vector space; stored with every chunk

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed documents. Returns one L2-normalized vector per input text."""

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query (some models use an asymmetric query prefix)."""
        return (await self.embed([text]))[0]


class SentenceTransformerEmbedder(Embedder):
    """sentence-transformers model, imported and loaded lazily, inference run in a worker thread."""

    def __init__(self, model_name: str | None = None, *, batch_size: int = 32):
        if model_name is None:
            from app.config import get_settings

            model_name = get_settings().embedding_model
        self.model_name = model_name
        self.name = model_name
        self._batch_size = batch_size
        self._model = None
        self._dim: int | None = _KNOWN_DIMS.get(model_name)

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy: heavy import

            self._model = SentenceTransformer(self.model_name)
            getter = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
            self._dim = int(getter())
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        return int(self._dim)  # type: ignore[arg-type]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vecs = model.encode(texts, batch_size=self._batch_size, normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in v] for v in vecs]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, texts)

    async def embed_query(self, text: str) -> list[float]:
        prefix = BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return (await self.embed([prefix + text]))[0]


_WORD_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder(Embedder):
    """Deterministic, offline, dependency-free embedder for tests and no-model environments.

    Feature hashing over word unigrams plus character trigrams (so inflections and typos such
    as "replicating"/"replication" still land close together), signed to reduce collisions,
    then L2-normalized. Uses blake2b, not `hash()`, so vectors are stable across processes.
    """

    def __init__(self, dim: int = 256):
        self._dim = dim
        self.name = f"hashing-{dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _features(self, text: str) -> list[tuple[str, float]]:
        feats: list[tuple[str, float]] = []
        for word in _WORD_RE.findall(text.lower()):
            feats.append((f"w:{word}", 1.0))
            padded = f"^{word}$"
            feats.extend((f"c:{padded[i:i + 3]}", 0.5) for i in range(len(padded) - 2))
        return feats

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for feat, weight in self._features(text):
            digest = hashlib.blake2b(feat.encode(), digest_size=8).digest()
            n = int.from_bytes(digest, "big")
            vec[n % self._dim] += weight if (n >> 63) & 1 else -weight
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]
