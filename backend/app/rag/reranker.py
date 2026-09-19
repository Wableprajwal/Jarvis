"""Second-stage reranking of the fused dense+BM25 candidates."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from itertools import pairwise

from app.rag.textutil import tokenize


class Reranker(ABC):
    @abstractmethod
    async def score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance score per text (higher = more relevant); order-aligned with `texts`."""


class CrossEncoderReranker(Reranker):
    """sentence-transformers CrossEncoder (default ms-marco-MiniLM-L-6-v2); lazy load, worker thread."""

    def __init__(self, model_name: str | None = None):
        if model_name is None:
            from app.config import get_settings

            model_name = get_settings().reranker_model
        self.model_name = model_name
        self._model = None

    def _predict(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy: heavy import

            self._model = CrossEncoder(self.model_name)
        return [float(s) for s in self._model.predict([(query, t) for t in texts], show_progress_bar=False)]

    async def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return await asyncio.to_thread(self._predict, query, texts)


class SimpleReranker(Reranker):
    """Deterministic lexical-overlap reranker (offline / tests): fraction of distinct query terms
    present in the passage, plus a smaller bonus for adjacent query-term pairs (bigrams)."""

    async def score(self, query: str, texts: list[str]) -> list[float]:
        q = tokenize(query)
        q_terms = set(q)
        q_bigrams = set(pairwise(q))
        scores: list[float] = []
        for text in texts:
            toks = tokenize(text)
            terms = set(toks)
            bigrams = set(pairwise(toks))
            term_cov = len(q_terms & terms) / len(q_terms) if q_terms else 0.0
            bi_cov = len(q_bigrams & bigrams) / len(q_bigrams) if q_bigrams else 0.0
            scores.append(term_cov + 0.5 * bi_cov)
        return scores
