"""Hybrid retrieval: dense (top-20) + BM25 (top-20) -> reciprocal-rank fusion + dedupe ->
rerank -> top 5 `Citation`s.

Why hybrid: dense vectors capture meaning but blur exact identifiers, error codes and config
keys; BM25 nails exact terms but misses paraphrases. RRF fuses the two rank lists without
having to calibrate incomparable score scales; the reranker then reads query and passage
together to pick the final few.
"""
from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.rag.embedder import Embedder
from app.rag.reranker import Reranker
from app.rag.store import StoredChunk, VectorStore
from app.rag.textutil import tokenize


@dataclass
class Citation:
    source: str
    chunk_id: int  # index of the chunk within `source`
    score: float  # reranker score (higher is better)
    text: str
    doc_type: str | None = None
    date: str | None = None

    @property
    def ref(self) -> str:
        """Inline citation label, e.g. `adr-0007-retry-backoff.md#1`."""
        return f"{self.source}#{self.chunk_id}"

    def to_dict(self) -> dict:
        return {
            "ref": self.ref, "source": self.source, "chunk_id": self.chunk_id, "score": round(self.score, 4),
            "doc_type": self.doc_type, "date": self.date, "text": self.text,
        }


@dataclass
class Candidate:
    chunk: StoredChunk
    rrf_score: float
    dense_rank: int | None = None  # 1-based rank in the dense list, None if absent
    bm25_rank: int | None = None


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        reranker: Reranker,
        *,
        dense_k: int = 20,
        bm25_k: int = 20,
        top_n: int = 5,
        rrf_k: int = 60,
        min_score: float | None = None,
    ):
        self.store, self.embedder, self.reranker = store, embedder, reranker
        self.dense_k, self.bm25_k, self.top_n, self.rrf_k = dense_k, bm25_k, top_n, rrf_k
        self.min_score = min_score  # optional floor on the reranker score (scale depends on the reranker)
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[StoredChunk] = []
        self._bm25_tokens: list[set[str]] = []
        self._bm25_fp: str | None = None

    async def _ensure_bm25(self) -> None:
        fp = await self.store.fingerprint()
        if fp == self._bm25_fp and self._bm25_fp is not None:
            return
        chunks = await self.store.all_chunks()
        tokenized = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if any(tokenized) else None
        self._bm25_chunks = chunks
        self._bm25_tokens = [set(t) for t in tokenized]
        self._bm25_fp = fp

    def _bm25_search(self, query: str) -> list[StoredChunk]:
        q = tokenize(query)
        if self._bm25 is None or not q:
            return []
        scores = self._bm25.get_scores(q)
        q_set = set(q)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        # a chunk with no query term at all is not a keyword hit, whatever the score arithmetic says
        return [self._bm25_chunks[i] for i in order if q_set & self._bm25_tokens[i]][: self.bm25_k]

    async def candidates(self, query: str) -> list[Candidate]:
        """Fused, deduped candidate list (best RRF first), before reranking."""
        await self._ensure_bm25()
        q_vec = await self.embedder.embed_query(query)
        dense = await self.store.dense_search(q_vec, self.dense_k)
        sparse = self._bm25_search(query)

        by_id: dict[int, Candidate] = {}
        for rank, sc in enumerate(dense, start=1):
            cand = by_id.setdefault(sc.chunk.id, Candidate(sc.chunk, 0.0))
            cand.dense_rank = rank
            cand.rrf_score += 1.0 / (self.rrf_k + rank)
        for rank, chunk in enumerate(sparse, start=1):
            cand = by_id.setdefault(chunk.id, Candidate(chunk, 0.0))
            cand.bm25_rank = rank
            cand.rrf_score += 1.0 / (self.rrf_k + rank)

        fused = sorted(by_id.values(), key=lambda c: (-c.rrf_score, c.chunk.id))
        seen_text: set[str] = set()
        out: list[Candidate] = []
        for c in fused:  # same passage ingested under two sources: keep the better-ranked one
            key = " ".join(c.chunk.text.split())
            if key not in seen_text:
                seen_text.add(key)
                out.append(c)
        return out

    async def search(self, query: str) -> list[Citation]:
        cands = await self.candidates(query)
        if not cands:
            return []
        scores = await self.reranker.score(query, [c.chunk.text for c in cands])
        ranked = sorted(zip(cands, scores, strict=True), key=lambda cs: (-cs[1], -cs[0].rrf_score, cs[0].chunk.id))
        return [
            Citation(
                source=c.chunk.source,
                chunk_id=c.chunk.chunk_index,
                score=float(s),
                text=c.chunk.text,
                doc_type=c.chunk.metadata.get("doc_type"),
                date=c.chunk.metadata.get("date"),
            )
            for c, s in ranked[: self.top_n]
            if self.min_score is None or s >= self.min_score
        ]
