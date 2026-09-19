"""Hybrid retriever: BM25-only wins, dense-only wins, fusion/dedupe, top-5 cap, reranking.
Vectors are hand-placed so each case isolates one retrieval leg. No DB, model or network."""

from app.rag.embedder import Embedder
from app.rag.reranker import Reranker, SimpleReranker
from app.rag.retriever import Citation, HybridRetriever
from app.rag.store import InMemoryVectorStore, NewChunk


class StubEmbedder(Embedder):
    """Maps exact texts to fixed 4-d vectors (unknown text -> orthogonal 'nothing' axis)."""

    name = "stub"
    dim = 4

    def __init__(self, table: dict[str, list[float]]):
        self.table = table

    async def embed(self, texts):
        return [self.table.get(t, [0.0, 0.0, 0.0, 1.0]) for t in texts]


class ReverseReranker(Reranker):
    """Scores in reverse candidate order: proves the reranker, not RRF, decides the final order."""

    async def score(self, query, texts):
        return [float(i) for i in range(len(texts))]


async def _store(docs: dict[str, tuple[str, list[float]]]) -> InMemoryVectorStore:
    store = InMemoryVectorStore(4, "stub")
    for source, (text, vec) in docs.items():
        await store.replace_document(
            source, "h", [NewChunk(text, 0, {"source": source, "date": "2026-01-01", "doc_type": "notes"}, vec)]
        )
    return store


# axis 0: "pooling", axis 1: "resilience", axis 2: "storage"
BM25_DOC = "Set PGBOUNCER_MAX_CLIENT_CONN to 400 on the ledger pooler before the migration."
DECOY_DOC = "Connection pooling overview: pools keep database connections warm for reuse."
DENSE_DOC = "The circuit breaker opens after repeated failures and fails fast for thirty seconds."


async def _corpus():
    return await _store(
        {
            "bm25.md": (BM25_DOC, [0.0, 0.0, 1.0, 0.0]),  # vector is far from the identifier query
            "decoy.md": (DECOY_DOC, [1.0, 0.0, 0.0, 0.0]),
            "dense.md": (DENSE_DOC, [0.0, 1.0, 0.0, 0.0]),
            "filler.md": ("Unrelated notes about lunch menus and parking.", [0.0, 0.0, 0.0, 1.0]),
        }
    )


async def test_bm25_leg_rescues_exact_identifier_the_dense_leg_misses():
    store = await _corpus()
    # the query embeds right next to the decoy; dense_k=1 means dense returns ONLY the decoy
    emb = StubEmbedder({"PGBOUNCER_MAX_CLIENT_CONN": [1.0, 0.0, 0.0, 0.0]})
    retriever = HybridRetriever(store, emb, SimpleReranker(), dense_k=1)

    cands = {c.chunk.source: c for c in await retriever.candidates("PGBOUNCER_MAX_CLIENT_CONN")}
    assert cands["decoy.md"].dense_rank == 1 and cands["decoy.md"].bm25_rank is None
    assert cands["bm25.md"].dense_rank is None and cands["bm25.md"].bm25_rank == 1

    results = await retriever.search("PGBOUNCER_MAX_CLIENT_CONN")
    assert results[0].source == "bm25.md"


async def test_dense_leg_finds_paraphrase_with_no_keyword_overlap():
    store = await _corpus()
    query = "how do we stop hammering a dependency that is down"
    assert not ({"breaker", "circuit", "failures"} & set(query.split()))  # truly no lexical bridge
    emb = StubEmbedder({query: [0.1, 0.99, 0.0, 0.0]})
    retriever = HybridRetriever(store, emb, ReverseReranker(), dense_k=2, top_n=5)

    cands = await retriever.candidates(query)
    assert all(c.bm25_rank is None for c in cands)  # BM25 found nothing
    assert cands[0].chunk.source == "dense.md" and cands[0].dense_rank == 1
    # with an order-preserving-ish reranker the paraphrase target is surfaced too
    good = HybridRetriever(store, emb, SimpleReranker(), dense_k=2, top_n=1)
    assert [c.source for c in await good.search(query)] == ["dense.md"]


async def test_chunk_found_by_both_legs_appears_once_with_higher_fused_score():
    store = await _corpus()
    emb = StubEmbedder({"circuit breaker failures": [0.0, 1.0, 0.0, 0.0]})
    retriever = HybridRetriever(store, emb, SimpleReranker())
    cands = await retriever.candidates("circuit breaker failures")
    sources = [c.chunk.source for c in cands]
    assert len(sources) == len(set(sources))  # deduped
    top = cands[0]
    assert top.chunk.source == "dense.md" and top.dense_rank == 1 and top.bm25_rank == 1
    assert top.rrf_score > cands[1].rrf_score


async def test_identical_passage_under_two_sources_is_deduped():
    store = await _store(
        {
            "a.md": ("Same passage about retries.", [1.0, 0.0, 0.0, 0.0]),
            "b.md": ("Same  passage about retries.", [1.0, 0.0, 0.0, 0.0]),  # differs only in whitespace
        }
    )
    retriever = HybridRetriever(store, StubEmbedder({"retries": [1.0, 0.0, 0.0, 0.0]}), SimpleReranker())
    assert len(await retriever.search("retries")) == 1


async def test_returns_at_most_five_unique_citations_after_rerank():
    docs = {f"doc{i}.md": (f"retry backoff policy variant {i} with jitter", [1.0, 0.0, 0.0, 0.0]) for i in range(12)}
    store = await _store(docs)
    retriever = HybridRetriever(store, StubEmbedder({"retry backoff jitter": [1.0, 0.0, 0.0, 0.0]}), SimpleReranker())
    assert len(await retriever.candidates("retry backoff jitter")) == 12  # 20+20 pool, deduped
    results = await retriever.search("retry backoff jitter")
    assert len(results) == 5
    assert len({c.ref for c in results}) == 5
    assert all(isinstance(c, Citation) for c in results)
    assert [c.score for c in results] == sorted((c.score for c in results), reverse=True)


async def test_candidate_pool_capped_at_dense_k_plus_bm25_k():
    docs = {f"d{i}.md": (f"alpha beta gamma {i}", [1.0, float(i) / 100, 0.0, 0.0]) for i in range(60)}
    store = await _store(docs)
    retriever = HybridRetriever(store, StubEmbedder({"alpha beta": [1.0, 0.0, 0.0, 0.0]}), SimpleReranker())
    assert 20 <= len(await retriever.candidates("alpha beta")) <= 40


async def test_reranker_decides_final_order():
    store = await _corpus()
    emb = StubEmbedder({"circuit breaker failures": [0.0, 1.0, 0.0, 0.0]})
    fused = [c.chunk.source for c in await HybridRetriever(store, emb, SimpleReranker()).candidates("circuit breaker failures")]
    reranked = [c.source for c in await HybridRetriever(store, emb, ReverseReranker()).search("circuit breaker failures")]
    assert reranked == list(reversed(fused))[:5]


async def test_citation_fields_and_ref():
    store = await _corpus()
    retriever = HybridRetriever(store, StubEmbedder({"circuit breaker": [0.0, 1.0, 0.0, 0.0]}), SimpleReranker())
    c = (await retriever.search("circuit breaker"))[0]
    assert (c.source, c.chunk_id, c.doc_type, c.date) == ("dense.md", 0, "notes", "2026-01-01")
    assert c.ref == "dense.md#0" and c.text == DENSE_DOC
    assert set(c.to_dict()) >= {"ref", "source", "chunk_id", "score", "text", "doc_type", "date"}


async def test_min_score_filters_weak_matches():
    store = await _corpus()
    emb = StubEmbedder({"circuit breaker": [0.0, 1.0, 0.0, 0.0]})
    strict = HybridRetriever(store, emb, SimpleReranker(), min_score=0.9)
    results = await strict.search("circuit breaker")
    assert [c.source for c in results] == ["dense.md"]


async def test_empty_store_returns_nothing():
    empty = HybridRetriever(InMemoryVectorStore(4, "stub"), StubEmbedder({}), SimpleReranker())
    assert await empty.search("anything") == []


async def test_bm25_index_refreshes_when_store_changes():
    store = await _corpus()
    retriever = HybridRetriever(store, StubEmbedder({}), SimpleReranker(), dense_k=1)
    assert [c.source for c in await retriever.search("zebrafish")] != ["new.md"]
    await store.replace_document("new.md", "h2", [NewChunk("zebrafish husbandry notes", 0, {}, [0.0, 0.0, 0.0, 1.0])])
    assert (await retriever.search("zebrafish"))[0].source == "new.md"
