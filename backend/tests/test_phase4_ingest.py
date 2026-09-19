"""Ingest idempotency on the real Postgres `db` fixture (array store), the store factory, and
the DB-free InMemoryVectorStore end to end. No model, no network."""
import uuid
from pathlib import Path

import pytest

from app.rag.embedder import HashingEmbedder
from app.rag.ingest import ingest_path, parse_front_matter
from app.rag.reranker import SimpleReranker
from app.rag.retriever import HybridRetriever
from app.rag.store import InMemoryVectorStore, PgArrayVectorStore, PgVectorStore, create_vector_store

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"


@pytest.fixture
def emb():
    return HashingEmbedder(dim=128)


@pytest.fixture
async def pg_store(db, emb):
    table = f"rag_test_{uuid.uuid4().hex[:8]}"
    store = PgArrayVectorStore(db.pool, emb.dim, emb.name, table)
    await store.init()
    yield store
    async with db.pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {table}")


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "retry.md").write_text(
        "---\ndoc_type: adr\ndate: 2026-03-04\n---\n# Retries\nUse exponential backoff with a 10 second cap.\n"
    )
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "notes.txt").write_text("Postgres HNSW index uses m=16.")
    (tmp_path / "util.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "ignore.png").write_bytes(b"\x89PNG")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.md").write_text("should never be ingested")
    return tmp_path


async def _count(db, table: str) -> int:
    async with db.pool.acquire() as conn:
        return await conn.fetchval(f"SELECT count(*) FROM {table}")


def test_front_matter_parsing():
    meta, body = parse_front_matter("---\ndoc_type: runbook\ndate: 2026-01-02\n---\nBody")
    assert meta == {"doc_type": "runbook", "date": "2026-01-02"} and body == "Body"
    assert parse_front_matter("no front matter") == ({}, "no front matter")


async def test_ingest_is_idempotent_on_postgres(db, emb, pg_store, corpus):
    first = await ingest_path(corpus, emb, pg_store)
    assert sorted(first.ingested) == ["retry.md", "sub/notes.txt", "util.py"]  # png + node_modules skipped
    assert first.skipped == [] and first.chunks == 3
    assert await _count(db, pg_store.table) == 3

    second = await ingest_path(corpus, emb, pg_store)
    assert second.ingested == [] and sorted(second.skipped) == ["retry.md", "sub/notes.txt", "util.py"]
    assert second.chunks == 0
    assert await _count(db, pg_store.table) == 3  # no duplicates


async def test_changed_file_replaces_its_chunks_only(db, emb, pg_store, corpus):
    await ingest_path(corpus, emb, pg_store)
    (corpus / "retry.md").write_text("---\ndoc_type: adr\n---\nNow we use decorrelated jitter, 8 attempts.")
    report = await ingest_path(corpus, emb, pg_store)
    assert report.ingested == ["retry.md"] and len(report.skipped) == 2
    assert await _count(db, pg_store.table) == 3
    texts = {c.source: c.text for c in await pg_store.all_chunks()}
    assert "decorrelated jitter" in texts["retry.md"] and "exponential" not in texts["retry.md"]


async def test_metadata_roundtrip_and_dense_search(emb, pg_store, corpus):
    await ingest_path(corpus, emb, pg_store)
    chunks = {c.source: c for c in await pg_store.all_chunks()}
    assert chunks["retry.md"].metadata == {"source": "retry.md", "date": "2026-03-04", "doc_type": "adr"}
    assert chunks["util.py"].metadata["doc_type"] == "code"

    hits = await pg_store.dense_search(await emb.embed_query("exponential backoff cap"), k=3)
    assert hits[0].chunk.source == "retry.md"
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert -1.0 <= hits[0].score <= 1.0


async def test_store_is_scoped_to_its_embedder(db, emb, pg_store, corpus):
    await ingest_path(corpus, emb, pg_store)
    other = PgArrayVectorStore(db.pool, 64, "other-embedder", pg_store.table)
    assert await other.all_chunks() == []
    assert not await other.has_document("retry.md", "whatever")


async def test_wrong_dimension_rejected(emb, pg_store):
    with pytest.raises(ValueError):
        await pg_store.dense_search([0.1] * 5, 3)


async def test_invalid_table_name_rejected(db, emb):
    with pytest.raises(ValueError):
        PgArrayVectorStore(db.pool, emb.dim, emb.name, "rag; DROP TABLE users")


async def test_factory_picks_store_by_extension_availability(db, emb):
    table = f"rag_test_{uuid.uuid4().hex[:8]}"
    try:
        async with db.pool.acquire() as conn:
            has_pgvector = bool(await conn.fetchval("SELECT 1 FROM pg_available_extensions WHERE name='vector'"))
        store = await create_vector_store(db.pool, emb.dim, emb.name, table=table)
        assert isinstance(store, PgVectorStore if has_pgvector else PgArrayVectorStore)
        forced = await create_vector_store(db.pool, emb.dim, emb.name, table=f"{table}_a", prefer="array")
        assert isinstance(forced, PgArrayVectorStore)
        if not has_pgvector:
            with pytest.raises(RuntimeError):
                await create_vector_store(db.pool, emb.dim, emb.name, table=f"{table}_v", prefer="pgvector")
    finally:
        async with db.pool.acquire() as conn:
            for suffix in ("", "_a", "_v"):
                await conn.execute(f"DROP TABLE IF EXISTS {table}{suffix}")


async def test_factory_reuses_existing_table_type(db, emb):
    table = f"rag_test_{uuid.uuid4().hex[:8]}"
    try:
        array = await create_vector_store(db.pool, emb.dim, emb.name, table=table, prefer="array")
        again = await create_vector_store(db.pool, emb.dim, emb.name, table=table)  # auto
        assert type(again) is type(array)
    finally:
        async with db.pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {table}")


async def test_real_docs_ingest_and_retrieve_on_postgres(emb, pg_store):
    report = await ingest_path(DOCS_DIR, emb, pg_store)
    assert len(report.ingested) >= 6
    retriever = HybridRetriever(pg_store, emb, SimpleReranker())
    top = (await retriever.search("what is the maximum number of retry attempts and the backoff cap?"))[0]
    assert top.source == "adr-0007-retry-backoff.md" and top.doc_type == "adr" and top.date == "2026-03-04"


# ---------------------------------------------------- InMemoryVectorStore (no DB: eval / CI path)
async def test_in_memory_store_ingest_and_hybrid_retrieve_end_to_end(emb):
    store = InMemoryVectorStore(emb.dim, emb.name)
    first = await ingest_path(DOCS_DIR, emb, store)
    assert len(first.ingested) >= 6 and first.chunks >= len(first.ingested)
    second = await ingest_path(DOCS_DIR, emb, store)  # idempotent here too
    assert second.ingested == [] and len(second.skipped) == len(first.ingested)
    assert len(await store.all_chunks()) == first.chunks

    retriever = HybridRetriever(store, emb, SimpleReranker())
    cases = {
        "who gets paged if the primary on-call does not acknowledge a SEV1": "oncall-escalation-policy.md",
        "what percentage of traffic does the canary get and for how long": "ci-deploy-checklist.md",
        "hnsw ef_search recommended value": "postgres-indexing-notes.md",
        "token bucket burst capacity for the standard tier": "api-rate-limiting-design.md",
    }
    for query, expected in cases.items():
        results = await retriever.search(query)
        assert results and results[0].source == expected, (query, [c.ref for c in results])
        assert len(results) <= 5


async def test_in_memory_store_replaces_changed_document(emb, tmp_path):
    f = tmp_path / "a.md"
    f.write_text("alpha beta gamma")
    store = InMemoryVectorStore(emb.dim, emb.name)
    await ingest_path(f, emb, store)
    f.write_text("delta epsilon")
    report = await ingest_path(f, emb, store)
    assert report.ingested == ["a.md"]
    assert [c.text for c in await store.all_chunks()] == ["delta epsilon"]
