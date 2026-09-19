"""OPTIONAL smoke test of the real sentence-transformers models. Auto-skips when the models are
neither cached locally nor downloadable (offline CI), or when JARVIS_SKIP_MODEL_DOWNLOAD=1.
Everything else in Phase 4 uses HashingEmbedder / SimpleReranker."""
import os
import socket
from pathlib import Path

import pytest

from app.config import get_settings
from app.rag.embedder import SentenceTransformerEmbedder
from app.rag.ingest import ingest_path
from app.rag.reranker import CrossEncoderReranker
from app.rag.retriever import HybridRetriever
from app.rag.store import InMemoryVectorStore

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"


def _models_reachable() -> str | None:
    """None if usable, else a skip reason."""
    if os.environ.get("JARVIS_SKIP_MODEL_DOWNLOAD") == "1":
        return "JARVIS_SKIP_MODEL_DOWNLOAD=1"
    try:
        from huggingface_hub import snapshot_download

        settings = get_settings()
        cached = True
        for repo in (settings.embedding_model, settings.reranker_model):
            try:
                snapshot_download(repo, local_files_only=True)
            except Exception:  # noqa: BLE001 - model download unavailable -> skip
                cached = False
        if cached:
            return None
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        return None
    except Exception as exc:  # noqa: BLE001 - model download unavailable -> skip
        return f"real models unavailable ({type(exc).__name__}: {exc})"


@pytest.fixture(scope="module")
def real_models():
    reason = _models_reachable()
    if reason:
        pytest.skip(reason)
    embedder, reranker = SentenceTransformerEmbedder(), CrossEncoderReranker()
    try:
        embedder._load()
        reranker._predict("warmup", ["warmup"])
    except Exception as exc:  # noqa: BLE001 - model download unavailable -> skip
        pytest.skip(f"could not load real models: {type(exc).__name__}: {exc}")
    return embedder, reranker


async def test_real_embedder_and_reranker_retrieve_from_sample_docs(real_models):
    embedder, reranker = real_models
    vecs = await embedder.embed(["hello world", "goodbye"])
    assert len(vecs) == 2 and len(vecs[0]) == embedder.dim == 384
    assert abs(sum(x * x for x in vecs[0]) - 1.0) < 1e-3  # normalized

    store = InMemoryVectorStore(embedder.dim, embedder.name)
    await ingest_path(DOCS_DIR, embedder, store)
    retriever = HybridRetriever(store, embedder, reranker)
    top = await retriever.search("How long do we wait before paging the secondary on-call for a SEV1?")
    assert top[0].source == "oncall-escalation-policy.md"
    # a paraphrase with little lexical overlap
    top = await retriever.search("what stops clients from hammering a struggling downstream service?")
    assert {c.source for c in top[:3]} & {"adr-0007-retry-backoff.md", "incident-2026-02-ledger-retry-storm.md"}
