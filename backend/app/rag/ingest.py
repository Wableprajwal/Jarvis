"""Ingest .md/.txt/.rst/.py files into a VectorStore.  CLI: `python -m app.rag.ingest <dir>`.

Idempotent per (source, content hash): an unchanged file is skipped, a changed file has its
old chunks atomically replaced. `source` is the path relative to the ingested root (or the
file name when a single file is given), which is also what citations show.

Optional front matter in .md/.txt/.rst files overrides metadata:

    ---
    doc_type: runbook
    date: 2026-03-02
    ---
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.rag.chunking import DEFAULT_CHUNK_TOKENS, DEFAULT_OVERLAP_RATIO, chunk_text
from app.rag.embedder import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from app.rag.store import NewChunk, VectorStore, create_vector_store

SUPPORTED_SUFFIXES = {".md", ".txt", ".rst", ".py"}
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache"}
_DEFAULT_DOC_TYPE = {".md": "markdown", ".txt": "text", ".rst": "rst", ".py": "code"}
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
EMBED_BATCH = 64


@dataclass
class IngestReport:
    scanned: int = 0
    ingested: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # unchanged since last ingest
    chunks: int = 0

    def summary(self) -> str:
        return (
            f"scanned={self.scanned} ingested={len(self.ingested)} "
            f"skipped_unchanged={len(self.skipped)} chunks_written={self.chunks}"
        )


def parse_front_matter(text: str) -> tuple[dict, str]:
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip().lower()] = v.strip()
    return meta, text[m.end():]


def _iter_files(root: Path):
    if root.is_file():
        if root.suffix.lower() in SUPPORTED_SUFFIXES:
            yield root, root.name
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not (set(p.relative_to(root).parts) & _SKIP_DIRS):
            yield p, p.relative_to(root).as_posix()


async def ingest_path(
    path: str | Path,
    embedder: Embedder,
    store: VectorStore,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> IngestReport:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"no such file or directory: {root}")
    report = IngestReport()
    for file, source in _iter_files(root):
        report.scanned += 1
        raw = file.read_bytes()
        content_hash = hashlib.sha256(raw).hexdigest()
        if await store.has_document(source, content_hash):
            report.skipped.append(source)
            continue
        text = raw.decode("utf-8", errors="replace")
        meta, body = parse_front_matter(text) if file.suffix.lower() != ".py" else ({}, text)
        date = meta.get("date") or datetime.fromtimestamp(file.stat().st_mtime, timezone.utc).date().isoformat()
        doc_type = meta.get("doc_type") or _DEFAULT_DOC_TYPE[file.suffix.lower()]
        pieces = chunk_text(
            body, source=source, date=date, doc_type=doc_type,
            chunk_tokens=chunk_tokens, overlap_ratio=overlap_ratio,
        )
        vectors: list[list[float]] = []
        for i in range(0, len(pieces), EMBED_BATCH):
            vectors.extend(await embedder.embed([c.text for c in pieces[i : i + EMBED_BATCH]]))
        new = [NewChunk(c.text, c.index, c.metadata, v) for c, v in zip(pieces, vectors)]
        report.chunks += await store.replace_document(source, content_hash, new)
        report.ingested.append(source)
    return report


def make_embedder(kind: str) -> Embedder:
    return HashingEmbedder() if kind == "hashing" else SentenceTransformerEmbedder()


async def _main(args: argparse.Namespace) -> int:
    from app.config import get_settings
    from app.db import Database

    embedder = make_embedder(args.embedder)
    db = await Database.connect(get_settings().database_url)
    try:
        store = await create_vector_store(db.pool, embedder.dim, embedder.name, prefer=args.store)
        report = await ingest_path(args.path, embedder, store, chunk_tokens=args.chunk_tokens)
    finally:
        await db.close()
    print(f"[{type(store).__name__} / {embedder.name}] {report.summary()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest documents into the Jarvis RAG store.")
    ap.add_argument("path", help="directory or file to ingest (.md/.txt/.rst/.py)")
    ap.add_argument("--embedder", choices=["sentence-transformers", "hashing"], default="sentence-transformers",
                    help="'hashing' is an offline, lower-quality embedder for environments without the model")
    ap.add_argument("--store", choices=["auto", "pgvector", "array"], default="auto")
    ap.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    return asyncio.run(_main(ap.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
