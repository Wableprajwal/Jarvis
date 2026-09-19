"""Vector stores for RAG.

* `PgVectorStore`      pgvector `vector(dim)` column + HNSW (cosine) index. Preferred.
* `PgArrayVectorStore` fallback for Postgres without the pgvector extension: `float8[]` column,
                       cosine computed in numpy in-process (fine up to ~tens of thousands of chunks).
* `InMemoryVectorStore` no database at all; used by the eval harness / CI and unit tests.
* `create_vector_store` factory picks between the Postgres two.

All stores keep chunk text + JSONB-style metadata next to the vector (BM25 is built from the
same rows via `all_chunks`) and are scoped to one embedder (`embedder_name`): vectors from
different embedders are never compared. Ingestion is idempotent per (source, content_hash).
Tables are created by `init()`, not by app/schema.sql.
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NewChunk:
    text: str
    chunk_index: int
    metadata: dict
    embedding: list[float]


@dataclass
class StoredChunk:
    id: int
    source: str
    chunk_index: int
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoredChunk:
    chunk: StoredChunk
    score: float  # cosine similarity in [-1, 1]


class VectorStore(ABC):
    dim: int
    embedder_name: str

    @abstractmethod
    async def init(self) -> None:
        """Create tables/indexes if missing. Idempotent."""

    @abstractmethod
    async def has_document(self, source: str, content_hash: str) -> bool: ...

    @abstractmethod
    async def replace_document(self, source: str, content_hash: str, chunks: list[NewChunk]) -> int:
        """Atomically replace every chunk of `source` with `chunks`. Returns the number stored."""

    @abstractmethod
    async def delete_source(self, source: str) -> int: ...

    @abstractmethod
    async def dense_search(self, query_vec: list[float], k: int) -> list[ScoredChunk]:
        """Top-k chunks by cosine similarity, best first."""

    @abstractmethod
    async def all_chunks(self) -> list[StoredChunk]:
        """Every chunk (for building the BM25 index)."""

    @abstractmethod
    async def fingerprint(self) -> str:
        """Cheap value that changes whenever the stored chunks change (BM25 cache invalidation)."""

    def _check_dim(self, vec: list[float]) -> None:
        if len(vec) != self.dim:
            raise ValueError(f"embedding has {len(vec)} dims, store expects {self.dim}")


def _cosine_top_k(matrix: np.ndarray, query: list[float], k: int) -> list[tuple[int, float]]:
    """(row index, cosine) for the top-k rows of `matrix`. Handles un-normalized inputs."""
    if matrix.shape[0] == 0 or k <= 0:
        return []
    q = np.asarray(query, dtype=np.float64)
    qn = np.linalg.norm(q)
    norms = np.linalg.norm(matrix, axis=1)
    denom = norms * qn
    sims = np.divide(matrix @ q, denom, out=np.zeros(matrix.shape[0]), where=denom > 0)
    k = min(k, len(sims))
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top], kind="stable")]
    return [(int(i), float(sims[i])) for i in top]


# --------------------------------------------------------------------------------- in-memory
class InMemoryVectorStore(VectorStore):
    """Same interface, no database: deterministic CI/eval runs and unit tests."""

    def __init__(self, dim: int, embedder_name: str = "default"):
        self.dim = dim
        self.embedder_name = embedder_name
        self._chunks: list[StoredChunk] = []
        self._vecs: list[list[float]] = []
        self._hashes: dict[str, str] = {}
        self._next_id = 1
        self._version = 0

    async def init(self) -> None:
        return None

    async def has_document(self, source: str, content_hash: str) -> bool:
        return self._hashes.get(source) == content_hash

    async def replace_document(self, source: str, content_hash: str, chunks: list[NewChunk]) -> int:
        for c in chunks:
            self._check_dim(c.embedding)
        await self.delete_source(source)
        for c in chunks:
            self._chunks.append(StoredChunk(self._next_id, source, c.chunk_index, c.text, dict(c.metadata)))
            self._vecs.append(list(c.embedding))
            self._next_id += 1
        self._hashes[source] = content_hash
        self._version += 1
        return len(chunks)

    async def delete_source(self, source: str) -> int:
        keep = [i for i, c in enumerate(self._chunks) if c.source != source]
        removed = len(self._chunks) - len(keep)
        self._chunks = [self._chunks[i] for i in keep]
        self._vecs = [self._vecs[i] for i in keep]
        self._hashes.pop(source, None)
        if removed:
            self._version += 1
        return removed

    async def dense_search(self, query_vec: list[float], k: int) -> list[ScoredChunk]:
        self._check_dim(query_vec)
        if not self._chunks:
            return []
        matrix = np.asarray(self._vecs, dtype=np.float64)
        return [ScoredChunk(self._chunks[i], s) for i, s in _cosine_top_k(matrix, query_vec, k)]

    async def all_chunks(self) -> list[StoredChunk]:
        return list(self._chunks)

    async def fingerprint(self) -> str:
        return f"mem:{self._version}:{len(self._chunks)}"


# --------------------------------------------------------------------------------- postgres
_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class _PgStoreBase(VectorStore):
    # Subclasses define the embedding column type and how it is written/read/searched.
    def __init__(self, pool, dim: int, embedder_name: str, table: str = "rag_chunks"):
        if not _IDENT_RE.match(table):
            raise ValueError(f"invalid table name: {table!r}")
        self.pool = pool
        self.dim = dim
        self.embedder_name = embedder_name
        self.table = table

    @abstractmethod
    def _embedding_ddl(self) -> str: ...

    async def _extra_init(self, conn) -> None:
        return None

    async def init(self) -> None:
        t = self.table
        async with self.pool.acquire() as conn:
            await self._pre_init(conn)
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {t} (
                    id           BIGSERIAL PRIMARY KEY,
                    source       TEXT NOT NULL,
                    chunk_index  INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedder     TEXT NOT NULL,
                    text         TEXT NOT NULL,
                    metadata     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding    {self._embedding_ddl()} NOT NULL,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (embedder, source, chunk_index)
                )
                """
            )
            await conn.execute(f"CREATE INDEX IF NOT EXISTS ix_{t}_source ON {t} (embedder, source)")
            await self._extra_init(conn)

    async def _pre_init(self, conn) -> None:
        return None

    @abstractmethod
    def _embedding_param(self, vec: list[float]): ...

    @abstractmethod
    def _embedding_placeholder(self, n: int) -> str: ...

    async def has_document(self, source: str, content_hash: str) -> bool:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT 1 FROM {self.table} WHERE embedder=$1 AND source=$2 AND content_hash=$3 LIMIT 1",
                self.embedder_name, source, content_hash,
            )
        return row is not None

    async def replace_document(self, source: str, content_hash: str, chunks: list[NewChunk]) -> int:
        for c in chunks:
            self._check_dim(c.embedding)
        rows = [
            (
                source, c.chunk_index, content_hash, self.embedder_name, c.text,
                json.dumps(c.metadata), self._embedding_param(c.embedding),
            )
            for c in chunks
        ]
        insert = (
            f"INSERT INTO {self.table} (source, chunk_index, content_hash, embedder, text, metadata, embedding) "
            f"VALUES ($1,$2,$3,$4,$5,$6::jsonb,{self._embedding_placeholder(7)})"
        )
        async with self.pool.acquire() as conn, conn.transaction():
            # serialize concurrent ingests of the same source
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", f"{self.table}:{source}")
            await conn.execute(
                f"DELETE FROM {self.table} WHERE embedder=$1 AND source=$2", self.embedder_name, source
            )
            if rows:
                await conn.executemany(insert, rows)
        self._invalidate()
        return len(rows)

    async def delete_source(self, source: str) -> int:
        async with self.pool.acquire() as conn:
            status = await conn.execute(
                f"DELETE FROM {self.table} WHERE embedder=$1 AND source=$2", self.embedder_name, source
            )
        self._invalidate()
        return int(status.split()[-1])

    def _invalidate(self) -> None:
        return None

    @staticmethod
    def _row_to_chunk(r) -> StoredChunk:
        meta = r["metadata"]
        return StoredChunk(
            id=r["id"], source=r["source"], chunk_index=r["chunk_index"], text=r["text"],
            metadata=json.loads(meta) if isinstance(meta, str) else dict(meta),
        )

    async def all_chunks(self) -> list[StoredChunk]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, source, chunk_index, text, metadata::text AS metadata FROM {self.table} "
                f"WHERE embedder=$1 ORDER BY source, chunk_index",
                self.embedder_name,
            )
        return [self._row_to_chunk(r) for r in rows]

    async def fingerprint(self) -> str:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT count(*) AS n, coalesce(max(id),0) AS m FROM {self.table} WHERE embedder=$1",
                self.embedder_name,
            )
        return f"pg:{row['n']}:{row['m']}"


class PgVectorStore(_PgStoreBase):
    """pgvector. Cosine distance via `<=>`, HNSW index (m=16, ef_construction=64).

    Note: HNSW requires a fixed-dimension column, so one table holds one embedding dimension."""

    def _embedding_ddl(self) -> str:
        return f"vector({self.dim})"

    async def _pre_init(self, conn) -> None:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    async def _extra_init(self, conn) -> None:
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS ix_{self.table}_hnsw ON {self.table} "
            f"USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        )

    def _embedding_param(self, vec: list[float]) -> str:
        return _vec_literal(vec)  # text literal, cast server-side: no client codec needed

    def _embedding_placeholder(self, n: int) -> str:
        return f"${n}::text::vector"  # text param (a type asyncpg knows), cast server-side

    async def dense_search(self, query_vec: list[float], k: int) -> list[ScoredChunk]:
        self._check_dim(query_vec)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, source, chunk_index, text, metadata::text AS metadata, "
                f"1 - (embedding <=> $1::text::vector) AS score FROM {self.table} WHERE embedder=$2 "
                f"ORDER BY embedding <=> $1::text::vector LIMIT $3",
                _vec_literal(query_vec), self.embedder_name, k,
            )
        return [ScoredChunk(self._row_to_chunk(r), float(r["score"])) for r in rows]


class PgArrayVectorStore(_PgStoreBase):
    """Fallback without pgvector: `float8[]` column; the embedding matrix is cached in-process
    (invalidated by `fingerprint`) and cosine similarity is computed with numpy."""

    def __init__(self, pool, dim: int, embedder_name: str, table: str = "rag_chunks"):
        super().__init__(pool, dim, embedder_name, table)
        self._cache: tuple[str, list[StoredChunk], np.ndarray] | None = None

    def _embedding_ddl(self) -> str:
        return "DOUBLE PRECISION[]"

    def _embedding_param(self, vec: list[float]) -> list[float]:
        return [float(x) for x in vec]

    def _embedding_placeholder(self, n: int) -> str:
        return f"${n}"

    def _invalidate(self) -> None:
        self._cache = None

    async def _load(self) -> tuple[list[StoredChunk], np.ndarray]:
        fp = await self.fingerprint()
        if self._cache is not None and self._cache[0] == fp:
            return self._cache[1], self._cache[2]
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, source, chunk_index, text, metadata::text AS metadata, embedding FROM {self.table} "
                f"WHERE embedder=$1 ORDER BY id",
                self.embedder_name,
            )
        chunks = [self._row_to_chunk(r) for r in rows]
        matrix = np.asarray([r["embedding"] for r in rows], dtype=np.float64).reshape(len(rows), self.dim)
        self._cache = (fp, chunks, matrix)
        return chunks, matrix

    async def dense_search(self, query_vec: list[float], k: int) -> list[ScoredChunk]:
        self._check_dim(query_vec)
        chunks, matrix = await self._load()
        return [ScoredChunk(chunks[i], s) for i, s in _cosine_top_k(matrix, query_vec, k)]


# --------------------------------------------------------------------------------- factory
async def _table_embedding_type(conn, table: str) -> str | None:
    """'vector' / '_float8' / ... for an existing table's embedding column, None if no table."""
    return await conn.fetchval(
        "SELECT t.typname FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
        "JOIN pg_type t ON t.oid = a.atttypid "
        "WHERE c.relname = $1 AND a.attname = 'embedding' AND c.relkind = 'r' AND pg_table_is_visible(c.oid)",
        table,
    )


async def create_vector_store(
    pool, dim: int, embedder_name: str, *, table: str = "rag_chunks", prefer: str = "auto"
) -> VectorStore:
    """Build and `init()` a Postgres vector store.

    prefer="auto"     use pgvector if the extension is available (trying CREATE EXTENSION), else
                      the numpy array store. If `table` already exists, stay with its column type
                      so data is never silently orphaned.
    prefer="pgvector" require pgvector (raises RuntimeError if unavailable).
    prefer="array"    force the array store.
    """
    if prefer not in {"auto", "pgvector", "array"}:
        raise ValueError(f"unknown store preference: {prefer!r}")
    use_pgvector = False
    if prefer != "array":
        async with pool.acquire() as conn:
            existing = await _table_embedding_type(conn, table)
            if existing is not None and prefer == "auto":
                use_pgvector = existing == "vector"
            else:
                use_pgvector = await _try_enable_pgvector(conn)
        if prefer == "pgvector" and not use_pgvector:
            raise RuntimeError("pgvector extension is not available in this Postgres")
    cls = PgVectorStore if use_pgvector else PgArrayVectorStore
    if not use_pgvector and prefer == "auto":
        logger.info("pgvector not available; using numpy array vector store")
    store = cls(pool, dim, embedder_name, table)
    await store.init()
    return store


async def _try_enable_pgvector(conn) -> bool:
    available = await conn.fetchval("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
    if not available:
        return False
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception as exc:  # noqa: BLE001 - probing for the pgvector extension; any failure means 'not available'
        logger.warning("could not enable pgvector (%s); falling back to array store", exc)
        return False
    return bool(await conn.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
