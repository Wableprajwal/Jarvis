---
doc_type: notes
date: 2026-01-19
---
# Postgres indexing notes

## Choosing an index type

- B-tree is the default and handles equality and range queries (`=`, `<`, `BETWEEN`, `ORDER BY`).
- GIN suits `jsonb` containment (`@>`), arrays and full-text search (`tsvector`). Inserts are slower; use
  `fastupdate` with a `gin_pending_list_limit` of 4MB for write-heavy tables.
- BRIN works for huge append-only tables whose column is physically correlated with insert order, such as
  `created_at` on the 400M-row events table. It is about 1,000 times smaller than a B-tree.
- HNSW (pgvector) for nearest-neighbour search. We use `m = 16` and `ef_construction = 64`. At query time set
  `hnsw.ef_search = 40` for a good recall/latency balance; raise it to 100 when recall matters more than speed.

## Practical rules

- Build indexes on live tables with `CREATE INDEX CONCURRENTLY`. It takes about twice as long but does not block
  writes. If it fails it leaves an INVALID index that must be dropped by hand.
- Prefer partial indexes for skewed data. Example: `CREATE INDEX ON orders (customer_id) WHERE status = 'open'`
  is 95% smaller than the full index because only about 5% of orders are open.
- Composite indexes: put equality columns first, then the range column. Column order matters.
- Covering indexes: `INCLUDE (col)` lets index-only scans avoid heap fetches.
- Check for bloat when an index exceeds 30% dead space; run `REINDEX INDEX CONCURRENTLY`.
- Foreign key columns are not indexed automatically. Missing FK indexes are the top cause of slow deletes.

## Diagnosing

Run `EXPLAIN (ANALYZE, BUFFERS)`. A sequential scan on a table above 100k rows with a selective filter usually
means a missing index. Compare `pg_stat_user_indexes.idx_scan`; an index with zero scans over 30 days is a
candidate for removal.
