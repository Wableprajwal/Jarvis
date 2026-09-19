from __future__ import annotations

from pathlib import Path

import asyncpg

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class Database:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    @classmethod
    async def connect(cls, url: str, init_schema: bool = True) -> "Database":
        pool = await asyncpg.create_pool(url, min_size=1, max_size=5)
        db = cls(pool)
        if init_schema:
            await db.init_schema()
        return db

    async def init_schema(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(SCHEMA_PATH.read_text())

    async def close(self) -> None:
        await self.pool.close()
