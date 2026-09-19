import os
import re
import uuid

import asyncpg
import pytest

from app.config import get_settings
from app.db import Database

_DB_NAME = re.compile(r"^[A-Za-z0-9_]+$")


def database_url_for_tests() -> str:
    """Tests get their OWN database, never the one the dev server writes to.

    Sharing it meant a `make test` run left hundreds of mock-latency trace rows in the dev
    database, which then skewed the /metrics dashboard (p50 of 0.01 ms from MockProvider runs).
    `TEST_DATABASE_URL` overrides; otherwise the configured database name gets a `_test` suffix.
    """
    explicit = os.environ.get("TEST_DATABASE_URL")
    if explicit:
        return explicit
    base, _, name = get_settings().database_url.rpartition("/")
    name = name.split("?", 1)[0]
    return f"{base}/{name}_test"


async def _ensure_database(url: str) -> None:
    """Create the test database if it does not exist yet (first run on a new machine)."""
    base, _, name = url.rpartition("/")
    if not _DB_NAME.match(name):  # name comes from local config, but CREATE DATABASE cannot be parameterized
        raise ValueError(f"unsafe test database name: {name!r}")
    admin = await asyncpg.connect(f"{base}/postgres")
    try:
        if not await admin.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name):
            await admin.execute(f'CREATE DATABASE "{name}"')
    except asyncpg.InsufficientPrivilegeError as exc:  # role lacks CREATEDB
        raise RuntimeError(
            f"the test database {name!r} does not exist and this role may not create it. "
            f'Run: createdb -O <owner> {name}   (or `make setup-db`)'
        ) from exc
    finally:
        await admin.close()


@pytest.fixture
async def db():
    """A real Postgres connection; tests that need it are skipped when none is reachable."""
    url = database_url_for_tests()
    try:
        try:
            database = await Database.connect(url)
        except asyncpg.InvalidCatalogNameError:
            await _ensure_database(url)
            database = await Database.connect(url)
    except Exception as exc:  # noqa: BLE001 - any connection problem should skip, not error the suite
        pytest.skip(f"postgres not available: {exc}")
    yield database
    await database.close()


@pytest.fixture
def session_id() -> str:
    return f"test-{uuid.uuid4().hex[:10]}"
