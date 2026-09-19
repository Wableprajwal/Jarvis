import os
import uuid

import pytest

from app.config import get_settings
from app.db import Database


@pytest.fixture
async def db():
    """A real Postgres connection; tests that need it are skipped when none is reachable."""
    url = os.environ.get("TEST_DATABASE_URL") or get_settings().database_url
    try:
        database = await Database.connect(url)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"postgres not available: {exc}")
    yield database
    await database.close()


@pytest.fixture
def session_id() -> str:
    return f"test-{uuid.uuid4().hex[:10]}"
