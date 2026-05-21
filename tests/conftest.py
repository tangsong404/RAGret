from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ragret.cache import IndexCache, ModelCache
from server.store.pool import SqliteConnectionPool
from server.store.sqlite_store import SqliteAppStore


@pytest.fixture
def pool(tmp_path: Path) -> SqliteConnectionPool:
    db = tmp_path / "test_app.sqlite"
    p = SqliteConnectionPool(db, min_size=2, max_size=4)
    yield p
    p.close()


@pytest.fixture
def store(pool: SqliteConnectionPool) -> SqliteAppStore:
    return SqliteAppStore(pool)


@pytest.fixture
def model_cache() -> MagicMock:
    cache = MagicMock(spec=ModelCache)
    cache.embed_query.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    return cache


@pytest.fixture
def index_cache(tmp_path: Path) -> IndexCache:
    return IndexCache(max_entries=8)
