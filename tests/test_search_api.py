from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ragret.cache import ModelCache
from server.config import Settings
from server.main import create_app
from server.store.pool import SqliteConnectionPool
from server.store.sqlite_store import SqliteAppStore


@pytest.fixture
def store(tmp_path: Path):
    pool = SqliteConnectionPool(tmp_path / "test.sqlite", min_size=1, max_size=2)
    s = SqliteAppStore(pool)
    yield s
    pool.close()


@pytest.fixture
def model_cache():
    m = MagicMock(spec=ModelCache)
    m.embed_query.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    m.rerank.return_value = []
    return m


@pytest.fixture
def client(store, model_cache):
    app = create_app(store=store, model_cache=model_cache, settings=Settings())
    with TestClient(app) as c:
        yield c


class TestSearch:
    def test_missing_query_param(self, client: TestClient):
        resp = client.get("/api/search/myindex")
        assert resp.status_code == 422

    def test_unknown_index(self, client: TestClient):
        resp = client.get("/api/search/nonexistent?query=hello")
        assert resp.status_code == 404

    def test_empty_index(self, client: TestClient, store: SqliteAppStore, tmp_path: Path):
        db = tmp_path / "mykb.sqlite"
        conn = sqlite3.connect(db)
        conn.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                content TEXT NOT NULL,
                metadata_json TEXT,
                embedding BLOB NOT NULL,
                UNIQUE(source, chunk_index)
            );
            INSERT INTO meta(key, value) VALUES('embed_dim', '3');
            """
        )
        conn.close()

        reg = client.post(
            "/api/auth/register",
            json={"username": "owner", "password": "secret123"},
        ).json()
        uid = reg["user"]["id"]
        token = reg["token"]
        store.create_knowledge_base(
            name="mykb",
            description="test",
            readme_md="",
            db_path=str(db),
            owner_id=uid,
        )
        resp = client.get(
            "/api/search/mykb?query=hello",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 500
        assert "empty" in resp.json()["detail"].lower()
