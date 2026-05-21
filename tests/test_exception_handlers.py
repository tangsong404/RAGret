from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from server.config import Settings
from server.main import create_app
from server.store.pool import SqliteConnectionPool
from server.store.sqlite_store import SqliteAppStore


def test_http_error_uses_ok_false_shape(tmp_path: Path):
    pool = SqliteConnectionPool(tmp_path / "t.sqlite", min_size=1, max_size=2)
    store = SqliteAppStore(pool)
    try:
        app = create_app(store=store, settings=Settings(), repo_root=tmp_path)
        with TestClient(app) as client:
            resp = client.get("/api/auth/me")
            assert resp.status_code == 401
            body = resp.json()
            assert body == {"ok": False, "error": "Not logged in"}
    finally:
        pool.close()
