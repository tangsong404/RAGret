from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from server.config import Settings
from server.main import create_app
from server.store.pool import SqliteConnectionPool
from server.store.sqlite_store import SqliteAppStore


def test_user_api_keys_and_webhook_base(tmp_path: Path):
    pool = SqliteConnectionPool(tmp_path / "u.sqlite", min_size=1, max_size=2)
    store = SqliteAppStore(pool)
    try:
        app = create_app(store=store, settings=Settings(), repo_root=tmp_path)
        with TestClient(app) as client:
            reg = client.post(
                "/api/auth/register",
                json={"username": "patuser", "password": "secret123"},
            ).json()
            h = {"Authorization": f"Bearer {reg['token']}"}

            wb = client.get("/api/webhook-base")
            assert wb.status_code == 200
            assert "gitlab" in wb.json()["bases"]

            created = client.post("/api/user/api-keys", headers=h, json={"name": "ci"}).json()
            assert created["ok"] is True
            assert created["key"]["key"].startswith("sk-")

            listed = client.get("/api/user/api-keys", headers=h).json()["keys"]
            assert len(listed) >= 1

            gen = client.get("/api/user/webhook-secret/generate", headers=h).json()
            assert gen["ok"] is True
            assert gen["secret"]

            client.post("/api/user/gitlab-pat", headers=h, json={"pat": "glpat-test"})
            got = client.get("/api/user/gitlab-pat", headers=h).json()
            assert got["pat"] == "glpat-test"
    finally:
        pool.close()
