from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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
def client(store: SqliteAppStore, tmp_path: Path):
    app = create_app(store=store, settings=Settings(), repo_root=tmp_path)
    with TestClient(app) as c:
        yield c


class TestPushApi:
    def test_kb_detail_includes_folder_push_url(
        self, client: TestClient, store: SqliteAppStore, tmp_path: Path
    ):
        reg = client.post(
            "/api/auth/register",
            json={"username": "pushowner", "password": "secret123"},
        ).json()
        token = reg["token"]
        uid = reg["user"]["id"]
        db = tmp_path / "pushkb.sqlite"
        db.touch()
        store.create_knowledge_base(
            name="pushkb",
            description="desc",
            readme_md="",
            db_path=str(db),
            owner_id=uid,
            webhook_secret="sec-push-1",
        )
        detail = client.get(
            "/api/kb/pushkb",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert detail.status_code == 200
        body = detail.json()
        assert body["folder_push_url"].endswith("/api/push/pushkb")
        assert body["webhook_secret_len"] == len("sec-push-1")

    def test_push_requires_token(self, client: TestClient, store: SqliteAppStore, tmp_path: Path):
        reg = client.post(
            "/api/auth/register",
            json={"username": "pushuser", "password": "secret123"},
        ).json()
        uid = reg["user"]["id"]
        db = tmp_path / "tokb.sqlite"
        db.touch()
        store.create_knowledge_base(
            name="tokb",
            description="d",
            readme_md="",
            db_path=str(db),
            owner_id=uid,
            webhook_secret="tokensec",
        )
        resp = client.post("/api/push/tokb", files={"file": ("x.tar", b"not-tar", "application/x-tar")})
        assert resp.status_code == 401
