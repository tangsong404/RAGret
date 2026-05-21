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


class TestKbApi:
    def test_list_and_get_kb(self, client: TestClient, store: SqliteAppStore, tmp_path: Path):
        reg = client.post(
            "/api/auth/register",
            json={"username": "kbowner", "password": "secret123"},
        ).json()
        token = reg["token"]
        uid = reg["user"]["id"]
        db = tmp_path / "mykb.sqlite"
        db.touch()
        store.create_knowledge_base(
            name="mykb",
            description="desc",
            readme_md="",
            db_path=str(db),
            owner_id=uid,
        )
        headers = {"Authorization": f"Bearer {token}"}
        listed = client.get("/api/indexes", headers=headers)
        assert listed.status_code == 200
        names = [x["name"] for x in listed.json()["indexes"]]
        assert "mykb" in names

        detail = client.get("/api/kb/mykb", headers=headers)
        assert detail.status_code == 200
        assert detail.json()["name"] == "mykb"
        assert detail.json()["description"] == "desc"

    def test_patch_description(self, client: TestClient, store: SqliteAppStore, tmp_path: Path):
        reg = client.post(
            "/api/auth/register",
            json={"username": "patcher", "password": "secret123"},
        ).json()
        token = reg["token"]
        uid = reg["user"]["id"]
        db = tmp_path / "patchkb.sqlite"
        db.touch()
        store.create_knowledge_base(
            name="patchkb",
            description="old",
            readme_md="",
            db_path=str(db),
            owner_id=uid,
        )
        resp = client.patch(
            "/api/kb/patchkb",
            headers={"Authorization": f"Bearer {reg['token']}"},
            json={"description": "new desc"},
        )
        assert resp.status_code == 200
        got = store.get_knowledge_base("patchkb")
        assert got is not None
        assert got.description == "new desc"
