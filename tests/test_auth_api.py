from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.main import create_app
from server.store.pool import SqliteConnectionPool
from server.store.sqlite_store import SqliteAppStore


@pytest.fixture
def store(tmp_path: Path):
    db = tmp_path / "test.sqlite"
    pool = SqliteConnectionPool(db, min_size=1, max_size=2)
    s = SqliteAppStore(pool)
    yield s
    pool.close()


@pytest.fixture
def client(store: SqliteAppStore):
    app = create_app(store=store)
    with TestClient(app) as c:
        yield c


class TestRegister:
    def test_success(self, client: TestClient):
        resp = client.post("/api/auth/register", json={"username": "alice", "password": "secret123"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "token" in data
        assert data["user"]["username"] == "alice"

    def test_duplicate_username(self, client: TestClient):
        client.post("/api/auth/register", json={"username": "bob", "password": "secret123"})
        resp = client.post("/api/auth/register", json={"username": "bob", "password": "other456"})
        assert resp.status_code == 400

    def test_short_password(self, client: TestClient):
        resp = client.post("/api/auth/register", json={"username": "charlie", "password": "short"})
        assert resp.status_code == 422


class TestLogin:
    def test_success(self, client: TestClient):
        client.post("/api/auth/register", json={"username": "dave", "password": "secret123"})
        resp = client.post("/api/auth/login", json={"username": "dave", "password": "secret123"})
        assert resp.status_code == 200
        assert resp.json()["user"]["username"] == "dave"
        assert client.cookies.get("ragret_session")

    def test_sync_cookie(self, client: TestClient):
        reg = client.post(
            "/api/auth/register", json={"username": "syncuser", "password": "secret123"}
        ).json()
        client.cookies.clear()
        resp = client.post(
            "/api/auth/sync-cookie",
            headers={"Authorization": f"Bearer {reg['token']}"},
        )
        assert resp.status_code == 200
        assert client.cookies.get("ragret_session")

    def test_wrong_password(self, client: TestClient):
        client.post("/api/auth/register", json={"username": "eve", "password": "secret123"})
        resp = client.post("/api/auth/login", json={"username": "eve", "password": "wrongpass"})
        assert resp.status_code == 401


class TestMe:
    def test_authenticated(self, client: TestClient):
        reg = client.post("/api/auth/register", json={"username": "frank", "password": "secret123"}).json()
        token = reg["token"]
        resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["user"]["username"] == "frank"

    def test_unauthenticated(self, client: TestClient):
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401
