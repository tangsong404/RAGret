from __future__ import annotations

import secrets
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


class TestJobsApi:
    def test_list_jobs_empty(self, client: TestClient):
        reg = client.post(
            "/api/auth/register",
            json={"username": "jobuser", "password": "secret123"},
        ).json()
        resp = client.get("/api/jobs", headers={"Authorization": f"Bearer {reg['token']}"})
        assert resp.status_code == 200
        assert resp.json()["jobs"] == []

    def test_build_missing_upload(self, client: TestClient):
        reg = client.post(
            "/api/auth/register",
            json={"username": "builder", "password": "secret123"},
        ).json()
        resp = client.post(
            "/api/indexes/build",
            headers={"Authorization": f"Bearer {reg['token']}"},
            json={
                "name": "newkb",
                "description": "desc",
                "upload_id": "a" * 24,
            },
        )
        assert resp.status_code == 404
        assert resp.json()["ok"] is False
        assert "error" in resp.json()

    def test_cancel_queued_job(self, client: TestClient, store: SqliteAppStore):
        reg = client.post(
            "/api/auth/register",
            json={"username": "canceler", "password": "secret123"},
        ).json()
        uid = reg["user"]["id"]
        store.create_pending_knowledge_base(
            name="cancelkb",
            description="d",
            readme_md="",
            db_path="/tmp/cancelkb.sqlite",
            owner_id=uid,
        )
        job_id = secrets.token_hex(12)
        store.enqueue_build_job(
            job_id=job_id,
            user_id=uid,
            task_kind="upload",
            op="create",
            kb_name="cancelkb",
            upload_id="d" * 24,
            payload={"description": "d"},
        )
        headers = {"Authorization": f"Bearer {reg['token']}"}
        resp = client.post(f"/api/jobs/{job_id}/cancel", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert store.get_knowledge_base("cancelkb") is None
