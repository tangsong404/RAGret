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


def test_gitlab_webhook_invalid_secret(client: TestClient, store: SqliteAppStore, tmp_path: Path):
    user = store.create_user("whowner", "hash")
    store.create_pending_knowledge_base(
        name="whkb",
        description="d",
        readme_md="",
        db_path=str(tmp_path / "whkb.sqlite"),
        owner_id=user.id,
        source_type="webhook",
        webhook_provider="gitlab",
        webhook_secret="secret123",
        webhook_repo_url="https://gitlab.com/g/p.git",
        webhook_ref="main",
    )
    payload = {
        "object_kind": "push",
        "project": {"git_http_url": "https://gitlab.com/g/p.git"},
        "checkout_sha": "abc",
    }
    resp = client.post(
        "/api/webhooks/gitlab/whkb",
        json=payload,
        headers={"X-Gitlab-Token": "wrong"},
    )
    assert resp.status_code == 403
