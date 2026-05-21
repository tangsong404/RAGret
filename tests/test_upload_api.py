from __future__ import annotations

import io
import json
import tarfile
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


def _minimal_tar_bytes() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"hello"
        info = tarfile.TarInfo(name="readme.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_upload_tar_archive(client: TestClient, tmp_path: Path):
    reg = client.post(
        "/api/auth/register",
        json={"username": "uploader", "password": "secret123"},
    ).json()
    token = reg["token"]
    tar_bytes = _minimal_tar_bytes()
    resp = client.post(
        "/api/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("bundle.tar.gz", tar_bytes, "application/gzip")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert len(body["upload_id"]) == 24
    staging = tmp_path / "runtime" / "upload" / "staging" / body["upload_id"]
    assert (staging / "blob").is_file()
    meta = json.loads((staging / "meta.json").read_text(encoding="utf-8"))
    assert meta["original_name"] == "bundle.tar.gz"
