from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from server.config import Settings
from server.main import create_app
from server.runtime_paths import kb_parents_dir
from server.store.pool import SqliteConnectionPool
from server.store.sqlite_store import SqliteAppStore


def test_parent_get_requires_read(tmp_path: Path) -> None:
    pool = SqliteConnectionPool(tmp_path / "app.sqlite", min_size=1, max_size=2)
    store = SqliteAppStore(pool)
    app = create_app(store=store, settings=Settings(), repo_root=tmp_path)
    client = TestClient(app)

    reg = client.post("/api/auth/register", json={"username": "powner", "password": "secret123"}).json()
    token = reg["token"]
    uid = reg["user"]["id"]
    store.create_knowledge_base(
        name="parkb",
        description="d",
        readme_md="",
        db_path=str(tmp_path / "parkb.sqlite"),
        owner_id=uid,
    )
    parents = kb_parents_dir(tmp_path, "parkb", create=True)
    (parents / "doc.md.txt").write_text("parent body", encoding="utf-8")

    resp = client.get(
        "/api/kb/parkb/parents/doc.md.txt",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.text == "parent body"

    other = client.post("/api/auth/register", json={"username": "pother", "password": "secret123"}).json()
    denied = client.get(
        "/api/kb/parkb/parents/doc.md.txt",
        headers={"Authorization": f"Bearer {other['token']}"},
    )
    assert denied.status_code == 403
    pool.close()


def test_parent_get_accepts_session_cookie(tmp_path: Path) -> None:
    pool = SqliteConnectionPool(tmp_path / "app.sqlite", min_size=1, max_size=2)
    store = SqliteAppStore(pool)
    app = create_app(store=store, settings=Settings(), repo_root=tmp_path)
    client = TestClient(app)

    reg = client.post("/api/auth/register", json={"username": "cowner", "password": "secret123"}).json()
    token = reg["token"]
    uid = reg["user"]["id"]
    store.create_knowledge_base(
        name="cookiekb",
        description="d",
        readme_md="",
        db_path=str(tmp_path / "cookiekb.sqlite"),
        owner_id=uid,
    )
    parents = kb_parents_dir(tmp_path, "cookiekb", create=True)
    (parents / "doc.md.txt").write_text("via cookie", encoding="utf-8")

    resp = client.get(
        "/api/kb/cookiekb/parents/doc.md.txt",
        cookies={"ragret_session": token},
    )
    assert resp.status_code == 200
    assert resp.text == "via cookie"
    pool.close()


def test_asset_get_requires_read(tmp_path: Path) -> None:
    pool = SqliteConnectionPool(tmp_path / "app.sqlite", min_size=1, max_size=2)
    store = SqliteAppStore(pool)
    app = create_app(store=store, settings=Settings(), repo_root=tmp_path)
    client = TestClient(app)

    reg = client.post("/api/auth/register", json={"username": "aowner", "password": "secret123"}).json()
    token = reg["token"]
    uid = reg["user"]["id"]
    store.create_knowledge_base(
        name="assetkb",
        description="d",
        readme_md="",
        db_path=str(tmp_path / "assetkb.sqlite"),
        owner_id=uid,
    )
    assets = tmp_path / "runtime" / "data" / "kb_assets" / "assetkb" / "img"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "x.png").write_bytes(b"\x89PNG\r\n\x1a\nrest")

    ok = client.get(
        "/api/kb/assetkb/assets/img/x.png",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert ok.status_code == 200
    assert ok.headers["content-type"].startswith("image/png")

    other = client.post("/api/auth/register", json={"username": "aother", "password": "secret123"}).json()
    denied = client.get(
        "/api/kb/assetkb/assets/img/x.png",
        headers={"Authorization": f"Bearer {other['token']}"},
    )
    assert denied.status_code == 403
    pool.close()
