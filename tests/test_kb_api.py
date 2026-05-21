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

    def test_check_kb_name_duplicate(self, client: TestClient, store: SqliteAppStore, tmp_path: Path):
        reg = client.post(
            "/api/auth/register",
            json={"username": "dupuser", "password": "secret123"},
        ).json()
        token = reg["token"]
        uid = reg["user"]["id"]
        db = tmp_path / "dupkb.sqlite"
        db.touch()
        store.create_knowledge_base(
            name="dupkb",
            description="d",
            readme_md="",
            db_path=str(db),
            owner_id=uid,
        )
        headers = {"Authorization": f"Bearer {token}"}
        free = client.get("/api/kb/check-name?name=newkb", headers=headers)
        assert free.status_code == 200
        assert free.json()["available"] is True
        taken = client.get("/api/kb/check-name?name=dupkb", headers=headers)
        assert taken.status_code == 200
        assert taken.json()["available"] is False

    def test_build_rejects_duplicate_name(self, client: TestClient, store: SqliteAppStore, tmp_path: Path):
        reg = client.post(
            "/api/auth/register",
            json={"username": "builddup", "password": "secret123"},
        ).json()
        token = reg["token"]
        uid = reg["user"]["id"]
        db = tmp_path / "existkb.sqlite"
        db.touch()
        store.create_knowledge_base(
            name="existkb",
            description="d",
            readme_md="",
            db_path=str(db),
            owner_id=uid,
        )
        resp = client.post(
            "/api/indexes/build",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "name": "existkb",
                "description": "another",
                "upload_id": "a" * 24,
            },
        )
        assert resp.status_code == 409
        assert "already taken" in resp.json()["error"].lower()

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

    def test_members_and_subscribe(self, client: TestClient, store: SqliteAppStore, tmp_path: Path):
        owner = client.post(
            "/api/auth/register",
            json={"username": "owner1", "password": "secret123"},
        ).json()
        member = client.post(
            "/api/auth/register",
            json={"username": "member1", "password": "secret123"},
        ).json()
        db = tmp_path / "sharekb.sqlite"
        db.touch()
        store.create_knowledge_base(
            name="sharekb",
            description="shared",
            readme_md="",
            db_path=str(db),
            owner_id=owner["user"]["id"],
            is_public=True,
        )
        owner_h = {"Authorization": f"Bearer {owner['token']}"}
        resp = client.post(
            "/api/kb/sharekb/members",
            headers=owner_h,
            json={"username": "member1", "can_write": False},
        )
        assert resp.status_code == 200
        roster = client.get("/api/kb/sharekb/members", headers=owner_h).json()["members"]
        assert any(m.get("username") == "member1" for m in roster)

        member_h = {"Authorization": f"Bearer {member['token']}"}
        sub = client.post("/api/kb/sharekb/subscribe", headers=member_h)
        assert sub.status_code == 200
        assert sub.json()["subscribed"] is True

    def test_subscribe_indexes_api_key(self, client: TestClient, store: SqliteAppStore, tmp_path: Path):
        owner = client.post(
            "/api/auth/register",
            json={"username": "keyowner", "password": "secret123"},
        ).json()
        uid = owner["user"]["id"]
        db = tmp_path / "apikb.sqlite"
        db.touch()
        store.create_knowledge_base(
            name="apikb",
            description="api",
            readme_md="",
            db_path=str(db),
            owner_id=uid,
        )
        key_value = "sk-" + "a" * 32
        store.create_api_key_for_user(uid, name="default", key_value=key_value)
        resp = client.get("/api/subscribe-indexes", headers={"Authorization": f"Bearer {key_value}"})
        assert resp.status_code == 200
        names = [x["name"] for x in resp.json()["indexes"]]
        assert "apikb" in names

    def test_kb_icon_roundtrip_pending(self, client: TestClient, store: SqliteAppStore, tmp_path: Path):
        reg = client.post(
            "/api/auth/register",
            json={"username": "pendingicon", "password": "secret123"},
        ).json()
        uid = reg["user"]["id"]
        store.create_pending_knowledge_base(
            name="pendingicon",
            description="d",
            readme_md="",
            db_path=str(tmp_path / "pendingicon.sqlite"),
            owner_id=uid,
        )
        headers = {"Authorization": f"Bearer {reg['token']}"}
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
            b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        assert client.post(
            "/api/kb/pendingicon/icon",
            headers=headers,
            files={"file": ("icon.png", png, "image/png")},
        ).status_code == 200
        got = client.get("/api/kb/pendingicon/icon", headers=headers)
        assert got.status_code == 200
        assert got.content == png

    def test_kb_icon_roundtrip(self, client: TestClient, store: SqliteAppStore, tmp_path: Path):
        reg = client.post(
            "/api/auth/register",
            json={"username": "iconowner", "password": "secret123"},
        ).json()
        uid = reg["user"]["id"]
        db = tmp_path / "iconkb.sqlite"
        db.touch()
        store.create_knowledge_base(
            name="iconkb",
            description="d",
            readme_md="",
            db_path=str(db),
            owner_id=uid,
        )
        headers = {"Authorization": f"Bearer {reg['token']}"}
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
            b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        up = client.post(
            "/api/kb/iconkb/icon",
            headers=headers,
            files={"file": ("icon.png", png, "image/png")},
        )
        assert up.status_code == 200
        got = client.get("/api/kb/iconkb/icon", headers=headers)
        assert got.status_code == 200
        assert got.content == png
        assert client.delete("/api/kb/iconkb/icon", headers=headers).status_code == 200

    def test_webhook_secret_and_pull(self, client: TestClient, store: SqliteAppStore, tmp_path: Path):
        reg = client.post(
            "/api/auth/register",
            json={"username": "whpull", "password": "secret123"},
        ).json()
        uid = reg["user"]["id"]
        headers = {"Authorization": f"Bearer {reg['token']}"}
        store.create_pending_knowledge_base(
            name="whpullkb",
            description="d",
            readme_md="",
            db_path=str(tmp_path / "wh.sqlite"),
            owner_id=uid,
            source_type="webhook",
            webhook_provider="gitlab",
            webhook_secret="sec123",
            webhook_repo_url="https://gitlab.com/g/p.git",
            webhook_ref="main",
        )
        sec = client.get("/api/kb/whpullkb/webhook-secret", headers=headers)
        assert sec.status_code == 200
        assert sec.json()["secret"] == "sec123"
        pull = client.post("/api/kb/whpullkb/webhook-pull", headers=headers)
        assert pull.status_code == 200
        assert pull.json()["ok"] is True
        assert "job_id" in pull.json()
