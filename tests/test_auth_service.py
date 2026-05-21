from __future__ import annotations

import pytest

from server.services import auth_service
from server.store.sqlite_store import SqliteAppStore


def test_register_and_login(store: SqliteAppStore) -> None:
    reg = auth_service.register_user(store, "alice", "secret12345")
    assert reg["user"]["username"] == "alice"
    assert reg["user"]["has_avatar"] is False
    assert reg["token"]

    bad = auth_service.login_user(store, "alice", "wrong")
    assert bad is None

    ok = auth_service.login_user(store, "alice", "secret12345")
    assert ok is not None
    assert ok["user"]["username"] == "alice"


def test_register_duplicate_username(store: SqliteAppStore) -> None:
    auth_service.register_user(store, "bob", "secret12345")
    with pytest.raises(ValueError, match="Username already taken"):
        auth_service.register_user(store, "bob", "otherpass99")


def test_logout_invalidates_session(store: SqliteAppStore) -> None:
    reg = auth_service.register_user(store, "carol", "secret12345")
    token = reg["token"]
    assert store.get_session_user_id(token) is not None
    auth_service.logout_user(store, token)
    assert store.get_session_user_id(token) is None


def test_change_password(store: SqliteAppStore) -> None:
    reg = auth_service.register_user(store, "dave", "secret12345")
    uid = reg["user"]["id"]
    assert auth_service.change_user_password(store, uid, "secret12345", "newpass999")
    assert auth_service.login_user(store, "dave", "newpass999") is not None
    assert auth_service.login_user(store, "dave", "secret12345") is None
