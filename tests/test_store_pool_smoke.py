"""Smoke test: SqliteAppStore with SqliteConnectionPool."""
from __future__ import annotations

from server.store.pool import SqliteConnectionPool
from server.store.sqlite_store import SqliteAppStore


def test_create_user_with_pool(store: SqliteAppStore) -> None:
    u = store.create_user("pool_test_user", "pwhash")
    assert u.username == "pool_test_user"
    assert store.get_user_by_username("pool_test_user") is not None
