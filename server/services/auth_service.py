from __future__ import annotations

from server.passwords import hash_password
from server.store.protocol import AppStore

_SESSION_TTL = 30 * 24 * 3600  # 30 days


def register_user(store: AppStore, username: str, password: str) -> dict:
    user = store.create_user(username.strip(), hash_password(password))
    token = store.create_session(user.id, ttl_seconds=_SESSION_TTL)
    return {
        "token": token,
        "user": {"id": user.id, "username": user.username, "has_avatar": False},
    }


def login_user(store: AppStore, username: str, password: str) -> dict | None:
    user = store.verify_user_password(username.strip(), password)
    if user is None:
        return None
    token = store.create_session(user.id, ttl_seconds=_SESSION_TTL)
    has_avatar = store.user_has_avatar(user.id)
    return {
        "token": token,
        "user": {"id": user.id, "username": user.username, "has_avatar": has_avatar},
    }


def logout_user(store: AppStore, token: str) -> None:
    if token:
        store.delete_session(token)


def change_user_password(
    store: AppStore,
    user_id: int,
    current_password: str,
    new_password: str,
) -> bool:
    return store.change_password(user_id, current_password, hash_password(new_password))
