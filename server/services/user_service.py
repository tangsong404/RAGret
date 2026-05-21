from __future__ import annotations

import secrets
from typing import Any

from server.media_util import resolve_image_mime
from server.serializers import serialize_kb
from server.store.protocol import AppStore


def list_subscriptions(user_id: int, store: AppStore) -> list[dict[str, Any]]:
    rows = store.list_subscribed_knowledge_bases_for_user(user_id)
    return [serialize_kb(r) for r in rows]


def list_api_keys(user_id: int, store: AppStore) -> list[dict]:
    return store.list_api_keys_for_user(user_id)


def create_api_key(user_id: int, store: AppStore, name: str) -> dict:
    key_value = "sk-" + secrets.token_urlsafe(24).replace("-", "").replace("_", "")[:36]
    rec = store.create_api_key_for_user(user_id, name=name.strip(), key_value=key_value)
    if rec is None:
        raise ValueError("You can create at most 3 API keys")
    return rec


def delete_api_key(user_id: int, store: AppStore, key_id: int) -> None:
    if not store.delete_api_key_for_user(user_id, key_id):
        raise LookupError("API key not found")


def get_gitlab_pat(user_id: int, store: AppStore) -> dict[str, Any]:
    pat = store.get_user_gitlab_pat(user_id)
    return {"has_pat": bool(pat), "pat": pat}


def set_gitlab_pat(user_id: int, store: AppStore, pat: str) -> None:
    store.set_user_gitlab_pat(user_id, str(pat or ""))


def get_github_pat(user_id: int, store: AppStore) -> dict[str, Any]:
    pat = store.get_user_github_pat(user_id)
    return {"has_pat": bool(pat), "pat": pat}


def set_github_pat(user_id: int, store: AppStore, pat: str) -> None:
    store.set_user_github_pat(user_id, str(pat or ""))


def generate_webhook_secret() -> str:
    return secrets.token_urlsafe(24)


def load_avatar(user_id: int, store: AppStore) -> tuple[str, bytes]:
    av = store.load_avatar(user_id)
    if av is None:
        raise LookupError("No avatar")
    return av


def save_avatar(
    user_id: int,
    store: AppStore,
    mime: str,
    raw: bytes,
    *,
    max_bytes: int,
) -> None:
    if len(raw) > max_bytes:
        raise ValueError(f"Avatar must be ≤ {max_bytes} bytes")
    resolved = resolve_image_mime(mime, raw)
    if resolved is None:
        raise ValueError("Use PNG, JPEG, GIF, or WebP")
    store.save_avatar(user_id, resolved, raw)


def clear_avatar(user_id: int, store: AppStore) -> None:
    store.clear_avatar(user_id)
