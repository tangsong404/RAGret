from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Request, status

from ragret.cache import IndexCache, ModelCache
from server.auth_actor import effective_api_key
from server.config import Settings
from server.store.protocol import AppStore


async def get_settings(request: Request) -> Settings:
    return request.app.state.settings


async def get_store(request: Request) -> AppStore:
    return request.app.state.app_store


async def get_model_cache(request: Request) -> ModelCache:
    return request.app.state.model_cache


async def get_index_cache(request: Request) -> IndexCache:
    return request.app.state.index_cache


def _super_token(settings: Settings) -> str | None:
    t = settings.api_token
    return t.strip() if t else None


async def require_actor(
    request: Request,
    store: AppStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Returns actor dict. Raises 401 if anon."""
    actor = request.state.actor
    token = str(actor.get("token") or "")

    super_tok = _super_token(settings)
    if super_tok and token == super_tok:
        return {"kind": "superuser", "user_id": None, "token": token}

    uid = store.get_session_user_id(token)
    if uid is not None:
        return {"kind": "user", "user_id": int(uid), "token": token}

    api_key = effective_api_key(actor)
    uid_by_key = store.get_api_key_owner_user_id(api_key)
    if uid_by_key is not None:
        return {"kind": "api_key", "user_id": int(uid_by_key), "token": token, "api_key": api_key}

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login required")


async def optional_actor(
    request: Request,
    store: AppStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Returns actor dict. For anon, returns {"kind": "anon", "user_id": None}."""
    try:
        return await require_actor(request, store, settings)
    except HTTPException:
        return {"kind": "anon", "user_id": None}
