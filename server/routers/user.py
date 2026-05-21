from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response

from server.config import Settings
from server.deps import get_settings, get_store, require_actor, require_user_id
from server.services import user_service
from server.store.protocol import AppStore

router = APIRouter(prefix="/api/user", tags=["user"])


@router.get("/subscriptions")
def list_subscriptions(
    user_id: int = Depends(require_user_id),
    store: AppStore = Depends(get_store),
):
    indexes = user_service.list_subscriptions(user_id, store)
    return {"ok": True, "indexes": indexes}


@router.get("/api-keys")
def list_api_keys(
    user_id: int = Depends(require_user_id),
    store: AppStore = Depends(get_store),
):
    return {"ok": True, "keys": user_service.list_api_keys(user_id, store)}


@router.post("/api-keys")
def create_api_key(
    body: dict,
    user_id: int = Depends(require_user_id),
    store: AppStore = Depends(get_store),
):
    try:
        rec = user_service.create_api_key(user_id, store, str(body.get("name") or ""))
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    return {"ok": True, "key": rec}


@router.delete("/api-keys/{key_id}")
def delete_api_key(
    key_id: str,
    user_id: int = Depends(require_user_id),
    store: AppStore = Depends(get_store),
):
    try:
        user_service.delete_api_key(user_id, store, int(key_id))
    except ValueError:
        raise HTTPException(400, detail="Invalid key id")
    except LookupError as e:
        raise HTTPException(404, detail=str(e))
    return {"ok": True}


@router.get("/gitlab-pat")
def get_gitlab_pat(
    user_id: int = Depends(require_user_id),
    store: AppStore = Depends(get_store),
):
    return {"ok": True, **user_service.get_gitlab_pat(user_id, store)}


@router.post("/gitlab-pat")
def set_gitlab_pat(
    body: dict,
    user_id: int = Depends(require_user_id),
    store: AppStore = Depends(get_store),
):
    user_service.set_gitlab_pat(user_id, store, str(body.get("pat") or ""))
    return {"ok": True}


@router.get("/github-pat")
def get_github_pat(
    user_id: int = Depends(require_user_id),
    store: AppStore = Depends(get_store),
):
    return {"ok": True, **user_service.get_github_pat(user_id, store)}


@router.post("/github-pat")
def set_github_pat(
    body: dict,
    user_id: int = Depends(require_user_id),
    store: AppStore = Depends(get_store),
):
    user_service.set_github_pat(user_id, store, str(body.get("pat") or ""))
    return {"ok": True}


@router.get("/webhook-secret/generate")
def generate_webhook_secret(
    _user_id: int = Depends(require_user_id),
):
    return {"ok": True, "secret": user_service.generate_webhook_secret()}


@router.get("/avatar")
def get_my_avatar(
    user_id: int = Depends(require_user_id),
    store: AppStore = Depends(get_store),
):
    try:
        mime, raw = user_service.load_avatar(user_id, store)
    except LookupError as e:
        raise HTTPException(404, detail=str(e))
    return Response(content=raw, media_type=mime)


@router.post("/avatar")
async def upload_my_avatar(
    file: UploadFile = File(...),
    user_id: int = Depends(require_user_id),
    store: AppStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
):
    raw = await file.read()
    try:
        user_service.save_avatar(
            user_id, store, file.content_type or "", raw, max_bytes=settings.avatar_max_bytes
        )
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except OSError as e:
        raise HTTPException(500, detail=str(e))
    return {"ok": True}


@router.delete("/avatar")
def delete_my_avatar(
    user_id: int = Depends(require_user_id),
    store: AppStore = Depends(get_store),
):
    user_service.clear_avatar(user_id, store)
    return {"ok": True}
