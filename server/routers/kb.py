from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ragret.registry import IndexRegistry
from server.config import Settings
from server.deps import get_registry, get_settings, get_store, optional_actor, require_actor, require_user_id
from server.services import kb_service
from server.store.protocol import AppStore
from server.webhook_urls import webhook_url_for_kb

router = APIRouter(prefix="/api", tags=["kb"])


@router.get("/indexes")
def list_indexes(
    actor: dict = Depends(require_actor),
    store: AppStore = Depends(get_store),
):
    try:
        indexes = kb_service.list_indexes(actor, store)
    except PermissionError as e:
        raise HTTPException(403, detail=str(e))
    return {"ok": True, "indexes": indexes}


@router.get("/kb/{name}")
def get_kb(
    name: str,
    actor: dict = Depends(require_actor),
    store: AppStore = Depends(get_store),
    registry: IndexRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
):
    wh_url = webhook_url_for_kb(
        name,
        store,
        settings=settings,
        port=settings.port,
    )
    try:
        body = kb_service.get_kb_detail(name, actor, store, registry, webhook_url=wh_url)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(403, detail=str(e))
    except LookupError as e:
        raise HTTPException(404, detail=str(e))
    return {"ok": True, **body}


@router.patch("/kb/{name}")
def patch_kb(
    name: str,
    body: dict,
    actor: dict = Depends(require_actor),
    store: AppStore = Depends(get_store),
    registry: IndexRegistry = Depends(get_registry),
):
    try:
        active = kb_service.patch_kb(name, body, actor, store, registry)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(403, detail=str(e))
    except LookupError as e:
        raise HTTPException(404, detail=str(e))
    return {"ok": True, "name": active}


@router.get("/kb/{name}/members")
def list_members(
    name: str,
    actor: dict = Depends(require_actor),
    store: AppStore = Depends(get_store),
):
    try:
        roster = store.list_members_roster(name)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    if roster is None:
        raise HTTPException(404, detail="Unknown knowledge base")
    kind = actor.get("kind")
    uid = actor.get("user_id")
    if kind != "superuser":
        if uid is None:
            raise HTTPException(403, detail="Login required")
        perm = store.permission_for(int(uid), name)
        if perm is None or not perm.can_read:
            raise HTTPException(403, detail="Forbidden")
    return {"ok": True, "members": roster}


@router.post("/kb/{name}/members")
def add_member(
    name: str,
    body: dict,
    user_id: int = Depends(require_user_id),
    store: AppStore = Depends(get_store),
):
    try:
        kb_service.upsert_member(
            name,
            user_id,
            str(body.get("username") or "").strip(),
            can_write=bool(body.get("can_write", False)),
            store=store,
        )
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except LookupError as e:
        raise HTTPException(404, detail=str(e))
    return {"ok": True}


@router.delete("/kb/{name}/members/{username}")
def delete_member(
    name: str,
    username: str,
    user_id: int = Depends(require_user_id),
    store: AppStore = Depends(get_store),
):
    try:
        kb_service.remove_member(name, user_id, username, store)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except LookupError as e:
        raise HTTPException(404, detail=str(e))
    return {"ok": True}


@router.post("/kb/{name}/subscribe")
def subscribe(
    name: str,
    user_id: int = Depends(require_user_id),
    store: AppStore = Depends(get_store),
):
    try:
        kb_service.set_subscription(name, True, user_id, store)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(403, detail=str(e))
    except LookupError as e:
        raise HTTPException(404, detail=str(e))
    return {"ok": True, "subscribed": True}


@router.delete("/kb/{name}/subscribe")
def unsubscribe(
    name: str,
    user_id: int = Depends(require_user_id),
    store: AppStore = Depends(get_store),
):
    try:
        kb_service.set_subscription(name, False, user_id, store)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(403, detail=str(e))
    except LookupError as e:
        raise HTTPException(404, detail=str(e))
    return {"ok": True, "subscribed": False}
