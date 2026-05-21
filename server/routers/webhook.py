from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request

from server.deps import get_store
from server.services import webhook_service
from server.store.protocol import AppStore

router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])


@router.post("/gitlab/{name}")
async def gitlab_webhook(
    name: str,
    request: Request,
    store: AppStore = Depends(get_store),
):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(400, detail="Invalid JSON body")
    if not isinstance(data, dict):
        raise HTTPException(400, detail="JSON body must be an object")
    token = request.headers.get("X-Gitlab-Token") or ""
    try:
        result = webhook_service.handle_gitlab_push(name, data, token, store)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(403, detail=str(e))
    except LookupError as e:
        raise HTTPException(404, detail=str(e))
    if result.get("ignored"):
        return {"ok": True, **result}
    return {"ok": True, **result}


@router.post("/github/{name}")
async def github_webhook(
    name: str,
    request: Request,
    store: AppStore = Depends(get_store),
):
    ctype = request.headers.get("Content-Type") or ""
    if "application/json" not in ctype:
        raise HTTPException(415, detail="Content-Type must be application/json")
    raw = await request.body()
    try:
        data = json.loads(raw.decode("utf-8") if raw else "{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(400, detail="Invalid JSON body")
    if not isinstance(data, dict):
        raise HTTPException(400, detail="JSON body must be an object")
    try:
        result = webhook_service.handle_github_push(
            name,
            data,
            raw,
            request.headers.get("X-Hub-Signature-256") or "",
            request.headers.get("X-GitHub-Event") or "",
            store,
        )
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(403, detail=str(e))
    except LookupError as e:
        raise HTTPException(404, detail=str(e))
    if result.get("ignored"):
        return {"ok": True, **result}
    return {"ok": True, **result}
