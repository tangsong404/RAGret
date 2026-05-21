from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from server.deps import get_store, require_actor
from server.services import user_service
from server.store.protocol import AppStore

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("/{user_id}/avatar")
def get_user_avatar(
    user_id: int,
    actor: dict = Depends(require_actor),
    store: AppStore = Depends(get_store),
):
    if actor.get("kind") == "anon":
        raise HTTPException(401, detail="Login required")
    try:
        mime, raw = user_service.load_avatar(user_id, store)
    except LookupError as e:
        raise HTTPException(404, detail=str(e))
    return Response(content=raw, media_type=mime)
