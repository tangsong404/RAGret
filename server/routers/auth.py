from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from server.deps import get_store, optional_actor, require_actor
from server.schemas import (
    AuthLoginRequest,
    AuthRegisterRequest,
    AuthResponse,
    MeResponse,
    UserOut,
)
from server.services import auth_service
from server.store.protocol import AppStore

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
def register(body: AuthRegisterRequest, store: AppStore = Depends(get_store)):
    try:
        result = auth_service.register_user(store, body.username, body.password)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    return AuthResponse(token=result["token"], user=UserOut(**result["user"]))


@router.post("/login", response_model=AuthResponse)
def login(body: AuthLoginRequest, store: AppStore = Depends(get_store)):
    result = auth_service.login_user(store, body.username, body.password)
    if result is None:
        raise HTTPException(401, detail="Invalid username or password")
    return AuthResponse(token=result["token"], user=UserOut(**result["user"]))


@router.post("/logout")
def logout(
    actor: dict = Depends(require_actor),
    store: AppStore = Depends(get_store),
):
    auth_service.logout_user(store, str(actor.get("token", "")))
    return {"ok": True}


@router.get("/me", response_model=MeResponse)
def me(
    actor: dict = Depends(optional_actor),
    store: AppStore = Depends(get_store),
):
    kind = actor.get("kind")
    if kind == "superuser":
        return MeResponse(user=None, superuser=True)
    uid = actor.get("user_id")
    if uid is None:
        raise HTTPException(401, detail="Not logged in")
    user = store.get_user_by_id(int(uid))
    if user is None:
        raise HTTPException(401, detail="Invalid session")
    has_avatar = store.user_has_avatar(int(uid))
    return MeResponse(user=UserOut(id=user.id, username=user.username, has_avatar=has_avatar))


@router.post("/password")
def change_password(
    body: dict,
    actor: dict = Depends(require_actor),
    store: AppStore = Depends(get_store),
):
    uid = actor.get("user_id")
    if uid is None:
        raise HTTPException(403, detail="User ID required")
    current = str(body.get("current_password", ""))
    new_pw = str(body.get("new_password", ""))
    if len(new_pw) < 8:
        raise HTTPException(400, detail="New password must be at least 8 characters")
    if not auth_service.change_user_password(store, int(uid), current, new_pw):
        raise HTTPException(401, detail="Current password is incorrect")
    return {"ok": True}
