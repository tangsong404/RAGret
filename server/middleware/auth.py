from __future__ import annotations

from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from server.session_cookie import SESSION_COOKIE_NAME


class AuthMiddleware(BaseHTTPMiddleware):
    """Parse Authorization header and store actor in request.state.actor."""

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        auth = (request.headers.get("Authorization") or "").strip()
        api_key = (request.headers.get("X-API-Key") or "").strip()
        if not auth:
            cookie_token = (request.cookies.get(SESSION_COOKIE_NAME) or "").strip()
            if cookie_token:
                auth = f"Bearer {cookie_token}"
        request.state.actor = _resolve_actor(auth, api_key)
        return await call_next(request)


def _resolve_actor(auth: str, api_key: str) -> dict[str, Any]:
    """Return dict with keys: kind ('superuser'|'user'|'api_key'|'anon'), token, user_id."""
    bearer = ""
    if auth.lower().startswith("bearer "):
        bearer = auth[7:].strip()
    # superuser check via env token — done later in deps.py
    return {
        "kind": "anon",
        "token": bearer,
        "api_key": api_key if api_key else "",
    }
