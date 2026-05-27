from __future__ import annotations

from fastapi import Response

SESSION_COOKIE_NAME = "ragret_session"


def _cookie_kwargs(*, max_age: int, secure: bool) -> dict[str, object]:
    return {
        "key": SESSION_COOKIE_NAME,
        "httponly": True,
        "samesite": "lax",
        "path": "/",
        "max_age": max_age,
        "secure": secure,
    }


def set_session_cookie(response: Response, token: str, *, max_age: int, secure: bool = False) -> None:
    if not token:
        return
    response.set_cookie(value=token, **_cookie_kwargs(max_age=max_age, secure=secure))


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")
