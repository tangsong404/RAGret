from __future__ import annotations

from typing import Any


def effective_api_key(actor: dict[str, Any]) -> str:
    """Match httpd._api_key_raw: X-API-Key, else Bearer sk-... as API key."""
    api_key = str(actor.get("api_key") or "")
    if api_key:
        return api_key
    token = str(actor.get("token") or "")
    if token.startswith("sk-"):
        return token
    return ""
