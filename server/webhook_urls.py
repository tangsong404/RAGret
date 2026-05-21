from __future__ import annotations

import os
import socket

from server.config import Settings
from server.store.protocol import AppStore


def best_public_host(settings: Settings) -> str:
    env = str(settings.public_host or os.environ.get("RAGRET_PUBLIC_HOST") or "").strip()
    if env:
        return env
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = str(s.getsockname()[0] or "").strip()
            if ip:
                return ip
    except OSError:
        pass
    return "127.0.0.1"


def webhook_url_for_kb(
    kb_name: str,
    store: AppStore,
    *,
    proto: str = "http",
    host: str | None = None,
    port: int = 8765,
    settings: Settings | None = None,
) -> str:
    settings = settings or Settings()
    host = host or best_public_host(settings)
    if port not in (80, 443):
        host = f"{host}:{int(port)}"
    prov = "gitlab"
    if kb_name:
        rec = store.get_kb_record_any_state(kb_name)
        if rec is not None:
            p = str(rec.webhook_provider or "").strip().lower()
            if p in ("gitlab", "github"):
                prov = p
    return f"{proto}://{host}/api/webhooks/{prov}/{kb_name}"


def folder_push_url_for_kb(
    kb_name: str,
    settings: Settings | None = None,
    *,
    proto: str = "http",
    port: int = 8765,
) -> str:
    settings = settings or Settings()
    host = best_public_host(settings)
    if port not in (80, 443):
        host = f"{host}:{int(port)}"
    seg = kb_name if kb_name else "<kb-name>"
    return f"{proto}://{host}/api/push/{seg}"


def webhook_base_urls(
    settings: Settings | None = None,
    *,
    proto: str = "http",
    port: int = 8765,
) -> dict[str, str]:
    settings = settings or Settings()
    host = best_public_host(settings)
    if port not in (80, 443):
        host = f"{host}:{int(port)}"
    base = f"{proto}://{host}/api/webhooks"
    return {"gitlab": f"{base}/gitlab/", "github": f"{base}/github/"}
