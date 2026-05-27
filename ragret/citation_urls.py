from __future__ import annotations

from urllib.parse import quote


def parent_api_path(kb_name: str, source_key: str) -> str:
    rel = source_key.replace("\\", "/").lstrip("/")
    encoded_kb = quote(kb_name, safe="")
    encoded_rel = quote(f"{rel}.txt", safe="/")
    return f"/api/kb/{encoded_kb}/parents/{encoded_rel}"


def build_parent_url(
    *,
    kb_name: str,
    source_key: str,
    public_host: str | None = None,
) -> str:
    path = parent_api_path(kb_name, source_key)
    host = str(public_host or "").strip().rstrip("/")
    if host:
        if not host.startswith(("http://", "https://")):
            host = f"https://{host}"
        return f"{host}{path}"
    return path


def asset_api_path(kb_name: str, asset_rel_path: str) -> str:
    rel = asset_rel_path.replace("\\", "/").lstrip("/")
    encoded_kb = quote(kb_name, safe="")
    encoded_rel = quote(rel, safe="/")
    return f"/api/kb/{encoded_kb}/assets/{encoded_rel}"


def build_asset_url(
    *,
    kb_name: str,
    asset_rel_path: str,
    public_host: str | None = None,
) -> str:
    path = asset_api_path(kb_name, asset_rel_path)
    host = str(public_host or "").strip().rstrip("/")
    if host:
        if not host.startswith(("http://", "https://")):
            host = f"https://{host}"
        return f"{host}{path}"
    return path
