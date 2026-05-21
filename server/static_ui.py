"""Serve built frontend from ``ragret/static`` (Vite outDir), with SPA fallback."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse


def static_dir_for_repo(repo_root: Path) -> Path:
    return (repo_root / "ragret" / "static").resolve()


def _safe_static_file(static_dir: Path, rel: str) -> Path | None:
    rel = rel.lstrip("/").replace("\\", "/")
    if not rel:
        candidate = static_dir / "index.html"
        return candidate if candidate.is_file() else None
    candidate = (static_dir / rel).resolve()
    try:
        candidate.relative_to(static_dir)
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        index = candidate / "index.html"
        return index if index.is_file() else None
    return None


def register_static_ui(app: FastAPI, repo_root: Path) -> None:
    """Register UI routes last so /api routers take precedence."""
    static_dir = static_dir_for_repo(repo_root)
    index_html = static_dir / "index.html"

    if not index_html.is_file():
        @app.get("/", include_in_schema=False)
        async def root_without_ui() -> dict[str, str]:
            return {
                "service": "ragret",
                "api": "/api/…",
                "auth": "/api/auth/login",
                "hint": "Build the UI: cd frontend && npm install && npm run build",
            }

        @app.get("/{path:path}", include_in_schema=False)
        async def spa_fallback_without_build(path: str) -> dict[str, str]:
            if path.startswith("api"):
                raise HTTPException(404, detail="Not Found")
            return {
                "ok": False,
                "error": "UI not built",
                "hint": "cd frontend && npm install && npm run build",
            }
        return

    @app.get("/", include_in_schema=False)
    async def root_ui() -> FileResponse:
        return FileResponse(index_html)

    @app.get("/{path:path}", include_in_schema=False)
    async def spa_ui(path: str) -> FileResponse:
        if path.startswith("api"):
            raise HTTPException(404, detail="Not Found")
        disk = _safe_static_file(static_dir, path)
        if disk is not None:
            return FileResponse(disk)
        return FileResponse(index_html)
