from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from ragret.cache import IndexCache, ModelCache
from ragret.registry import IndexRegistry
from server.config import Settings, load_settings
from server.middleware.auth import AuthMiddleware
from server.routers import admin, auth, jobs, kb, search, upload, webhook
from server.runtime_paths import default_registry_path, runtime_upload_dir
from server.store.factory import create_app_store
from server.store.protocol import AppStore


def create_app(
    store: AppStore | None = None,
    model_cache: ModelCache | None = None,
    index_cache: IndexCache | None = None,
    settings: Settings | None = None,
    *,
    repo_root: Path | None = None,
    registry: IndexRegistry | None = None,
) -> FastAPI:
    app = FastAPI(title="RAGret")
    root = (repo_root or Path.cwd()).resolve()
    app.state.repo_root = root
    app.state.settings = settings or load_settings()
    app.state.app_store = store or create_app_store(root)
    app.state.model_cache = model_cache or ModelCache(device="cpu")
    app.state.index_cache = index_cache or IndexCache()
    app.state.upload_base = runtime_upload_dir(root)

    reg_path = app.state.settings.registry_path or default_registry_path(root)
    reg = registry or IndexRegistry(reg_path)
    reg.load()
    app.state.registry = reg

    app.add_middleware(AuthMiddleware)
    app.include_router(auth.router)
    app.include_router(search.router)
    app.include_router(kb.router)
    app.include_router(jobs.router)
    app.include_router(upload.router)
    app.include_router(webhook.router)
    app.include_router(admin.router)
    return app
