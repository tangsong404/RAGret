from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from ragret.cache import IndexCache, ModelCache
from server.config import Settings, load_settings
from server.middleware.auth import AuthMiddleware
from server.routers import auth, search
from server.store.factory import create_app_store
from server.store.protocol import AppStore


def create_app(
    store: AppStore | None = None,
    model_cache: ModelCache | None = None,
    index_cache: IndexCache | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    app = FastAPI(title="RAGret")
    app.state.settings = settings or load_settings()
    app.state.app_store = store or create_app_store(Path.cwd())
    app.state.model_cache = model_cache or ModelCache(device="cpu")
    app.state.index_cache = index_cache or IndexCache()

    app.add_middleware(AuthMiddleware)
    app.include_router(auth.router)
    app.include_router(search.router)
    return app
