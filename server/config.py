from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8765
    registry_path: Path | None = None
    app_db_path: Path | None = None
    session_ttl: int = 30 * 24 * 3600
    search_index_cache_max: int = 64
    search_rerank_cache_top: int = 256
    git_http_connect_timeout_s: float = 20.0
    git_http_read_timeout_s: float = 30.0
    git_clone_wall_timeout_s: float = 30.0
    api_token: str | None = None
    avatar_max_bytes: int = 2 * 1024 * 1024
    public_host: str | None = None

    model_config = {"env_prefix": "RAGRET_"}
