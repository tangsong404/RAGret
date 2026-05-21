from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# Legacy env names used by httpd.py / factory (not RAGRET_<FIELD> pydantic names).
_LEGACY_ENV_MAP: dict[str, str] = {
    "app_db_path": "RAGRET_APP_DB",
    "registry_path": "RAGRET_REGISTRY",
}


class _LegacyEnvSource(PydanticBaseSettingsSource):
    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for field_name, env_key in _LEGACY_ENV_MAP.items():
            raw = os.environ.get(env_key)
            if raw is not None and str(raw).strip():
                out[field_name] = raw
        return out

    def prepare_field_value(self, field_name: str, field: Any, value: Any, value_is_complex: bool) -> Any:
        return value


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

    model_config = SettingsConfigDict(env_prefix="RAGRET_")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            _LegacyEnvSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )

    def apply_legacy_environ(self) -> None:
        """Sync Settings to legacy env keys for factory/httpd during parallel migration."""
        if self.app_db_path is not None:
            os.environ["RAGRET_APP_DB"] = str(self.app_db_path)
        if self.registry_path is not None:
            os.environ["RAGRET_REGISTRY"] = str(self.registry_path)
        if self.api_token:
            os.environ["RAGRET_API_TOKEN"] = self.api_token
        os.environ["RAGRET_SESSION_TTL"] = str(self.session_ttl)
        os.environ["RAGRET_AVATAR_MAX_BYTES"] = str(self.avatar_max_bytes)
        if self.public_host:
            os.environ["RAGRET_PUBLIC_HOST"] = self.public_host


def load_settings() -> Settings:
    """Load settings and push legacy-compatible values into os.environ."""
    settings = Settings()
    settings.apply_legacy_environ()
    return settings
