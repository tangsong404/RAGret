from __future__ import annotations

import os
from pathlib import Path

import pytest

from server.config import Settings, load_settings
from server.auth_actor import effective_api_key


class TestEffectiveApiKey:
    def test_x_api_key_header(self) -> None:
        assert effective_api_key({"api_key": "sk-abc", "token": "session-tok"}) == "sk-abc"

    def test_bearer_sk_when_no_header(self) -> None:
        assert effective_api_key({"api_key": "", "token": "sk-secret"}) == "sk-secret"

    def test_session_token_not_used_as_api_key(self) -> None:
        assert effective_api_key({"api_key": "", "token": "opaque-session"}) == ""


class TestSettingsLegacyEnv:
    def test_reads_ragret_app_db_and_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAGRET_APP_DB", "/data/app.sqlite")
        monkeypatch.setenv("RAGRET_REGISTRY", "/data/registry")
        s = Settings()
        assert s.app_db_path == Path("/data/app.sqlite")
        assert s.registry_path == Path("/data/registry")

    def test_apply_legacy_environ(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db = tmp_path / "app.sqlite"
        reg = tmp_path / "registry"
        monkeypatch.setenv("RAGRET_APP_DB", str(db))
        monkeypatch.delenv("RAGRET_REGISTRY", raising=False)
        s = Settings()
        s.apply_legacy_environ()
        assert os.environ.get("RAGRET_APP_DB") == str(db)

    def test_load_settings_syncs_environ(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db = tmp_path / "load.sqlite"
        monkeypatch.setenv("RAGRET_APP_DB", str(db))
        load_settings()
        assert os.environ.get("RAGRET_APP_DB") == str(db)
