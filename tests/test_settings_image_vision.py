from __future__ import annotations

import pytest

from server.config import Settings, load_settings


class TestImageVisionSettings:
    def test_image_ingest_disabled_by_default(self) -> None:
        s = Settings()
        assert s.image_ingest_enabled is False

    def test_reads_image_ingest_and_vision_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAGRET_IMAGE_INGEST_ENABLED", "true")
        monkeypatch.setenv("RAGRET_VISION_PROVIDER", "openai")
        monkeypatch.setenv("RAGRET_VISION_BASE_URL", "https://vision.example/v1")
        monkeypatch.setenv("RAGRET_VISION_MODEL", "vision-model")
        monkeypatch.setenv("RAGRET_VISION_API_KEY", "vision-key")
        s = Settings()
        assert s.image_ingest_enabled is True
        assert s.vision_provider == "openai"
        assert s.vision_base_url == "https://vision.example/v1"
        assert s.vision_model == "vision-model"
        assert s.vision_api_key == "vision-key"
