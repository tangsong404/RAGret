from __future__ import annotations

import pytest

from ragret.vision_config import require_vision_settings, vision_configured


def test_vision_configured_openai() -> None:
    assert vision_configured(
        provider="openai",
        base_url="https://x/v1",
        model="m",
        api_key="k",
    )


def test_require_vision_raises_when_incomplete() -> None:
    with pytest.raises(RuntimeError, match="Image ingest requires"):
        require_vision_settings(provider="openai", base_url="", model="m", api_key="k")
