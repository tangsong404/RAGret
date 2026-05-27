from __future__ import annotations

import json

import httpx
import pytest

from ragret.vision_caption import caption_image_png, format_image_enrichment
from ragret.vision_config import VisionSettings


def test_format_image_enrichment() -> None:
    out = format_image_enrichment("系统架构图", "/api/kb/x/assets/a.png")
    assert out == "系统架构图\n\n图片: /api/kb/x/assets/a.png"


def test_caption_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"x" * 8

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content.decode())
        assert body["model"] == "gpt-4o"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "  流程示意图  "}}]},
        )

    transport = httpx.MockTransport(handler)
    _real_client = httpx.Client

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return _real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _client)
    settings = VisionSettings(
        provider="openai",
        base_url="https://api.example.com/v1",
        model="gpt-4o",
        api_key="sk-test",
    )
    assert caption_image_png(png, settings) == "流程示意图"
