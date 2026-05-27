from __future__ import annotations

import base64
import logging
import re

import httpx

from ragret.vision_config import VisionSettings

logger = logging.getLogger(__name__)

_CAPTION_PROMPT = (
    "请用简洁的中文描述这张图片中的文字、图表和要点，便于知识检索。"
    "只输出描述正文，不要标题、不要 markdown、不要前后废话。"
)


def format_image_enrichment(caption: str, asset_url: str) -> str:
    text = caption.strip() or "（图片描述）"
    return f"{text}\n\n图片: {asset_url}"


def _normalize_openai_chat_url(base_url: str) -> str:
    b = str(base_url or "").strip().rstrip("/")
    if not b:
        raise ValueError("OpenAI-compatible vision requires RAGRET_VISION_BASE_URL")
    if b.endswith("/chat/completions"):
        return b
    if b.endswith("/v1"):
        return f"{b}/chat/completions"
    return f"{b}/v1/chat/completions"


def _normalize_anthropic_messages_url(base_url: str) -> str:
    b = str(base_url or "https://api.anthropic.com").strip().rstrip("/")
    if b.endswith("/messages"):
        return b
    if b.endswith("/v1"):
        return f"{b}/messages"
    return f"{b}/v1/messages"


def _clean_caption(raw: str) -> str:
    text = str(raw or "").strip()
    text = re.sub(r"^[\"'「『]+|[\"'」』]+$", "", text).strip()
    return text or "（图片描述）"


def caption_image_png(png_bytes: bytes, settings: VisionSettings, *, timeout_s: float = 60.0) -> str:
    if not png_bytes:
        return "（图片描述）"
    b64 = base64.standard_b64encode(png_bytes).decode("ascii")
    try:
        if settings.provider == "anthropic":
            raw = _caption_anthropic(b64, settings, timeout_s=timeout_s)
        else:
            raw = _caption_openai(b64, settings, timeout_s=timeout_s)
        return _clean_caption(raw)
    except Exception as e:
        logger.warning("Vision caption failed: %s", e)
        return "（图片描述）"


def _caption_openai(b64: str, settings: VisionSettings, *, timeout_s: float) -> str:
    url = _normalize_openai_chat_url(settings.base_url)
    payload = {
        "model": settings.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _CAPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 512,
    }
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(timeout_s, connect=15.0)
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Vision API returned no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [str(p.get("text", "")) for p in content if isinstance(p, dict)]
        return "".join(parts)
    return str(content or "")


def _caption_anthropic(b64: str, settings: VisionSettings, *, timeout_s: float) -> str:
    url = _normalize_anthropic_messages_url(settings.base_url)
    payload = {
        "model": settings.model,
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": _CAPTION_PROMPT},
                ],
            }
        ],
    }
    headers = {
        "x-api-key": settings.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(timeout_s, connect=15.0)
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    blocks = data.get("content") or []
    texts = [str(b.get("text", "")) for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
    return "".join(texts)
