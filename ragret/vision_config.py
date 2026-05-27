from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisionSettings:
    provider: str
    base_url: str
    model: str
    api_key: str


def vision_configured(*, provider: str, base_url: str, model: str, api_key: str) -> bool:
    if str(provider or "").strip().lower() == "anthropic":
        return bool(str(model or "").strip() and str(api_key or "").strip())
    return bool(str(base_url or "").strip() and str(model or "").strip() and str(api_key or "").strip())


def require_vision_settings(
    *,
    provider: str,
    base_url: str,
    model: str,
    api_key: str,
) -> VisionSettings:
    if not vision_configured(provider=provider, base_url=base_url, model=model, api_key=api_key):
        raise RuntimeError(
            "Image ingest requires RAGRET_VISION_PROVIDER, RAGRET_VISION_MODEL, and "
            "RAGRET_VISION_API_KEY (and RAGRET_VISION_BASE_URL for openai)."
        )
    prov = "anthropic" if str(provider or "").strip().lower() == "anthropic" else "openai"
    return VisionSettings(
        provider=prov,
        base_url=str(base_url or "").strip(),
        model=str(model or "").strip(),
        api_key=str(api_key or "").strip(),
    )
