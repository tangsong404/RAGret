from __future__ import annotations

ALLOWED_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})


def sniff_image_mime(data: bytes) -> str | None:
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def resolve_image_mime(declared: str, raw: bytes) -> str | None:
    mime = (declared or "").strip().lower()
    if mime in ALLOWED_IMAGE_TYPES:
        return mime
    return sniff_image_mime(raw)
