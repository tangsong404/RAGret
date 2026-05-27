from __future__ import annotations

import hashlib
from pathlib import Path

from ragret.image_convert import convert_image_to_png, detect_image_mime

_STORED_MIMES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})


def normalize_image_payload(
    payload: bytes,
    *,
    mime: str | None = None,
    filename_hint: str = "",
    convert_to_png: bool = True,
) -> tuple[bytes, str]:
    if not payload:
        raise ValueError("Empty image payload")
    if convert_to_png:
        png = convert_image_to_png(payload, mime=mime, filename_hint=filename_hint)
        return png, "image/png"

    detected = detect_image_mime(payload)
    declared = (mime or "").strip().lower()
    use_mime = detected or (declared if declared in _STORED_MIMES else None)
    if use_mime is None:
        raise RuntimeError(f"Unsupported image mime: {mime!r}")
    return payload, use_mime


def save_asset_binary(
    *,
    assets_dir: Path,
    source_key: str,
    payload: bytes,
    mime: str | None = None,
    filename_hint: str = "",
    convert_to_png: bool = True,
) -> str:
    normalized = normalize_image_payload(
        payload,
        mime=mime,
        filename_hint=filename_hint,
        convert_to_png=convert_to_png,
    )
    payload, mime = normalized
    ext = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(mime)
    if ext is None:
        raise RuntimeError(f"Unsupported stored mime: {mime}")
    digest = hashlib.sha256(payload).hexdigest()[:24]
    source_norm = source_key.replace("\\", "/").strip("/")
    out_rel = f"{source_norm}/{digest}{ext}"
    out = (assets_dir / out_rel).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    return out_rel.replace("\\", "/")


def delete_assets_for_source(*, assets_dir: Path, source_key: str) -> None:
    source_norm = source_key.replace("\\", "/").strip("/")
    target = (assets_dir / source_norm).resolve()
    try:
        target.relative_to(assets_dir.resolve())
    except ValueError:
        return
    if not target.is_dir():
        return
    for p in sorted(target.rglob("*"), reverse=True):
        if p.is_file():
            p.unlink(missing_ok=True)
        elif p.is_dir():
            p.rmdir()
    target.rmdir()
