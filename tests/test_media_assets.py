from __future__ import annotations

from pathlib import Path

import pytest

from ragret.media_assets import normalize_image_payload, save_asset_binary


def test_normalize_jpeg_to_png() -> None:
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")

    buf = __import__("io").BytesIO()
    Image.new("RGB", (2, 2), color="red").save(buf, format="JPEG")
    payload = buf.getvalue()

    data, mime = normalize_image_payload(payload, mime="image/jpeg", convert_to_png=True)
    assert mime == "image/png"
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_save_asset_binary_raises_without_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    from ragret import image_convert as ic

    monkeypatch.setattr(ic, "_pil_to_png_bytes", lambda *a, **k: None)
    monkeypatch.setattr(ic, "_imagemagick_to_png", lambda *a, **k: None)
    monkeypatch.setattr(ic, "_inkscape_to_png", lambda *a, **k: None)
    monkeypatch.setattr(ic, "_libreoffice_to_png", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="Could not convert"):
        save_asset_binary(
            assets_dir=Path("."),
            source_key="doc/embedded/x",
            payload=b"\x01\x00\x00\x00\xff",
            mime="image/x-emf",
            convert_to_png=True,
        )


def test_save_asset_binary_writes_png(tmp_path: Path) -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"x" * 8
    rel = save_asset_binary(
        assets_dir=tmp_path,
        source_key="doc/embedded/a",
        payload=png,
        convert_to_png=True,
    )
    assert rel.endswith(".png")
    assert (tmp_path / rel).is_file()
