from __future__ import annotations

import types

import pytest

from ragret import image_convert as ic


def test_convert_png_passthrough() -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"x" * 16
    out = ic.convert_image_to_png(png)
    assert out == png


def test_convert_jpeg_via_pillow() -> None:
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    buf = __import__("io").BytesIO()
    Image.new("RGB", (2, 2), color="blue").save(buf, format="JPEG")
    out = ic.convert_image_to_png(buf.getvalue(), mime="image/jpeg")
    assert out[:8] == b"\x89PNG\r\n\x1a\n"


def test_convert_emf_via_imagemagick_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_png = b"\x89PNG\r\n\x1a\n" + b"ok"

    def _fake_magick(payload: bytes, *, suffix: str) -> bytes | None:
        if suffix == ".emf":
            return fake_png
        return None

    monkeypatch.setattr(ic, "_pil_to_png_bytes", lambda *a, **k: None)
    monkeypatch.setattr(ic, "_imagemagick_to_png", _fake_magick)
    monkeypatch.setattr(ic, "_inkscape_to_png", lambda *a, **k: None)
    monkeypatch.setattr(ic, "_libreoffice_to_png", lambda *a, **k: None)

    emf = b"\x01\x00\x00\x00" + b"\x00" * 40
    out = ic.convert_image_to_png(emf, mime="image/x-emf", filename_hint="x.emf")
    assert out == fake_png


def test_convert_raises_when_no_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ic, "_pil_to_png_bytes", lambda *a, **k: None)
    monkeypatch.setattr(ic, "_imagemagick_to_png", lambda *a, **k: None)
    monkeypatch.setattr(ic, "_inkscape_to_png", lambda *a, **k: None)
    monkeypatch.setattr(ic, "_libreoffice_to_png", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="Could not convert"):
        ic.convert_image_to_png(b"\x01\x00\x00\x00\xff", mime="image/x-emf")
