from __future__ import annotations

import io
import logging
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_CONVERT_HELP = (
    "Install one of ImageMagick (magick), Inkscape, or LibreOffice (soffice) on PATH, "
    "or on Windows rely on Pillow with EMF/WMF support."
)


def detect_image_mime(payload: bytes) -> str | None:
    if len(payload) >= 8 and payload[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(payload) >= 3 and payload[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(payload) >= 6 and payload[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp"
    if len(payload) >= 2 and payload[:2] == b"BM":
        return "image/bmp"
    if len(payload) >= 4 and payload[:4] == b"\x01\x00\x00\x00":
        return "image/x-emf"
    if len(payload) >= 4 and payload[:4] in (b"\xd7\xcd\xc9\xa1", b"\x01\x00\x09\x00"):
        return "image/x-wmf"
    if len(payload) >= 2 and payload[:2] in (b"\xd7\xcd", b"\x01\x00"):
        return "image/x-wmf"
    return None


def _suffix_for_payload(
    payload: bytes,
    *,
    mime: str | None = None,
    filename_hint: str = "",
) -> str:
    declared = (mime or "").strip().lower()
    if not declared and filename_hint:
        guessed, _ = mimetypes.guess_type(filename_hint)
        declared = (guessed or "").strip().lower()

    if filename_hint:
        ext = Path(filename_hint).suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".emf", ".wmf"}:
            return ext

    mime_to_ext = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "image/x-emf": ".emf",
        "image/emf": ".emf",
        "image/x-wmf": ".wmf",
        "image/wmf": ".wmf",
        "application/x-msmetafile": ".wmf",
    }
    if declared in mime_to_ext:
        return mime_to_ext[declared]

    detected = detect_image_mime(payload)
    if detected and detected in mime_to_ext:
        return mime_to_ext[detected]
    return ".bin"


def _pil_to_png_bytes(payload: bytes, *, suffix: str, dpi: int = 144) -> bytes | None:
    try:
        from PIL import Image
    except ImportError:
        return None

    def _save(img: Image.Image) -> bytes:
        rgba = img.convert("RGBA")
        out = io.BytesIO()
        rgba.save(out, format="PNG")
        return out.getvalue()

    try:
        with Image.open(io.BytesIO(payload)) as img:
            return _save(img)
    except Exception:
        pass

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(payload)
            tmp_path = tf.name
        with Image.open(tmp_path) as img:
            if suffix in (".emf", ".wmf") and hasattr(img, "load"):
                try:
                    img.load(dpi=dpi)
                except Exception:
                    pass
            return _save(img)
    except Exception:
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _run_subprocess(cmd: list[str], *, timeout_s: float = 120.0) -> bool:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.debug("converter failed: %s %s", cmd[0], e)
        return False
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace").strip()
        logger.debug("converter exit %s: %s", proc.returncode, err[:500])
        return False
    return True


def _imagemagick_to_png(payload: bytes, *, suffix: str) -> bytes | None:
    magick = shutil.which("magick")
    convert = shutil.which("convert")
    if not magick and not convert:
        return None
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / f"input{suffix}"
        out = Path(td) / "output.png"
        inp.write_bytes(payload)
        if magick:
            ok = _run_subprocess([magick, str(inp), str(out)])
        else:
            ok = _run_subprocess([convert, str(inp), str(out)])
        if not ok or not out.is_file():
            return None
        return out.read_bytes()


def _inkscape_to_png(payload: bytes, *, suffix: str) -> bytes | None:
    inkscape = shutil.which("inkscape")
    if not inkscape:
        return None
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / f"input{suffix}"
        out = Path(td) / "output.png"
        inp.write_bytes(payload)
        ok = _run_subprocess(
            [
                inkscape,
                str(inp),
                "--export-type=png",
                f"--export-filename={out}",
            ]
        )
        if not ok or not out.is_file():
            return None
        return out.read_bytes()


def _libreoffice_to_png(payload: bytes, *, suffix: str) -> bytes | None:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / f"input{suffix}"
        inp.write_bytes(payload)
        ok = _run_subprocess(
            [
                soffice,
                "--headless",
                "--convert-to",
                "png",
                "--outdir",
                td,
                str(inp),
            ],
            timeout_s=180.0,
        )
        if not ok:
            return None
        produced = Path(td) / f"{inp.stem}.png"
        if not produced.is_file():
            pngs = sorted(Path(td).glob("*.png"))
            if not pngs:
                return None
            produced = pngs[0]
        return produced.read_bytes()


def convert_image_to_png(
    payload: bytes,
    *,
    mime: str | None = None,
    filename_hint: str = "",
    dpi: int = 144,
) -> bytes:
    """Convert arbitrary image bytes to PNG. Raises RuntimeError if no backend succeeds."""
    if not payload:
        raise ValueError("Empty image payload")

    detected = detect_image_mime(payload)
    if detected == "image/png" and payload[:8] == b"\x89PNG\r\n\x1a\n":
        return payload

    suffix = _suffix_for_payload(payload, mime=mime, filename_hint=filename_hint)
    backends: list[tuple[str, callable]] = [
        ("pillow", lambda: _pil_to_png_bytes(payload, suffix=suffix, dpi=dpi)),
        ("imagemagick", lambda: _imagemagick_to_png(payload, suffix=suffix)),
        ("inkscape", lambda: _inkscape_to_png(payload, suffix=suffix)),
        ("libreoffice", lambda: _libreoffice_to_png(payload, suffix=suffix)),
    ]
    errors: list[str] = []
    for name, fn in backends:
        try:
            out = fn()
        except Exception as e:
            errors.append(f"{name}: {e}")
            continue
        if out:
            logger.debug("Converted image via %s (%s)", name, suffix)
            return out
        errors.append(f"{name}: no output")

    platform = sys.platform
    raise RuntimeError(
        f"Could not convert embedded image to PNG (suffix={suffix}, platform={platform}). "
        f"{_CONVERT_HELP} Details: {'; '.join(errors)}"
    )
