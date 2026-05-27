from __future__ import annotations

from pathlib import Path
from typing import Callable

ImageHandler = Callable[..., str]


def preprocess_pdf(path: Path, *, image_handler: ImageHandler | None = None) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        section = [f"--- page {i} ---", page_text]
        if image_handler is not None:
            images = getattr(page, "images", []) or []
            for j, img in enumerate(images, start=1):
                payload = getattr(img, "data", b"")
                if not payload:
                    continue
                name = str(getattr(img, "name", "") or f"page-{i}-image-{j}")
                block = image_handler(payload, name)
                if block.strip():
                    section.append("")
                    section.append(block.strip())
        parts.append("\n".join(section).strip())
    return "\n\n".join(parts).strip()
