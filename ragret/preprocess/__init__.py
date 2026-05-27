from __future__ import annotations

from pathlib import Path
from typing import Callable

from ragret.preprocess.pdf import preprocess_pdf
from ragret.preprocess.text import preprocess_text_file

ImageHandler = Callable[[bytes, str], str]

_SUPPORTED_SUFFIXES = frozenset(
    {
        ".pdf",
        ".txt",
        ".md",
        ".markdown",
        ".docx",
        ".xlsx",
    }
)


def preprocess_file(path: Path, *, image_handler: ImageHandler | None = None) -> str:
    suf = path.suffix.lower()
    if suf in (".txt", ".md", ".markdown"):
        return preprocess_text_file(path)
    if suf == ".pdf":
        return preprocess_pdf(path, image_handler=image_handler)
    if suf == ".docx":
        from ragret.preprocess.docx import preprocess_docx

        return preprocess_docx(path, image_handler=image_handler)
    if suf == ".xlsx":
        from ragret.preprocess.xlsx import preprocess_xlsx

        return preprocess_xlsx(path)
    raise ValueError(f"Unsupported file type for preprocess: {path.suffix}")


def is_preprocess_supported(path: Path) -> bool:
    return path.suffix.lower() in _SUPPORTED_SUFFIXES
