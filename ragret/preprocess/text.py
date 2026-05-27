from __future__ import annotations

from pathlib import Path

_TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030")


def preprocess_text_file(path: Path) -> str:
    raw = path.read_bytes()
    if not raw.strip():
        return ""
    last_err: UnicodeDecodeError | None = None
    for enc in _TEXT_ENCODINGS:
        try:
            text = raw.decode(enc)
            return text.replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError as e:
            last_err = e
    if last_err is not None:
        raise last_err
    return ""
