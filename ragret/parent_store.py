from __future__ import annotations

from pathlib import Path


def parent_txt_path(parents_dir: Path, source_key: str) -> Path:
    key = source_key.replace("\\", "/").lstrip("/")
    return (parents_dir / f"{key}.txt").resolve()


def write_parent_text(parents_dir: Path, source_key: str, text: str) -> Path:
    path = parent_txt_path(parents_dir, source_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def read_parent_text(parents_dir: Path, source_key: str) -> str | None:
    path = parent_txt_path(parents_dir, source_key)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def delete_parent_text(parents_dir: Path, source_key: str) -> None:
    path = parent_txt_path(parents_dir, source_key)
    if path.is_file():
        path.unlink()
