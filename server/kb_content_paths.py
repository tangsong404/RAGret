from __future__ import annotations

from pathlib import Path


def resolve_under_base(base: Path, rel_path: str) -> Path:
    base = base.resolve()
    rel = rel_path.replace("\\", "/").lstrip("/")
    if not rel or rel in (".", ".."):
        raise ValueError("Invalid path")
    target = (base / rel).resolve()
    try:
        target.relative_to(base)
    except ValueError as e:
        raise ValueError("Invalid path") from e
    return target


def cleanup_kb_content_dirs(*, repo_root: Path, kb_name: str) -> None:
    from server.runtime_paths import kb_assets_dir, kb_parents_dir

    for base in (kb_parents_dir(repo_root, kb_name, create=False), kb_assets_dir(repo_root, kb_name, create=False)):
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                p.rmdir()
        base.rmdir()
