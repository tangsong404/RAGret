from __future__ import annotations

from pathlib import Path
from urllib.parse import quote


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


def _kb_url_name_variants(kb_name: str) -> list[str]:
    raw = str(kb_name).strip().replace("\\", "/")
    variants = [raw]
    quoted = quote(raw, safe="")
    if quoted not in variants:
        variants.append(quoted)
    return variants


def rewrite_kb_urls_in_parents(parents_dir: Path, old_kb: str, new_kb: str) -> None:
    """Rewrite embedded /api/kb/{name}/... links inside parent .txt files after a KB rename."""
    if not parents_dir.is_dir():
        return
    old_variants = _kb_url_name_variants(old_kb)
    new_variants = _kb_url_name_variants(new_kb)
    new_frag = new_variants[0]
    for path in parents_dir.rglob("*.txt"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        updated = text
        for old_frag in old_variants:
            if old_frag == new_frag:
                continue
            updated = updated.replace(f"/api/kb/{old_frag}/", f"/api/kb/{new_frag}/")
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def _move_path_if_exists(src: Path, dst: Path, *, label: str) -> None:
    if not src.exists():
        return
    if dst.exists():
        raise ValueError(f"Cannot rename knowledge base: {label} already exists at {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)


def relocate_kb_runtime_on_rename(repo_root: Path, old_kb: str, new_kb: str) -> Path | None:
    """Move index sqlite, parent docs, and asset gallery to paths keyed by the new KB name."""
    from server.runtime_paths import kb_assets_dir, kb_parents_dir, kb_sqlite_path

    old_kb = str(old_kb).strip()
    new_kb = str(new_kb).strip()
    if not old_kb or not new_kb or old_kb == new_kb:
        return kb_sqlite_path(repo_root, new_kb) if kb_sqlite_path(repo_root, new_kb).is_file() else None

    old_sqlite = kb_sqlite_path(repo_root, old_kb)
    new_sqlite = kb_sqlite_path(repo_root, new_kb)
    if old_sqlite.is_file():
        _move_path_if_exists(old_sqlite, new_sqlite, label="index database")

    old_parents = kb_parents_dir(repo_root, old_kb, create=False)
    new_parents = kb_parents_dir(repo_root, new_kb, create=False)
    _move_path_if_exists(old_parents, new_parents, label="parent documents")

    old_assets = kb_assets_dir(repo_root, old_kb, create=False)
    new_assets = kb_assets_dir(repo_root, new_kb, create=False)
    _move_path_if_exists(old_assets, new_assets, label="asset gallery")

    rewrite_kb_urls_in_parents(new_parents, old_kb, new_kb)

    return new_sqlite if new_sqlite.is_file() else None


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
