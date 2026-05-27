from __future__ import annotations

from pathlib import Path

from ragret.citation_urls import build_asset_url
from server.kb_content_paths import relocate_kb_runtime_on_rename, rewrite_kb_urls_in_parents
from server.runtime_paths import kb_assets_dir, kb_parents_dir, kb_sqlite_path


def test_relocate_kb_runtime_moves_dirs_and_rewrites_urls(tmp_path: Path) -> None:
    old, new = "oldkb", "newkb"
    parents = kb_parents_dir(tmp_path, old, create=True)
    assets = kb_assets_dir(tmp_path, old, create=True)
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "img.png").write_bytes(b"png")

    old_url = build_asset_url(kb_name=old, asset_rel_path="img.png", public_host="https://ragret.test")
    parent_txt = parents / "doc.md.txt"
    parent_txt.write_text(f"intro\n\n图片: {old_url}\n", encoding="utf-8")

    index_db = kb_sqlite_path(tmp_path, old)
    index_db.parent.mkdir(parents=True, exist_ok=True)
    index_db.write_bytes(b"sqlite")

    moved = relocate_kb_runtime_on_rename(tmp_path, old, new)
    assert moved == kb_sqlite_path(tmp_path, new)
    assert moved.is_file()
    assert not kb_parents_dir(tmp_path, old, create=False).exists()
    assert not kb_assets_dir(tmp_path, old, create=False).exists()
    assert (kb_assets_dir(tmp_path, new, create=False) / "img.png").is_file()

    text = (kb_parents_dir(tmp_path, new, create=False) / "doc.md.txt").read_text(encoding="utf-8")
    assert f"/api/kb/{new}/assets/" in text
    assert f"/api/kb/{old}/assets/" not in text


def test_rewrite_kb_urls_handles_percent_encoded_kb_name(tmp_path: Path) -> None:
    old, new = "my kb", "your-kb"
    parents = kb_parents_dir(tmp_path, new, create=True)
    parents.mkdir(parents=True, exist_ok=True)
    parent_txt = parents / "a.txt"
    parent_txt.write_text("see /api/kb/my%20kb/assets/x.png\n", encoding="utf-8")
    rewrite_kb_urls_in_parents(parents, old, new)
    assert "/api/kb/your-kb/assets/x.png" in parent_txt.read_text(encoding="utf-8")
