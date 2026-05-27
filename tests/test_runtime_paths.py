from __future__ import annotations

from pathlib import Path

from ragret.registry import safe_sqlite_basename
from server.runtime_paths import kb_assets_dir, kb_build_cache_dir, kb_parents_dir, runtime_data_dir


def test_kb_parents_dir_under_runtime_data(tmp_path: Path) -> None:
    kb = "my/kb?name"
    parents = kb_parents_dir(tmp_path, kb, create=True)
    expected = runtime_data_dir(tmp_path) / "kb_parents" / safe_sqlite_basename(kb)
    assert parents == expected.resolve()
    assert parents.is_dir()


def test_kb_assets_dir_under_runtime_data(tmp_path: Path) -> None:
    kb = "team-docs"
    assets = kb_assets_dir(tmp_path, kb, create=True)
    expected = runtime_data_dir(tmp_path) / "kb_assets" / safe_sqlite_basename(kb)
    assert assets == expected.resolve()
    assert assets.is_dir()


def test_kb_build_cache_dir_under_runtime_data(tmp_path: Path) -> None:
    kb = "team-docs"
    cache_dir = kb_build_cache_dir(tmp_path, kb, create=True)
    expected = runtime_data_dir(tmp_path) / "kb_build_cache" / safe_sqlite_basename(kb)
    assert cache_dir == expected.resolve()
    assert cache_dir.is_dir()
