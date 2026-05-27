from __future__ import annotations

import io
import zipfile
from pathlib import Path

from server.skill_pack import build_skill_zip, skill_md_path


def test_skill_md_path_points_under_skill_dir() -> None:
    root = Path(__file__).resolve().parents[1]
    assert skill_md_path(root) == (root / "skill" / "SKILL.md").resolve()
    assert skill_md_path(root).is_file()


def test_build_skill_zip_includes_skill_directory_files() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = build_skill_zip(root)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = set(zf.namelist())
    assert "ragret/SKILL.md" in names
    assert "ragret/scripts/ragret.ps1" in names
    assert "ragret/evals/evals.json" in names
    assert "ragret/SKILL.zh.md" in names
