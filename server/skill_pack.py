from __future__ import annotations

import io
import zipfile
from pathlib import Path

SKILL_DIR_NAME = "skill"
SKILL_ZIP_PREFIX = "ragret"


def skill_dir(repo_root: Path) -> Path:
    return (repo_root / SKILL_DIR_NAME).resolve()


def skill_md_path(repo_root: Path) -> Path:
    return skill_dir(repo_root) / "SKILL.md"


def build_skill_zip(repo_root: Path) -> bytes:
    base = skill_dir(repo_root)
    if not base.is_dir():
        raise FileNotFoundError(f"{SKILL_DIR_NAME}/ directory not found")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(base).as_posix()
            zf.write(path, arcname=f"{SKILL_ZIP_PREFIX}/{rel}")
    return buf.getvalue()
