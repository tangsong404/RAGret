from __future__ import annotations

import json
import secrets
import shutil
from pathlib import Path

from server.archive_util import is_tar_archive_filename


def stage_archive_upload(file_obj: object, filename: str, upload_base: Path) -> str:
    archive_name = Path(filename or "").name
    if not archive_name:
        raise ValueError("Missing archive filename")
    if not is_tar_archive_filename(archive_name):
        raise ValueError("Expected a tar archive (.tar, .tar.gz, .tgz, …)")

    upload_id = secrets.token_hex(12)
    upload_base.mkdir(parents=True, exist_ok=True)
    staging = (upload_base / "staging" / upload_id).resolve()
    try:
        staging.relative_to(upload_base.resolve())
    except ValueError:
        raise ValueError("Invalid staging path")
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "meta.json").write_text(
        json.dumps({"original_name": archive_name}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (staging / "blob").open("wb") as out:
        shutil.copyfileobj(file_obj, out)
    return upload_id
