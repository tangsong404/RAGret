from __future__ import annotations

import json
import secrets
import sqlite3
from pathlib import Path
from typing import Any

from ragret.registry import safe_index_name
from server.build_queue import cleanup_upload_staging, wake_build_worker
from server.services import upload_service
from server.store.protocol import AppStore

_MAX_USER_UPLOAD_JOBS = 3


def push_token_from_headers(headers: dict[str, str]) -> str:
    token = str(headers.get("x-webhook-token") or headers.get("X-Webhook-Token") or "").strip()
    if token:
        return token
    return str(headers.get("x-gitlab-token") or headers.get("X-Gitlab-Token") or "").strip()


def _verify_push_token(kb_name: str, token: str, store: AppStore):
    key = safe_index_name(kb_name)
    rec = store.get_kb_record_any_state(key)
    if rec is None:
        raise LookupError("Unknown knowledge base")
    expected = str(rec.webhook_secret or "").strip()
    if not expected:
        raise PermissionError("Push token is not configured for this knowledge base")
    if not secrets.compare_digest(expected, token):
        raise PermissionError("Invalid push token")
    return rec


def enqueue_push_update(
    kb_name: str,
    token: str,
    file_obj: object,
    filename: str,
    store: AppStore,
    upload_base: Path,
) -> dict[str, Any]:
    rec = _verify_push_token(kb_name, token, store)
    key = safe_index_name(kb_name)
    if store.resolve_kb_db_path(key) is None:
        raise FileNotFoundError("Knowledge base is not ready yet")

    upload_id = upload_service.stage_archive_upload(file_obj, filename, upload_base)
    n_active = store.count_user_upload_tasks_active(int(rec.owner_id))
    if n_active >= _MAX_USER_UPLOAD_JOBS:
        cleanup_upload_staging(upload_base, upload_id)
        raise RuntimeError("Too many upload build jobs in queue or running (max 3).")

    job_id = secrets.token_hex(12)
    payload = {
        "description": str(rec.description or ""),
        "readme_md": str(rec.readme_md or ""),
        "is_public": bool(rec.is_public),
        "icon": str(rec.icon or "book"),
    }
    store.enqueue_build_job(
        job_id=job_id,
        user_id=int(rec.owner_id),
        task_kind="upload",
        op="update",
        kb_name=key,
        upload_id=upload_id,
        payload=payload,
    )
    wake_build_worker()
    return {"job_id": job_id}


def get_push_fingerprints(kb_name: str, token: str, store: AppStore) -> dict[str, Any]:
    rec = _verify_push_token(kb_name, token, store)
    key = safe_index_name(kb_name)
    db_path_s = store.resolve_kb_db_path(key)
    if not db_path_s:
        raise FileNotFoundError("Knowledge base is not ready yet")
    db_path = Path(db_path_s).resolve()
    if not db_path.is_file():
        raise FileNotFoundError("Knowledge base index file is missing")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='source_fingerprints'").fetchone()
        row_ts = conn.execute("SELECT value FROM meta WHERE key='indexed_at'").fetchone()
    finally:
        conn.close()

    raw = str(row[0] or "") if row else ""
    fp_map: dict[str, str] = {}
    if raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                fp_map = {str(k): str(v) for k, v in obj.items()}
        except json.JSONDecodeError:
            fp_map = {}
    indexed_at: int | None = None
    if row_ts and row_ts[0] is not None:
        try:
            indexed_at = int(str(row_ts[0]))
        except ValueError:
            indexed_at = None
    return {
        "kb_name": key,
        "indexed_at": indexed_at,
        "count": len(fp_map),
        "fingerprints": fp_map,
    }
