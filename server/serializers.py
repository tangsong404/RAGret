from __future__ import annotations

from pathlib import Path
from typing import Any

from server.store.protocol import KBRecord


def serialize_kb(rec: KBRecord) -> dict[str, Any]:
    p = rec.permission
    lc = max(0, min(4, int(rec.list_color_idx)))
    return {
        "name": rec.name,
        "description": rec.description,
        "sqlite_exists": Path(rec.db_path).is_file(),
        "is_public": bool(rec.is_public),
        "list_color_idx": lc,
        "icon": str(rec.icon or "book"),
        "source_type": str(rec.source_type or "tar"),
        "webhook_provider": str(rec.webhook_provider or ""),
        "webhook_secret_len": len(str(rec.webhook_secret or "")),
        "webhook_repo_url": str(rec.webhook_repo_url or ""),
        "webhook_ref": str(rec.webhook_ref or ""),
        "owner": {
            "id": rec.owner_id,
            "username": rec.owner_username,
            "has_avatar": rec.owner_has_avatar,
        },
        "permission": {
            "can_read": p.can_read,
            "can_write": p.can_write,
            "can_delete": p.can_delete,
            "is_owner": p.is_owner,
        },
    }


def job_public_view(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "phase": job.get("phase"),
        "percent": int(job.get("percent") or 0),
        "detail": str(job.get("detail") or ""),
        "error": job.get("error"),
        "result": job.get("result"),
        "op": job.get("op"),
        "kb_name": job.get("kb_name"),
        "task_kind": job.get("task_kind"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "cancel_requested": bool(job.get("cancel_requested")),
    }
