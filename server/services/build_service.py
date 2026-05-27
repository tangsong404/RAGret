from __future__ import annotations

import json
import re
import secrets
import shutil
from pathlib import Path
from typing import Any

from ragret.registry import IndexRegistry, safe_index_name
from server.build_queue import cleanup_upload_staging, is_http_git_clone_url, wake_build_worker
from server.kb_content_paths import cleanup_kb_content_dirs
from server.runtime_paths import kb_sqlite_path
from server.config import Settings
from server.serializers import job_public_view
from server.store.protocol import AppStore
from server.services.kb_service import is_kb_name_unavailable
from server.webhook_urls import folder_push_url_for_kb

_UPLOAD_ID_RE = re.compile(r"^[a-f0-9]{24}$")
_MAX_USER_UPLOAD_JOBS = 3


def _cleanup_kb_runtime(repo_root: Path, kb_name: str) -> None:
    cleanup_kb_content_dirs(repo_root=repo_root, kb_name=kb_name)


def list_jobs(user_id: int, store: AppStore) -> list[dict[str, Any]]:
    return [job_public_view(j) for j in store.list_build_jobs_for_user(user_id)]


def get_job(job_id: str, actor: dict[str, Any], store: AppStore) -> dict[str, Any]:
    snap = store.get_build_job(job_id)
    if snap is None:
        raise LookupError("Unknown job")
    kind = actor.get("kind")
    uid = actor.get("user_id")
    if kind != "superuser" and (uid is None or int(snap["user_id"]) != int(uid)):
        raise PermissionError("Forbidden")
    return job_public_view(snap)


def cancel_job(
    job_id: str,
    owner_user_id: int,
    store: AppStore,
    registry: IndexRegistry,
    upload_base: Path,
    repo_root: Path,
) -> dict[str, Any]:
    err, drop_meta = store.request_cancel_build_job(job_id, owner_user_id)
    if err:
        raise ValueError(err)
    if drop_meta and drop_meta.get("dropped_queued"):
        op_q = str(drop_meta.get("op") or "")
        kb_n = str(drop_meta.get("kb_name") or "")
        u_id = str(drop_meta.get("upload_id") or "")
        if op_q == "create":
            store.delete_knowledge_base(kb_n)
            registry.remove(kb_n)
            _cleanup_kb_runtime(repo_root, kb_n)
            fd = kb_sqlite_path(repo_root, kb_n)
            try:
                if fd.is_file():
                    fd.unlink()
            except OSError:
                pass
        cleanup_upload_staging(upload_base, u_id)
    j = store.get_build_job(job_id)
    return job_public_view(j) if j else {}


def start_build_job(
    data: dict[str, Any],
    owner_user_id: int,
    store: AppStore,
    registry: IndexRegistry,
    upload_base: Path,
    repo_root: Path,
    *,
    settings: Settings | None = None,
    port: int = 8765,
) -> dict[str, Any]:
    name_raw = data.get("name") or data.get("index")
    desc_raw = str(data.get("description") or "").strip()
    readme_raw = str(data.get("readme_md") or "").strip()
    upload_id = data.get("upload_id")
    source_type = str(data.get("source_type") or "tar").strip().lower() or "tar"
    webhook_provider = str(data.get("webhook_provider") or "").strip().lower()
    webhook_secret = str(data.get("webhook_secret") or "").strip()
    webhook_repo_url = str(data.get("repo_url") or "").strip()
    webhook_ref = str(data.get("ref") or "").strip()

    if not name_raw or not desc_raw:
        raise ValueError("JSON must include non-empty name and description")
    if source_type not in ("tar", "webhook"):
        raise ValueError("source_type must be tar or webhook")
    if source_type == "tar" and not upload_id:
        raise ValueError("JSON must include non-empty upload_id for tar build")
    if source_type == "webhook" and webhook_provider not in ("", "gitlab", "github"):
        raise ValueError("webhook_provider must be gitlab or github")
    if not webhook_secret:
        webhook_secret = secrets.token_urlsafe(24)
    if source_type == "webhook" and not webhook_repo_url:
        raise ValueError("repo_url is required for webhook first build")
    if source_type == "webhook" and not is_http_git_clone_url(webhook_repo_url):
        raise ValueError(
            "repo_url must be an http(s) Git remote (e.g. https://gitlab.com/group/project.git), not a token or SSH URL."
        )
    if source_type == "webhook" and not webhook_ref:
        raise ValueError("ref is required for webhook builds (branch name, e.g. main or refs/heads/main)")

    index_name = safe_index_name(str(name_raw))

    def _reject_duplicate_name() -> None:
        if is_kb_name_unavailable(index_name, store, registry, repo_root)[1]:
            raise FileExistsError("Knowledge base name already taken")

    if source_type == "tar":
        n_active = store.count_user_upload_tasks_active(owner_user_id)
        if n_active >= _MAX_USER_UPLOAD_JOBS:
            raise RuntimeError(
                "Too many upload build jobs in queue or running (max 3). Finish, cancel one, or wait."
            )

    existing_ready = store.get_knowledge_base(index_name)
    owner_perm = (
        store.permission_for(owner_user_id, index_name) if existing_ready is not None else None
    )
    is_update = (
        existing_ready is not None
        and owner_perm is not None
        and owner_perm.is_owner
    )

    if source_type == "webhook":
        if is_update or is_kb_name_unavailable(index_name, store, registry, repo_root)[1]:
            raise FileExistsError("Knowledge base name already taken")
        wh_prov = webhook_provider or "gitlab"
        if wh_prov not in ("gitlab", "github"):
            wh_prov = "gitlab"
        final_sqlite = str(kb_sqlite_path(repo_root, index_name))
        try:
            store.create_pending_knowledge_base(
                name=index_name,
                description=desc_raw,
                readme_md=readme_raw,
                db_path=final_sqlite,
                owner_id=owner_user_id,
                is_public=bool(data.get("is_public", False)),
                icon=str(data.get("icon") or "book").strip() or "book",
                source_type="webhook",
                webhook_provider=wh_prov,
                webhook_secret=webhook_secret,
                webhook_repo_url=webhook_repo_url,
                webhook_ref=webhook_ref,
            )
        except ValueError as e:
            raise FileExistsError(str(e)) from e
        job_id = secrets.token_hex(12)
        payload = {
            "description": desc_raw,
            "readme_md": readme_raw,
            "is_public": bool(data.get("is_public", False)),
            "icon": str(data.get("icon") or "book").strip() or "book",
            "repo_url": webhook_repo_url,
            "ref": webhook_ref,
            "checkout_sha": "",
        }
        try:
            store.enqueue_build_job(
                job_id=job_id,
                user_id=owner_user_id,
                task_kind="webhook",
                op="create",
                kb_name=index_name,
                upload_id=secrets.token_hex(12),
                payload=payload,
            )
        except Exception as e:
            store.delete_knowledge_base(index_name)
            _cleanup_kb_runtime(repo_root, index_name)
            raise RuntimeError(str(e)) from e
        wake_build_worker()
        return {"job_id": job_id}

    if is_update:
        if webhook_secret:
            store.update_knowledge_base_webhook_secret(index_name, webhook_secret)
        op = "update"
    else:
        if existing_ready is not None:
            raise FileExistsError("Knowledge base name already taken")
        _reject_duplicate_name()
        op = "create"
        final_sqlite = str(kb_sqlite_path(repo_root, index_name))
        try:
            store.create_pending_knowledge_base(
                name=index_name,
                description=desc_raw,
                readme_md=readme_raw,
                db_path=final_sqlite,
                owner_id=owner_user_id,
                is_public=bool(data.get("is_public", False)),
                icon=str(data.get("icon") or "book").strip() or "book",
                source_type="tar",
                webhook_provider="",
                webhook_secret=webhook_secret,
            )
        except ValueError as e:
            raise FileExistsError(str(e)) from e

    upload_id = str(upload_id).strip()
    if not _UPLOAD_ID_RE.match(upload_id):
        if op == "create":
            store.delete_knowledge_base(index_name)
            _cleanup_kb_runtime(repo_root, index_name)
        raise ValueError("Invalid upload_id")
    staging = (upload_base / "staging" / upload_id).resolve()
    try:
        staging.relative_to(upload_base.resolve())
    except ValueError:
        if op == "create":
            store.delete_knowledge_base(index_name)
            _cleanup_kb_runtime(repo_root, index_name)
        raise ValueError("Invalid upload_id")
    if not staging.is_dir():
        if op == "create":
            store.delete_knowledge_base(index_name)
            _cleanup_kb_runtime(repo_root, index_name)
        raise LookupError("Upload not found; upload the archive first")

    is_public_flag = bool(data.get("is_public", False))
    icon_key = str(data.get("icon") or "book").strip() or "book"
    meta_path = staging / "meta.json"
    try:
        meta0 = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta0, dict):
            meta0 = {}
    except (OSError, json.JSONDecodeError):
        meta0 = {}
    meta0["is_public"] = is_public_flag
    meta0["icon"] = icon_key
    meta0["readme_md"] = readme_raw
    meta_path.write_text(json.dumps(meta0, ensure_ascii=False) + "\n", encoding="utf-8")

    job_id = secrets.token_hex(12)
    payload = {
        "description": desc_raw,
        "readme_md": readme_raw,
        "is_public": is_public_flag,
        "icon": icon_key,
    }
    try:
        store.enqueue_build_job(
            job_id=job_id,
            user_id=owner_user_id,
            task_kind="upload",
            op=op,
            kb_name=index_name,
            upload_id=upload_id,
            payload=payload,
        )
    except Exception as e:
        if op == "create":
            store.delete_knowledge_base(index_name)
            _cleanup_kb_runtime(repo_root, index_name)
        raise RuntimeError(str(e)) from e
    wake_build_worker()
    push_url = folder_push_url_for_kb(index_name, settings, port=port) if settings else None
    return {"job_id": job_id, "folder_push_url": push_url}
