from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from ragret.registry import safe_index_name
from server.build_queue import is_http_git_clone_url, wake_build_worker
from server.store.protocol import AppStore, KBRecord


def github_signature256_valid(secret: str, payload: bytes, sig_header: str) -> bool:
    if not secret or not sig_header:
        return False
    sh = sig_header.strip()
    if not sh.startswith("sha256="):
        return False
    want = sh[7:].strip().lower()
    mac = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(mac, want)


def unwrap_github_logged_payload(data: dict[str, Any]) -> dict[str, Any]:
    summ = str(data.get("summary") or "")
    if not summ.startswith("payload="):
        return data
    try:
        inner = json.loads(unquote(summ[len("payload=") :]))
    except (json.JSONDecodeError, ValueError):
        return data
    return inner if isinstance(inner, dict) else data


def complete_webhook_push(
    safe_name: str,
    rec: KBRecord,
    repo_url: str,
    checkout_sha: str,
    store: AppStore,
) -> dict[str, Any]:
    store.update_knowledge_base_webhook_source(safe_name, repo_url=repo_url, ref=None)
    build_ref = str(rec.webhook_ref or "").strip()
    if not build_ref:
        raise ValueError(
            "Branch not configured: set ref (branch) on the knowledge base in the console, then retry."
        )
    op = "update" if Path(str(rec.db_path or "")).is_file() else "create"
    job_id = secrets.token_hex(12)
    payload = {
        "description": str(rec.description or ""),
        "readme_md": str(rec.readme_md or ""),
        "is_public": bool(rec.is_public),
        "icon": str(rec.icon or "book"),
        "repo_url": repo_url,
        "ref": build_ref,
        "checkout_sha": str(checkout_sha or ""),
    }
    store.enqueue_build_job(
        job_id=job_id,
        user_id=int(rec.owner_id),
        task_kind="webhook",
        op=op,
        kb_name=safe_name,
        upload_id=secrets.token_hex(12),
        payload=payload,
    )
    wake_build_worker()
    return {"job_id": job_id, "op": op}


def handle_gitlab_push(
    kb_name: str,
    data: dict[str, Any],
    token_header: str,
    store: AppStore,
) -> dict[str, Any]:
    if str(data.get("event_name") or data.get("object_kind") or "").lower() != "push":
        return {"ignored": True, "reason": "not_push_event"}
    safe_name = safe_index_name(kb_name)
    rec = store.get_kb_record_any_state(safe_name)
    if rec is None:
        raise LookupError("Unknown knowledge base")
    if str(rec.source_type or "tar") != "webhook" or str(rec.webhook_provider or "") != "gitlab":
        raise ValueError("Knowledge base is not configured for GitLab webhook")
    expected = str(rec.webhook_secret or "").strip()
    got = str(token_header or "").strip()
    if expected and not secrets.compare_digest(expected, got):
        raise PermissionError("Invalid webhook secret")
    project = data.get("project") if isinstance(data.get("project"), dict) else {}
    repository = data.get("repository") if isinstance(data.get("repository"), dict) else {}
    repo_url = (
        str(project.get("git_http_url") or "").strip()
        or str(repository.get("git_http_url") or "").strip()
        or str(project.get("http_url") or "").strip()
        or str(repository.get("url") or "").strip()
    )
    if not repo_url:
        raise ValueError("Missing repository URL in webhook payload")
    if not is_http_git_clone_url(repo_url):
        raise ValueError(
            "Webhook payload repository URL must be http(s)://… (got a non-URL value; check GitLab project fields)."
        )
    return complete_webhook_push(safe_name, rec, repo_url, str(data.get("checkout_sha") or ""), store)


def handle_github_push(
    kb_name: str,
    data: dict[str, Any],
    raw_body: bytes,
    sig_header: str,
    event_header: str,
    store: AppStore,
) -> dict[str, Any]:
    safe_name = safe_index_name(kb_name)
    rec = store.get_kb_record_any_state(safe_name)
    if rec is None:
        raise LookupError("Unknown knowledge base")
    if str(rec.source_type or "tar") != "webhook" or str(rec.webhook_provider or "") != "github":
        raise ValueError("Knowledge base is not configured for GitHub webhook")
    expected = str(rec.webhook_secret or "").strip()
    if expected and not github_signature256_valid(expected, raw_body, sig_header):
        raise PermissionError("Invalid webhook signature")
    data = unwrap_github_logged_payload(data)
    event = (event_header or "").strip().lower() or str(data.get("event") or "").strip().lower()
    repo = data.get("repository") if isinstance(data.get("repository"), dict) else {}
    clone_url = str(repo.get("clone_url") or "").strip()
    if not clone_url:
        raise ValueError("Missing repository clone_url in webhook payload")
    if not is_http_git_clone_url(clone_url):
        raise ValueError("repository.clone_url must be an https://… URL")
    if event and event != "push":
        return {"ignored": True, "reason": "not_push_event"}
    return complete_webhook_push(safe_name, rec, clone_url, str(data.get("after") or ""), store)
