from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

from ragret.registry import IndexRegistry, safe_index_name
from server.media_util import resolve_image_mime
from server.config import Settings
from server.runtime_paths import kb_sqlite_path
from server.serializers import serialize_kb
from server.store.protocol import AppStore, KBRecord
from server.webhook_urls import folder_push_url_for_kb


def is_kb_name_unavailable(
    name_raw: str,
    store: AppStore,
    registry: IndexRegistry,
    repo_root: Path,
) -> tuple[str, bool]:
    """Return sanitized name and whether it cannot be used for a new knowledge base."""
    key = safe_index_name(name_raw)
    if store.knowledge_base_name_taken(key):
        return key, True
    if registry.get_path(key) is not None:
        return key, True
    if kb_sqlite_path(repo_root, key).is_file():
        return key, True
    return key, False


def check_kb_name_for_create(
    name_raw: str,
    store: AppStore,
    registry: IndexRegistry,
    repo_root: Path,
) -> dict[str, Any]:
    key, taken = is_kb_name_unavailable(name_raw, store, registry, repo_root)
    return {"name": key, "available": not taken}


def list_subscribe_indexes(actor: dict[str, Any], store: AppStore) -> list[dict[str, Any]]:
    if actor.get("kind") != "api_key":
        raise PermissionError("Valid API key required")
    uid = actor.get("user_id")
    if uid is None:
        raise PermissionError("Valid API key required")
    rows = store.list_owned_and_subscribed_knowledge_bases_for_user(int(uid))
    return [serialize_kb(r) for r in rows]


def list_indexes(actor: dict[str, Any], store: AppStore) -> list[dict[str, Any]]:
    kind = actor.get("kind")
    uid = actor.get("user_id")
    if kind == "superuser":
        rows = store.list_all_knowledge_bases()
    elif kind == "api_key":
        raise PermissionError("Use /api/subscribe-indexes with API key")
    elif uid is not None:
        rows = store.list_knowledge_bases_for_user(int(uid))
    else:
        rows = []
    return [serialize_kb(r) for r in rows]


def get_kb_detail(
    name: str,
    actor: dict[str, Any],
    store: AppStore,
    registry: IndexRegistry,
    *,
    webhook_url: str | None = None,
    settings: Settings | None = None,
    port: int = 8765,
) -> dict[str, Any]:
    key = safe_index_name(name)
    kind = actor.get("kind")
    uid = actor.get("user_id")
    perm = None if kind == "superuser" else (store.permission_for(int(uid), key) if uid is not None else None)
    if kind == "api_key" and uid is not None:
        allowed = {
            str(r.name)
            for r in store.list_owned_and_subscribed_knowledge_bases_for_user(int(uid))
        }
        if key not in allowed:
            raise PermissionError("Forbidden")
    if kind != "superuser" and (perm is None or not perm.can_read):
        raise PermissionError("Forbidden")

    rec = store.get_knowledge_base(key)
    if rec is None and kind == "superuser":
        dbp = registry.get_path(key)
        if dbp is None:
            raise LookupError("Unknown knowledge base")
        return {
            "name": key,
            "description": registry.get_description(key) or "",
            "sqlite_exists": dbp.is_file(),
            "legacy_registry_only": True,
        }
    if rec is None:
        raise LookupError("Unknown knowledge base")

    body = serialize_kb(rec)
    body["readme_md"] = str(rec.readme_md or "")
    body["legacy_registry_only"] = False
    if webhook_url:
        body["webhook_url"] = webhook_url
    body["webhook_secret_masked"] = "*" * int(body.get("webhook_secret_len") or 0)
    if settings is not None:
        body["folder_push_url"] = folder_push_url_for_kb(key, settings, port=port)
    if kind != "superuser" and perm is not None:
        body["permission"] = {
            "can_read": perm.can_read,
            "can_write": perm.can_write,
            "can_delete": perm.can_delete,
            "is_owner": perm.is_owner,
        }
        if uid is not None:
            body["subscribed"] = store.kb_subscription_get(int(uid), key)
    return body


def set_subscription(
    name: str,
    subscribed: bool,
    user_id: int,
    store: AppStore,
) -> None:
    key = safe_index_name(name)
    perm = store.permission_for(user_id, key)
    if perm is None or not perm.can_read:
        raise PermissionError("Forbidden")
    if not store.kb_subscription_set(user_id, key, subscribed):
        raise LookupError("Unknown knowledge base")


def upsert_member(
    name: str,
    actor_user_id: int,
    member_username: str,
    *,
    can_write: bool,
    store: AppStore,
) -> None:
    safe_index_name(name)
    if not member_username:
        raise ValueError("Username is required")
    if store.get_user_by_username(member_username) is None:
        raise LookupError("User not found")
    if not store.upsert_member(
        name,
        actor_user_id=actor_user_id,
        member_username=member_username,
        can_read=True,
        can_write=can_write,
        can_delete=False,
    ):
        raise ValueError("User not found, is owner, or you are not owner")


def remove_member(
    name: str,
    actor_user_id: int,
    member_username: str,
    store: AppStore,
) -> None:
    safe_index_name(name)
    if not store.remove_member(name, actor_user_id=actor_user_id, member_username=member_username):
        raise LookupError("Member not found or not owner")


def patch_kb(
    name: str,
    data: dict[str, Any],
    actor: dict[str, Any],
    store: AppStore,
    registry: IndexRegistry,
) -> str:
    key = safe_index_name(name)
    kind = actor.get("kind")
    uid = actor.get("user_id")
    if not data:
        raise ValueError("No updates provided")

    if kind == "superuser":
        return _patch_kb_superuser(key, data, store, registry)
    if uid is None:
        raise PermissionError("Login required")
    perm = store.permission_for(int(uid), key)
    if perm is None or not perm.can_read:
        raise PermissionError("Forbidden")
    return _patch_kb_owner(key, data, int(uid), perm, store, registry)


def _patch_kb_superuser(
    key: str,
    data: dict[str, Any],
    store: AppStore,
    registry: IndexRegistry,
) -> str:
    active_key = key
    did = False
    if "name" in data:
        new_name = safe_index_name(str(data.get("name") or ""))
        if new_name != key:
            if not store.rename_knowledge_base(key, new_name):
                raise LookupError("Unknown knowledge base")
            old_path = registry.get_path(key)
            old_desc = registry.get_description(key) or ""
            if old_path is not None:
                registry.remove(key)
                registry.add(new_name, old_path, description=old_desc)
            active_key = new_name
            did = True
    if "description" in data:
        desc = str(data.get("description") or "").strip()
        if store.update_knowledge_base_description(active_key, desc):
            did = True
        cur = registry.get_path(active_key)
        if cur is not None:
            registry.add(active_key, cur, description=desc)
            did = True
    if "is_public" in data and store.update_knowledge_base_public(active_key, bool(data.get("is_public"))):
        did = True
    if "readme_md" in data and store.update_knowledge_base_readme(active_key, str(data.get("readme_md") or "").strip()):
        did = True
    if "webhook_secret" in data and store.update_knowledge_base_webhook_secret(
        active_key, str(data.get("webhook_secret") or "")
    ):
        did = True
    if bool(data.get("regenerate_webhook_secret")) and store.update_knowledge_base_webhook_secret(
        active_key, secrets.token_urlsafe(24)
    ):
        did = True
    if "repo_url" in data or "ref" in data:
        from server.build_queue import is_http_git_clone_url

        if "repo_url" in data:
            ru = str(data.get("repo_url") or "").strip()
            if ru and not is_http_git_clone_url(ru):
                raise ValueError(
                    "repo_url must start with http:// or https:// and include a path (clone URL, not a secret token)."
                )
        if store.update_knowledge_base_webhook_source(
            active_key,
            repo_url=str(data.get("repo_url") or "").strip() if "repo_url" in data else None,
            ref=str(data.get("ref") or "").strip() if "ref" in data else None,
        ):
            did = True
    if not did:
        raise LookupError("Unknown knowledge base")
    return active_key


def _patch_kb_owner(
    key: str,
    data: dict[str, Any],
    uid: int,
    perm: Any,
    store: AppStore,
    registry: IndexRegistry,
) -> str:
    if "is_public" in data and not perm.is_owner:
        raise PermissionError("Only the owner can change visibility")
    if ("webhook_secret" in data or bool(data.get("regenerate_webhook_secret"))) and not perm.is_owner:
        raise PermissionError("Only the owner can update webhook secret")
    if ("repo_url" in data or "ref" in data) and not perm.is_owner:
        raise PermissionError("Only the owner can update webhook repository settings")
    if "description" in data and not perm.can_write:
        raise PermissionError("Forbidden")
    if "name" in data and not perm.is_owner:
        raise PermissionError("Only the owner can rename knowledge base")

    active_key = key
    if "name" in data:
        new_name = safe_index_name(str(data.get("name") or ""))
        if new_name != key:
            if not store.rename_knowledge_base(key, new_name):
                raise LookupError("Unknown knowledge base")
            old_path = registry.get_path(key)
            old_desc = registry.get_description(key) or ""
            if old_path is not None:
                registry.remove(key)
                registry.add(new_name, old_path, description=old_desc)
            active_key = new_name
    if "description" in data:
        desc = str(data.get("description") or "").strip()
        if not store.update_knowledge_base_description(active_key, desc):
            raise LookupError("Unknown knowledge base")
        cur = registry.get_path(active_key)
        if cur is not None:
            registry.add(active_key, cur, description=desc)
    if "is_public" in data and not store.update_knowledge_base_public(active_key, bool(data.get("is_public"))):
        raise LookupError("Unknown knowledge base")
    if "readme_md" in data and not store.update_knowledge_base_readme(
        active_key, str(data.get("readme_md") or "").strip()
    ):
        raise LookupError("Unknown knowledge base")
    if "webhook_secret" in data and not store.update_knowledge_base_webhook_secret(
        active_key, str(data.get("webhook_secret") or "")
    ):
        raise LookupError("Unknown knowledge base")
    if bool(data.get("regenerate_webhook_secret")) and not store.update_knowledge_base_webhook_secret(
        active_key, secrets.token_urlsafe(24)
    ):
        raise LookupError("Unknown knowledge base")
    if "repo_url" in data or "ref" in data:
        from server.build_queue import is_http_git_clone_url

        if "repo_url" in data:
            ru = str(data.get("repo_url") or "").strip()
            if ru and not is_http_git_clone_url(ru):
                raise ValueError(
                    "repo_url must start with http:// or https:// and include a path (clone URL, not a secret token)."
                )
        if not store.update_knowledge_base_webhook_source(
            active_key,
            repo_url=str(data.get("repo_url") or "").strip() if "repo_url" in data else None,
            ref=str(data.get("ref") or "").strip() if "ref" in data else None,
        ):
            raise LookupError("Unknown knowledge base")
    return active_key


def load_kb_icon(name: str, actor: dict[str, Any], store: AppStore) -> tuple[str, bytes]:
    key = safe_index_name(name)
    kind = actor.get("kind")
    uid = actor.get("user_id")
    if kind != "superuser":
        if uid is None:
            raise PermissionError("Login required")
        perm = store.permission_for(int(uid), key)
        if perm is None or not perm.can_read:
            raise PermissionError("Forbidden")
    icon = store.load_kb_icon(key)
    if icon is None:
        raise LookupError("No icon")
    return icon


def save_kb_icon(
    name: str,
    actor: dict[str, Any],
    store: AppStore,
    mime: str,
    raw: bytes,
    *,
    max_bytes: int,
) -> None:
    key = safe_index_name(name)
    _check_icon_write(actor, store, key)
    if len(raw) > max_bytes:
        raise ValueError(f"Icon must be ≤ {max_bytes} bytes")
    resolved = resolve_image_mime(mime, raw)
    if resolved is None:
        raise ValueError("Use PNG, JPEG, GIF, or WebP")
    if not store.save_kb_icon(key, resolved, raw):
        raise LookupError("Unknown knowledge base")


def clear_kb_icon(name: str, actor: dict[str, Any], store: AppStore) -> None:
    key = safe_index_name(name)
    _check_icon_write(actor, store, key)
    if not store.clear_kb_icon(key):
        raise LookupError("No icon")


def get_webhook_secret(name: str, actor: dict[str, Any], store: AppStore) -> str:
    key = safe_index_name(name)
    kind = actor.get("kind")
    uid = actor.get("user_id")
    rec = store.get_kb_record_any_state(key)
    if rec is None:
        raise LookupError("Unknown knowledge base")
    if kind == "superuser":
        return str(rec.webhook_secret or "")
    if kind != "user" or uid is None:
        raise PermissionError("Login required")
    if int(rec.owner_id) != int(uid):
        raise PermissionError("Only owner can read webhook secret")
    return str(rec.webhook_secret or "")


def trigger_webhook_pull(name: str, owner_user_id: int, store: AppStore) -> dict[str, Any]:
    from pathlib import Path as _Path

    from server.build_queue import is_http_git_clone_url, wake_build_worker

    key = safe_index_name(name)
    rec = store.get_kb_record_any_state(key)
    if rec is None:
        raise LookupError("Unknown knowledge base")
    if int(rec.owner_id) != owner_user_id:
        raise PermissionError("Only the owner can trigger a manual pull")
    prov = str(rec.webhook_provider or "").strip().lower()
    if str(rec.source_type or "tar") != "webhook" or prov not in ("gitlab", "github"):
        raise ValueError("Not a GitLab/GitHub webhook knowledge base")
    repo_url = str(rec.webhook_repo_url or "").strip()
    ref = str(rec.webhook_ref or "").strip()
    if not repo_url:
        raise ValueError(
            "No repository URL stored yet; wait for a push webhook or set repo_url via PATCH"
        )
    if not is_http_git_clone_url(repo_url):
        raise ValueError(
            "Stored repo_url is not a valid http(s) address. Open manage and set repository URL to https://… (not the webhook secret)."
        )
    if not ref:
        raise ValueError("Branch (ref) is not set; configure it in knowledge base settings before pulling.")
    op = "update" if _Path(str(rec.db_path or "")).is_file() else "create"
    job_id = secrets.token_hex(12)
    payload = {
        "description": str(rec.description or ""),
        "readme_md": str(rec.readme_md or ""),
        "is_public": bool(rec.is_public),
        "icon": str(rec.icon or "book"),
        "repo_url": repo_url,
        "ref": ref,
        "checkout_sha": "",
    }
    store.enqueue_build_job(
        job_id=job_id,
        user_id=int(rec.owner_id),
        task_kind="webhook",
        op=op,
        kb_name=key,
        upload_id=secrets.token_hex(12),
        payload=payload,
    )
    wake_build_worker()
    return {"job_id": job_id, "op": op}


def _check_icon_write(actor: dict[str, Any], store: AppStore, key: str) -> None:
    kind = actor.get("kind")
    uid = actor.get("user_id")
    if kind == "superuser":
        return
    if uid is None:
        raise PermissionError("Login required")
    perm = store.permission_for(int(uid), key)
    if perm is None or not perm.can_write:
        raise PermissionError("Forbidden")
