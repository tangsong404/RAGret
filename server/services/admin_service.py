from __future__ import annotations

from pathlib import Path

from ragret.registry import IndexRegistry, safe_index_name
from server.kb_content_paths import cleanup_kb_content_dirs
from server.store.protocol import AppStore


def delete_index(
    name: str,
    actor: dict,
    store: AppStore,
    registry: IndexRegistry,
    repo_root: Path,
    *,
    delete_sqlite: bool = True,
) -> dict:
    safe_name = safe_index_name(name)
    kind = actor.get("kind")
    uid = actor.get("user_id")
    if kind == "superuser":
        perm_ok = True
    elif uid is not None:
        p = store.permission_for(int(uid), safe_name)
        perm_ok = p is not None and p.can_delete
    else:
        perm_ok = False
    if not perm_ok:
        raise PermissionError("Forbidden")

    sp = store.resolve_kb_db_path(safe_name)
    db: Path | None = Path(sp) if sp else registry.get_path(safe_name)

    in_store = store.delete_knowledge_base(safe_name)
    removed_reg = registry.remove(safe_name)
    if not in_store and not removed_reg:
        raise LookupError(f"Unknown index: {safe_name}")

    deleted_file = False
    if delete_sqlite and db is not None and db.is_file():
        try:
            db.unlink()
            deleted_file = True
        except OSError:
            deleted_file = False
    cleanup_kb_content_dirs(repo_root=repo_root, kb_name=safe_name)
    return {"name": safe_name, "deleted_sqlite": deleted_file}
