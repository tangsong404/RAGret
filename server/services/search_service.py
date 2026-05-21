from __future__ import annotations

from pathlib import Path

from ragret.cache import IndexCache, ModelCache
from ragret.searcher import search_db as core_search
from server.store.protocol import AppStore


def resolve_searchable_db(
    index_name: str,
    actor: dict,
    store: AppStore,
) -> Path | None:
    """Returns db_path if the actor has read permission, else None."""
    kind = actor.get("kind")
    uid = actor.get("user_id")

    store_path = store.resolve_kb_db_path(index_name)
    if kind == "superuser":
        return Path(store_path) if store_path else None
    if uid is None:
        return None
    if kind == "api_key":
        allowed = {
            str(r.name)
            for r in store.list_owned_and_subscribed_knowledge_bases_for_user(int(uid))
        }
        if index_name not in allowed:
            return None
    perm = store.permission_for(int(uid), index_name)
    if perm is None or not perm.can_read:
        return None
    return Path(store_path) if store_path else None


def search_index(
    db_path: Path,
    query: str,
    model_cache: ModelCache,
    index_cache: IndexCache,
    k: int = 10,
    score_threshold: float = 0.3,
    rerank_top_n: int = 5,
) -> list[dict]:
    return core_search(
        db_path,
        query,
        model_cache=model_cache,
        index_cache=index_cache,
        k=k,
        score_threshold=score_threshold,
        rerank_top_n=rerank_top_n,
    )
