from __future__ import annotations

from pathlib import Path

from ragret.cache import IndexCache, ModelCache
from ragret.registry import IndexRegistry
from ragret.searcher import search_db as core_search
from server.store.protocol import AppStore


def resolve_searchable_db(
    index_name: str,
    actor: dict,
    store: AppStore,
    registry: IndexRegistry | None = None,
) -> Path | None:
    """Returns db_path if the actor has read permission, else None."""
    kind = actor.get("kind")
    uid = actor.get("user_id")

    store_path = store.resolve_kb_db_path(index_name)
    if kind == "superuser":
        if store_path:
            return Path(store_path)
        if registry is not None:
            reg_path = registry.get_path(index_name)
            return reg_path
        return None
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


def format_search_text(
    query: str,
    results: list[dict],
    *,
    total_chunks: int | None = None,
) -> str:
    if not results:
        hint = (
            f"No passages above similarity threshold; try rephrasing or lower --threshold.\n"
        )
        if total_chunks is not None:
            hint += f"(Total chunks in index: {total_chunks})\n"
        return hint
    lines = [
        f"Query: {query}",
        f"Recalled {len(results)} passage(s) after rerank.",
        "",
        "--- Retrieved passages ---",
        "",
    ]
    for i, r in enumerate(results, 1):
        rs = r.get("relevance_score", "")
        vs = r.get("vector_score", "")
        src = r.get("source", "")
        lines.append(f"[{i}] rerank={rs}  vector={vs}")
        if src:
            lines.append(f"    source: {src}")
        lines.append(str(r.get("content", "")).strip())
        lines.append("")
    lines.append("--- Short summary ---")
    content = str(results[0].get("content", ""))
    lines.append(content.strip()[:800] + ("…" if len(content) > 800 else ""))
    return "\n".join(lines)
