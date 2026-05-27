from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, Response

from ragret.cache import IndexCache, ModelCache
from ragret.registry import IndexRegistry, safe_index_name
from server.config import Settings
from server.deps import get_index_cache, get_model_cache, get_registry, get_settings, get_store, optional_actor
from server.schemas import SearchResponse, SearchResultOut
from server.services.search_service import (
    format_search_text,
    resolve_searchable_db,
    search_index,
)
from server.store.protocol import AppStore

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search/{name}", response_model=None)
def search(
    name: str,
    request: Request,
    q: str | None = Query(default=None, alias="query"),
    q_alt: str | None = Query(default=None, alias="q"),
    k: int = Query(default=10, ge=1, le=100),
    threshold: float = Query(default=0.3, ge=0.0, le=1.0, alias="score_threshold"),
    top_n: int = Query(default=5, ge=1, le=50, alias="rerank_top_n"),
    format: str = Query(default="json"),
    store: AppStore = Depends(get_store),
    registry: IndexRegistry = Depends(get_registry),
    model_cache: ModelCache = Depends(get_model_cache),
    index_cache: IndexCache = Depends(get_index_cache),
    settings: Settings = Depends(get_settings),
    actor: dict = Depends(optional_actor),
):
    try:
        safe_index_name(name)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    query = (q or q_alt or request.query_params.get("query") or request.query_params.get("q") or "").strip()
    if not query:
        raise HTTPException(400, detail="Missing query parameter: ?query= or ?q=")

    db = resolve_searchable_db(name, actor, store, registry)
    if db is None:
        raise HTTPException(404, detail=f"Unknown or inaccessible index: {name!r}")
    if not db.is_file():
        raise HTTPException(404, detail=f"SQLite missing for index {name!r}")
    try:
        results = search_index(
            db,
            query,
            model_cache,
            index_cache,
            k=k,
            score_threshold=threshold,
            rerank_top_n=top_n,
            kb_name=name,
            public_host=settings.public_host,
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))

    if format.lower() == "text":
        text = format_search_text(query, results)
        return PlainTextResponse(text)

    return SearchResponse(index=name, query=query, results=[SearchResultOut(**r) for r in results])
