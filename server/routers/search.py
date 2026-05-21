from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ragret.cache import IndexCache, ModelCache
from server.deps import get_index_cache, get_model_cache, get_store, optional_actor
from server.schemas import SearchResponse, SearchResultOut
from server.services.search_service import resolve_searchable_db, search_index
from server.store.protocol import AppStore

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search/{name}", response_model=SearchResponse)
def search(
    name: str,
    q: str = Query(alias="query"),
    k: int = Query(default=10, ge=1, le=100),
    threshold: float = Query(default=0.3, ge=0.0, le=1.0, alias="score_threshold"),
    top_n: int = Query(default=5, ge=1, le=50, alias="rerank_top_n"),
    store: AppStore = Depends(get_store),
    model_cache: ModelCache = Depends(get_model_cache),
    index_cache: IndexCache = Depends(get_index_cache),
    actor: dict = Depends(optional_actor),
):
    db = resolve_searchable_db(name, actor, store)
    if db is None:
        raise HTTPException(404, detail=f"Unknown or inaccessible index: {name!r}")
    if not db.is_file():
        raise HTTPException(404, detail=f"SQLite missing for index {name!r}")
    try:
        results = search_index(
            db,
            q,
            model_cache,
            index_cache,
            k=k,
            score_threshold=threshold,
            rerank_top_n=top_n,
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    return SearchResponse(index=name, query=q, results=[SearchResultOut(**r) for r in results])
