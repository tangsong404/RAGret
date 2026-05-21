from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ragret.cache import IndexCache, ModelCache
from ragret.registry import IndexRegistry
from server.deps import (
    get_index_cache,
    get_model_cache,
    get_registry,
    get_store,
    require_actor,
    require_user_id,
)
from server.services import quick_qa_service
from server.store.protocol import AppStore

router = APIRouter(prefix="/api", tags=["quick-qa"])


class QuickQaMessage(BaseModel):
    role: str
    content: str


class QuickQaRequest(BaseModel):
    question: str = Field(min_length=1)
    stream: bool = False
    lang: str = "zh"
    messages: list[QuickQaMessage] | None = None


@router.post("/quick-qa")
async def quick_qa(
    body: QuickQaRequest,
    request: Request,
    uid: int = Depends(require_user_id),
    actor: dict[str, Any] = Depends(require_actor),
    store: AppStore = Depends(get_store),
    registry: IndexRegistry = Depends(get_registry),
    model_cache: ModelCache = Depends(get_model_cache),
    index_cache: IndexCache = Depends(get_index_cache),
):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, detail="question is required")

    raw_messages = (
        [{"role": m.role, "content": m.content} for m in body.messages]
        if body.messages
        else None
    )
    messages = quick_qa_service._normalize_messages(raw_messages)

    if body.stream:
        return StreamingResponse(
            quick_qa_service.stream_quick_qa_events(
                question=q,
                store=store,
                registry=registry,
                model_cache=model_cache,
                index_cache=index_cache,
                uid=uid,
                actor=actor,
                messages=messages,
                lang=body.lang,
            ),
            media_type="application/x-ndjson; charset=utf-8",
            headers={"Cache-Control": "no-cache"},
        )

    try:
        payload = quick_qa_service.run_quick_qa_request(
            question=q,
            store=store,
            registry=registry,
            model_cache=model_cache,
            index_cache=index_cache,
            uid=uid,
            actor=actor,
            messages=messages,
            lang=body.lang,
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e)) from e
    return JSONResponse(payload)
