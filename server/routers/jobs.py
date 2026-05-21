from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from ragret.registry import IndexRegistry
from server.config import Settings
from server.deps import (
    get_registry,
    get_repo_root,
    get_settings,
    get_store,
    get_upload_base,
    require_actor,
    require_user_id,
)
from server.schemas import BuildJobRequest, BuildJobResponse
from server.services import build_service
from server.store.protocol import AppStore
from server.webhook_urls import webhook_url_for_kb

router = APIRouter(prefix="/api", tags=["jobs"])


@router.get("/jobs")
def list_jobs(
    user_id: int = Depends(require_user_id),
    store: AppStore = Depends(get_store),
):
    return {"ok": True, "jobs": build_service.list_jobs(user_id, store)}


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    actor: dict = Depends(require_actor),
    store: AppStore = Depends(get_store),
):
    try:
        job = build_service.get_job(job_id, actor, store)
    except LookupError as e:
        raise HTTPException(404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(403, detail=str(e))
    return {"ok": True, **job}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(
    job_id: str,
    user_id: int = Depends(require_user_id),
    store: AppStore = Depends(get_store),
    registry: IndexRegistry = Depends(get_registry),
    upload_base: Path = Depends(get_upload_base),
    repo_root: Path = Depends(get_repo_root),
):
    try:
        job = build_service.cancel_job(job_id, user_id, store, registry, upload_base, repo_root)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    return {"ok": True, "job": job}


@router.post("/indexes/build", response_model=BuildJobResponse, status_code=202)
def start_build(
    body: BuildJobRequest,
    user_id: int = Depends(require_user_id),
    store: AppStore = Depends(get_store),
    registry: IndexRegistry = Depends(get_registry),
    upload_base: Path = Depends(get_upload_base),
    repo_root: Path = Depends(get_repo_root),
    settings: Settings = Depends(get_settings),
):
    data = body.model_dump()
    try:
        result = build_service.start_build_job(
            data,
            user_id,
            store,
            registry,
            upload_base,
            repo_root,
            settings=settings,
            port=settings.port,
        )
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(403, detail=str(e))
    except FileExistsError as e:
        raise HTTPException(409, detail=str(e))
    except LookupError as e:
        raise HTTPException(404, detail=str(e))
    except RuntimeError as e:
        code = 429 if "Too many upload" in str(e) else 500
        raise HTTPException(code, detail=str(e))

    wh_url = None
    if data.get("source_type") == "webhook":
        name = str(data.get("name") or "")
        wh_url = webhook_url_for_kb(name, store, settings=settings, port=settings.port)
    return BuildJobResponse(
        job_id=result["job_id"],
        webhook_url=wh_url,
        folder_push_url=result.get("folder_push_url"),
    )
