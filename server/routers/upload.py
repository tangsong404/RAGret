from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from server.deps import get_upload_base, require_user_id
from server.services import upload_service

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload")
async def upload_archive(
    file: UploadFile = File(...),
    _user_id: int = Depends(require_user_id),
    upload_base: Path = Depends(get_upload_base),
):
    if not file.filename:
        raise HTTPException(400, detail="Missing archive filename")
    try:
        upload_id = upload_service.stage_archive_upload(file.file, file.filename, upload_base)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    return {"ok": True, "upload_id": upload_id}
