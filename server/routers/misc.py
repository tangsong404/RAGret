from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from server.config import Settings
from server.deps import get_repo_root, get_settings
from server.skill_pack import build_skill_zip, skill_md_path
from server.webhook_urls import webhook_base_urls

router = APIRouter(prefix="/api", tags=["misc"])


@router.get("/webhook-base")
def webhook_base(settings: Settings = Depends(get_settings)):
    bases = webhook_base_urls(settings, port=settings.port)
    return {"ok": True, "base_url": bases["gitlab"], "bases": bases}


@router.get("/skill-md")
def skill_md(repo_root: Path = Depends(get_repo_root)):
    p = skill_md_path(repo_root)
    if not p.is_file():
        raise HTTPException(404, detail="skill/SKILL.md not found")
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise HTTPException(500, detail=str(e))
    return {"ok": True, "content": text, "filename": "SKILL.md"}


@router.get("/skill-md/download")
def skill_md_download(repo_root: Path = Depends(get_repo_root)):
    try:
        body = build_skill_zip(repo_root)
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e)) from e
    except OSError as e:
        raise HTTPException(500, detail=str(e)) from e
    return Response(
        content=body,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="ragret.zip"'},
    )
