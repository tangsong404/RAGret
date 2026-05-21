from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile

from server.deps import get_store, get_upload_base
from server.services import push_service
from server.store.protocol import AppStore

router = APIRouter(prefix="/api/push", tags=["push"])


def _token_headers(
    x_webhook_token: str | None = Header(None, alias="X-Webhook-Token"),
    x_gitlab_token: str | None = Header(None, alias="X-Gitlab-Token"),
) -> dict[str, str]:
    return {
        "X-Webhook-Token": str(x_webhook_token or ""),
        "X-Gitlab-Token": str(x_gitlab_token or ""),
    }


@router.post("/{name}")
async def push_kb_archive(
    name: str,
    file: UploadFile = File(...),
    headers: dict[str, str] = Depends(_token_headers),
    store: AppStore = Depends(get_store),
    upload_base: Path = Depends(get_upload_base),
):
    token = push_service.push_token_from_headers(headers)
    if not token:
        raise HTTPException(401, detail="Missing token")
    if not file.filename:
        raise HTTPException(400, detail="Missing archive filename")
    try:
        result = push_service.enqueue_push_update(
            name, token, file.file, file.filename, store, upload_base
        )
    except ValueError as e:
        raise HTTPException(400, detail=str(e)) from e
    except PermissionError as e:
        raise HTTPException(403, detail=str(e)) from e
    except LookupError as e:
        raise HTTPException(404, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(409, detail=str(e)) from e
    except RuntimeError as e:
        code = 429 if "Too many upload" in str(e) else 500
        raise HTTPException(code, detail=str(e)) from e
    return {"ok": True, **result}


@router.get("/{name}/fingerprints")
def push_kb_fingerprints(
    name: str,
    headers: dict[str, str] = Depends(_token_headers),
    store: AppStore = Depends(get_store),
):
    token = push_service.push_token_from_headers(headers)
    if not token:
        raise HTTPException(401, detail="Missing token")
    try:
        body = push_service.get_push_fingerprints(name, token, store)
    except ValueError as e:
        raise HTTPException(400, detail=str(e)) from e
    except PermissionError as e:
        raise HTTPException(403, detail=str(e)) from e
    except LookupError as e:
        raise HTTPException(404, detail=str(e)) from e
    except FileNotFoundError as e:
        code = 409 if "not ready" in str(e).lower() else 404
        raise HTTPException(code, detail=str(e)) from e
    return {"ok": True, **body}
