from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ragret.registry import IndexRegistry
from pathlib import Path

from server.deps import get_registry, get_repo_root, get_store, require_actor
from server.services import admin_service
from server.store.protocol import AppStore

router = APIRouter(prefix="/api", tags=["admin"])


@router.delete("/indexes/{name}")
def delete_index(
    name: str,
    delete_sqlite: bool = Query(default=True, alias="delete_sqlite"),
    actor: dict = Depends(require_actor),
    store: AppStore = Depends(get_store),
    registry: IndexRegistry = Depends(get_registry),
    repo_root: Path = Depends(get_repo_root),
):
    flag = delete_sqlite not in (False, "0", "false", "False")
    try:
        result = admin_service.delete_index(name, actor, store, registry, repo_root, delete_sqlite=flag)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(403, detail=str(e))
    except LookupError as e:
        raise HTTPException(404, detail=str(e))
    return {"ok": True, **result}
