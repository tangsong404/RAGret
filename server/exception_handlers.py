from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


def _error_body(detail: object) -> dict[str, object]:
    if isinstance(detail, str):
        msg = detail
    elif isinstance(detail, list) and detail:
        first = detail[0]
        if isinstance(first, dict):
            loc = first.get("loc", ())
            msg = str(first.get("msg", "Validation error"))
            if loc:
                msg = f"{'.'.join(str(x) for x in loc)}: {msg}"
        else:
            msg = str(first)
    else:
        msg = str(detail)
    return {"ok": False, "error": msg}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=_error_body(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content=_error_body(exc.errors()))
