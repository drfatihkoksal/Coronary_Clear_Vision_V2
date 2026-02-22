import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

from app.models.responses import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


def add_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict):
            code = exc.detail.get("code", "HTTP_ERROR")
            message = exc.detail.get("message", str(exc.detail))
        else:
            code = "HTTP_ERROR"
            message = str(exc.detail)

        body = ErrorResponse(error=ErrorDetail(code=code, message=message))
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        body = ErrorResponse(error=ErrorDetail(code="VALIDATION_ERROR", message=str(exc)))
        return JSONResponse(status_code=400, content=body.model_dump())

    @app.exception_handler(IndexError)
    async def index_error_handler(request: Request, exc: IndexError) -> JSONResponse:
        body = ErrorResponse(error=ErrorDetail(code="INDEX_ERROR", message=str(exc)))
        return JSONResponse(status_code=400, content=body.model_dump())

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
        body = ErrorResponse(error=ErrorDetail(code="NOT_FOUND", message=str(exc)))
        return JSONResponse(status_code=404, content=body.model_dump())

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
        body = ErrorResponse(
            error=ErrorDetail(code="INTERNAL_ERROR", message="An internal error occurred")
        )
        return JSONResponse(status_code=500, content=body.model_dump())
