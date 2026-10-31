"""RFC 9457 problem responses with stable error codes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"


class ProblemError(Exception):
    """Raise to return a problem response."""

    def __init__(
        self,
        status: int,
        code: str,
        title: str,
        detail: str | None = None,
        *,
        retryable: bool | None = None,
        errors: list[dict[str, str]] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(title)
        self.status = status
        self.code = code
        self.title = title
        self.detail = detail
        self.retryable = retryable
        self.errors = errors
        self.headers = headers


def problem_response(request: Request, problem: ProblemError) -> JSONResponse:
    body: dict[str, Any] = {
        "type": f"urn:fraud-api:problem:{problem.code.lower()}",
        "title": problem.title,
        "status": problem.status,
        "code": problem.code,
        "request_id": getattr(request.state, "request_id", ""),
    }
    if problem.detail is not None:
        body["detail"] = problem.detail
    if problem.retryable is not None:
        body["retryable"] = problem.retryable
    if problem.errors is not None:
        body["errors"] = problem.errors
    return JSONResponse(
        body, status_code=problem.status, media_type=PROBLEM_MEDIA_TYPE, headers=problem.headers
    )


def _field_name(loc: tuple[Any, ...]) -> str:
    # Drop the leading "body" segment; keep the field path only.
    parts = [str(p) for p in loc[1:]] if loc and loc[0] == "body" else [str(p) for p in loc]
    return ".".join(parts) or "body"


def install_problem_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProblemError)
    async def _problem(request: Request, exc: ProblemError) -> JSONResponse:
        return problem_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Only the location and message are returned; submitted values are never echoed.
        errors = [{"field": _field_name(e["loc"]), "message": e["msg"]} for e in exc.errors()]
        return problem_response(
            request,
            ProblemError(422, "VALIDATION_ERROR", "Request validation failed", errors=errors),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 405:
            problem = ProblemError(405, "METHOD_NOT_ALLOWED", "Method not allowed")
        elif exc.status_code == 413:
            problem = ProblemError(413, "PAYLOAD_TOO_LARGE", "Payload too large")
        elif exc.status_code == 404:
            problem = ProblemError(404, "NOT_FOUND", "Not found")
        else:
            problem = ProblemError(exc.status_code, "INTERNAL_ERROR", "Request failed")
        return problem_response(request, problem)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled error", extra={"request_id": getattr(request.state, "request_id", "")}
        )
        return problem_response(request, ProblemError(500, "INTERNAL_ERROR", "Internal error"))
