from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backfill.api import routers
from backfill.container import Container
from backfill.errors import AppError
from backfill.observability import metrics as m
from backfill.observability.logging import configure_logging, get_logger
from backfill.settings import Settings

log = get_logger(__name__)


def _error(status: int, code: str, message: str, request_id: str, details: object = None) -> JSONResponse:
    body: dict[str, object] = {"error": {"code": code, "message": message, "request_id": request_id}}
    if details is not None:
        body["error"]["details"] = details  # type: ignore[index]
    return JSONResponse(status_code=status, content=body, headers={"X-Request-ID": request_id})


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or (container.settings if container else Settings.from_env())
    configure_logging(settings.log_level, settings.log_json)
    built = container or Container.build(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if built.settings.worker_enabled:
            built.worker.start()
        log.info("service_started", routing=built.settings.routing_mode, worker=built.settings.worker_enabled)
        yield
        built.worker.stop()

    app = FastAPI(
        title="Billable Distance Service",
        version="1.0.0",
        description="Rebuilds trip distance from unreliable GPS and explains every kilometre.",
        lifespan=lifespan,
    )
    app.state.container = built

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        m.HTTP_REQUESTS.labels(path, request.method, str(response.status_code)).inc()
        m.HTTP_SECONDS.labels(path, request.method).observe(elapsed)
        response.headers["X-Request-ID"] = request_id
        log.info(
            "http_request",
            method=request.method,
            route=path,
            status=response.status_code,
            ms=round(elapsed * 1000, 2),
        )
        return response

    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message, request.state.request_id)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Report where and why, never the submitted value: it may be a home location.
        details = [{"field": ".".join(str(p) for p in e["loc"]), "problem": e["msg"]} for e in exc.errors()]
        return _error(422, "INVALID_INPUT", "The request is not valid.", request.state.request_id, details)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        return _error(exc.status_code, code, str(exc.detail), request.state.request_id)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_error", error=type(exc).__name__)
        return _error(500, AppError.code, AppError.message, request.state.request_id)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        body, content_type = m.render_latest()
        return Response(body, media_type=content_type)

    for r in (routers.health, routers.auth, routers.trips, routers.ops):
        app.include_router(r)
    return app


def app_from_env() -> FastAPI:
    return create_app(Settings.from_env())
