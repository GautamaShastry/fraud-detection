"""Liveness and readiness. Readiness checks every dependency the service needs."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import psycopg
import redis.asyncio as aioredis
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest, multiprocess

logger = logging.getLogger(__name__)

HealthCheck = Callable[[], Awaitable[None]]  # Raises on failure

router = APIRouter(tags=["health"])


def postgres_check(dsn: str) -> HealthCheck:
    async def check() -> None:
        async with await psycopg.AsyncConnection.connect(dsn, connect_timeout=2) as conn:
            await conn.execute("SELECT 1")

    return check


def pool_check(pool: Any) -> HealthCheck:
    """Ready only if the serving pool can hand out a working connection."""

    async def check() -> None:
        async with pool.connection(timeout=1.0) as conn:
            await conn.execute("SELECT 1")
            await conn.rollback()

    return check


def redis_check(url: str) -> HealthCheck:
    async def check() -> None:
        client = aioredis.from_url(url)  # type: ignore[no-untyped-call]
        try:
            await client.ping()
        finally:
            await client.aclose()

    return check


async def _run(name: str, check: HealthCheck, timeout_s: float) -> tuple[str, str]:
    try:
        await asyncio.wait_for(check(), timeout_s)
        return name, "ok"
    except Exception as exc:  # noqa: BLE001 - any failure means not ready
        logger.warning("Readiness check failed: %s (%s)", name, type(exc).__name__)
        return name, "unavailable"


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        # Several worker processes: aggregate every process's samples.
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)  # type: ignore[no-untyped-call]
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
async def ready(request: Request) -> JSONResponse:
    checks: Mapping[str, HealthCheck] = request.app.state.readiness_checks
    timeout_s: float = request.app.state.settings.readiness_timeout_s
    results = dict(await asyncio.gather(*(_run(n, c, timeout_s) for n, c in checks.items())))
    healthy = bool(results) and all(v == "ok" for v in results.values())
    return JSONResponse(
        {"status": "ok" if healthy else "unavailable", "checks": results},
        status_code=200 if healthy else 503,
    )
