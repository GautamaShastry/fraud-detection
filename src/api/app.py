"""Application factory."""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from psycopg_pool import PoolTimeout
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from threadpoolctl import threadpool_limits

from src.api import health, outcomes, scoring
from src.api.auth import FileTenantResolver, KeySource, RemoteKeys, StaticKeys, TokenVerifier
from src.api.health import HealthCheck, pool_check, postgres_check, redis_check
from src.api.problems import PROBLEM_MEDIA_TYPE, install_problem_handlers
from src.api.settings import Settings
from src.domain.identity import Pseudonymizer
from src.domain.models import Attempt
from src.features.velocity import CircuitBreaker, VelocityResult, VelocityStore
from src.scoring.artifact import ArtifactError, load_artifact
from src.scoring.engine import Budgets, ScoringEngine, VelocityLookup
from src.scoring.model import ScoringBundle
from src.scoring.stub import stub_bundle
from src.storage.attempts import PostgresTenantResolver
from src.storage.db import create_pool
from src.telemetry.setup import configure_logging, configure_tracing

_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestContextMiddleware:
    """Assigns a request ID and enforces the request body size limit."""

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope["headers"])
        supplied = headers.get(b"x-request-id", b"").decode("latin-1")
        request_id = supplied if _SAFE_REQUEST_ID.match(supplied) else str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        declared = headers.get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self.max_body_bytes):
            await self._too_large(request_id, send_with_id)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    # FastAPI re-raises HTTPException from body parsing; the problem handler
                    # renders it as PAYLOAD_TOO_LARGE.
                    raise HTTPException(status_code=413)
            return message

        await self.app(scope, limited_receive, send_with_id)

    async def _too_large(self, request_id: str, send: Send) -> None:
        body = json.dumps(
            {
                "type": "urn:fraud-api:problem:payload_too_large",
                "title": "Payload too large",
                "status": 413,
                "code": "PAYLOAD_TOO_LARGE",
                "request_id": request_id,
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", PROBLEM_MEDIA_TYPE.encode()),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _load_bundle(settings: Settings) -> ScoringBundle | None:
    if settings.scorer_mode == "stub":
        logging.getLogger(__name__).warning("Using the deterministic TEST STUB scorer")
        return stub_bundle()
    if settings.scorer_mode == "model":
        if settings.model_dir is None or not settings.model_manifest_sha256:
            raise ArtifactError("scorer_mode=model requires model_dir and model_manifest_sha256")
        loaded = load_artifact(settings.model_dir, settings.model_manifest_sha256)
        logging.getLogger(__name__).info(
            "Loaded model %s (policy %s, feature schema %s)",
            loaded.manifest["model_version"],
            loaded.manifest["policy"]["version"],
            loaded.manifest["feature_schema"],
        )
        return loaded.bundle
    return None


def _key_source(path: Path | None, url: str | None) -> KeySource:
    if (path is None) == (url is None):
        raise ValueError("Configure exactly one of a JWKS path or a JWKS URL")
    return StaticKeys.from_file(path) if path is not None else RemoteKeys(url)  # type: ignore[arg-type]


def create_app(
    settings: Settings | None = None,
    *,
    readiness_checks: Mapping[str, HealthCheck] | None = None,
    clock: Callable[[], datetime] | None = None,
    connect: bool = True,
    bundle: ScoringBundle | None = None,
    velocity: VelocityLookup | None = None,
) -> FastAPI:
    """Build the API.

    connect=False skips PostgreSQL/Redis (unit tests): score requests then answer with an
    explicit unavailable response. `bundle` and `velocity` override the configured scorer and
    the Redis velocity lookup for fault-injection tests.
    """
    settings = settings or Settings()  # populated from FRAUD_* env vars
    clock = clock or (lambda: datetime.now(UTC))
    # One native thread per inference: concurrency comes from requests, and OpenMP fan-out
    # for single rows costs far more than it saves.
    threadpool_limits(1)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not connect:
            yield
            return
        configure_logging(settings.log_level)
        configure_tracing("fraud-api")
        pool = create_pool(
            settings.postgres_dsn, min_size=settings.db_pool_min, max_size=settings.db_pool_max
        )
        await pool.open(wait=False)
        try:
            # Fill the pool before taking traffic so first requests don't pay connect latency.
            await pool.wait(timeout=5.0)
        except PoolTimeout:
            logging.getLogger(__name__).error("PostgreSQL pool not ready at startup")
        redis = aioredis.from_url(  # type: ignore[no-untyped-call]
            settings.redis_url, socket_timeout=0.1, socket_connect_timeout=0.1
        )
        app.state.pool = pool
        if readiness_checks is None:
            app.state.readiness_checks = {
                "postgres": pool_check(pool),
                "redis": redis_check(settings.redis_url),
            }
        if settings.tenant_source == "postgres":
            app.state.tenants = PostgresTenantResolver(pool)
        store = VelocityStore(redis)
        try:
            await store.warm()
        except Exception as exc:  # noqa: BLE001 - scoring degrades to fallback without Redis
            logging.getLogger(__name__).error("Redis not ready at startup (%s)", type(exc).__name__)

        breaker = CircuitBreaker()

        async def redis_velocity(attempt: Attempt) -> VelocityResult:
            return await breaker.call(
                lambda: store.f01(
                    attempt.tenant_id,
                    attempt.payment_attempt_id,
                    attempt.device_ref,
                    attempt.payment_method_ref,
                    int(attempt.received_at.timestamp() * 1000),
                )
            )

        active = bundle if bundle is not None else _load_bundle(settings)
        if active is not None:
            app.state.engine = ScoringEngine(
                active,
                velocity or redis_velocity,
                clock,
                Budgets(settings.primary_budget_s, settings.fallback_budget_s),
            )
        try:
            yield
        finally:
            await pool.close()
            await redis.aclose()

    # contracts/openapi.yaml is the published contract; the generated schema is not served.
    app = FastAPI(
        title="Marketplace Fraud Detection API", openapi_url=None, docs_url=None, lifespan=lifespan
    )
    app.state.settings = settings
    app.state.clock = clock
    app.state.pool = None
    app.state.engine = None
    app.state.pseudonymizer = Pseudonymizer(settings.pseudonym_key.get_secret_value().encode())
    app.state.token_verifier = TokenVerifier(
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
        keys=_key_source(settings.jwks_path, settings.jwks_url),
        leeway_s=settings.jwt_leeway_s,
    )
    app.state.tenants = (
        FileTenantResolver.from_file(settings.tenant_map_path)
        if settings.tenant_source == "file" and settings.tenant_map_path is not None
        else FileTenantResolver({})
    )
    app.state.readiness_checks = (
        readiness_checks
        if readiness_checks is not None
        else {
            "postgres": postgres_check(settings.postgres_dsn),
            "redis": redis_check(settings.redis_url),
        }
    )

    install_problem_handlers(app)
    app.include_router(health.router)
    app.include_router(scoring.router)
    app.include_router(outcomes.router)
    if settings.dashboard_enabled:
        from src.dashboard import routes as dashboard

        dashboard.install(app, settings)
    app.add_middleware(RequestContextMiddleware, max_body_bytes=settings.max_body_bytes)
    return app
