"""Integration fixtures: a fresh migrated PostgreSQL database and a clean Redis DB per session.

Uses the local compose services (or CI service containers). Tests are skipped when they are
unreachable, unless FRAUD_REQUIRE_INTEGRATION=1 (set in CI), which turns skips into failures.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
import redis
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.settings import Settings
from src.storage.migrate import migrate, set_role_passwords
from tests.conftest import AUDIENCE, ISSUER

OWNER_DSN = os.environ.get(
    "FRAUD_TEST_OWNER_DSN", "postgresql://fraud:REMOVED-CREDENTIAL@127.0.0.1:55432/postgres"
)
REDIS_URL = os.environ.get("FRAUD_TEST_REDIS_URL", "redis://127.0.0.1:56379/15")
APP_PASSWORD = "REMOVED-CREDENTIAL"
WORKER_PASSWORD = "REMOVED-CREDENTIAL"

TENANTS = {"tenant-a": "Tenant A", "tenant-b": "Tenant B"}
CLIENTS = {"merchant-a": "tenant-a", "merchant-b": "tenant-b"}


def _unreachable(reason: str) -> None:
    if os.environ.get("FRAUD_REQUIRE_INTEGRATION") == "1":
        pytest.fail(f"Integration dependencies required but unavailable: {reason}")
    pytest.skip(f"Integration dependencies unavailable: {reason}")


def _with_db(dsn: str, dbname: str) -> str:
    return psycopg.conninfo.make_conninfo(dsn, dbname=dbname)


@pytest.fixture(scope="session")
def database() -> Iterator[dict[str, str]]:
    try:
        admin = psycopg.connect(OWNER_DSN, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as exc:
        _unreachable(str(exc).splitlines()[0])
    name = f"fraud_test_{uuid.uuid4().hex[:8]}"
    admin.execute(f'CREATE DATABASE "{name}"')
    owner_dsn = _with_db(OWNER_DSN, name)
    os.environ["FRAUD_APP_DB_PASSWORD"] = APP_PASSWORD
    os.environ["FRAUD_WORKER_DB_PASSWORD"] = WORKER_PASSWORD
    with psycopg.connect(owner_dsn) as conn:
        migrate(conn)
        set_role_passwords(conn)
        conn.execute(f'GRANT CONNECT ON DATABASE "{name}" TO fraud_app, fraud_worker')
        for tenant_id, display in TENANTS.items():
            conn.execute("INSERT INTO tenants VALUES (%s, %s)", (tenant_id, display))
        for client_id, tenant_id in CLIENTS.items():
            conn.execute(
                "INSERT INTO api_clients (client_id, tenant_id) VALUES (%s, %s)",
                (client_id, tenant_id),
            )
        conn.commit()

    info = psycopg.conninfo.conninfo_to_dict(owner_dsn)
    app_dsn = psycopg.conninfo.make_conninfo(owner_dsn, user="fraud_app", password=APP_PASSWORD)
    worker_dsn = psycopg.conninfo.make_conninfo(
        owner_dsn, user="fraud_worker", password=WORKER_PASSWORD
    )
    yield {"owner": owner_dsn, "app": app_dsn, "worker": worker_dsn, "name": str(info["dbname"])}

    admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
    admin.close()


@pytest.fixture
def redis_client() -> Iterator[redis.Redis]:
    client = redis.Redis.from_url(REDIS_URL)
    try:
        client.ping()
    except redis.ConnectionError as exc:
        _unreachable(str(exc))
    client.flushdb()
    yield client
    client.flushdb()
    client.close()


@pytest.fixture
def owner_conn(database: dict[str, str]) -> Iterator[psycopg.Connection[Any]]:
    with psycopg.connect(database["owner"], autocommit=True) as conn:
        yield conn


@pytest.fixture
def integration_settings(app_settings: Settings, database: dict[str, str]) -> Settings:
    return app_settings.model_copy(
        update={
            "tenant_source": "postgres",
            "postgres_dsn": database["app"],
            "redis_url": REDIS_URL,
            "scorer_mode": "stub",
            # Functional tests get deadline headroom on shared CI/dev machines; deadline
            # behavior has its own test and latency is measured by the performance suite.
            "server_deadline_s": 0.5,
            "jwt_issuer": ISSUER,
            "jwt_audience": AUDIENCE,
        }
    )


def now_request(**overrides: Any) -> dict[str, Any]:
    """A valid score request stamped with the current time."""
    from datetime import UTC, datetime

    suffix = uuid.uuid4().hex[:10]
    body: dict[str, Any] = {
        "transaction_id": f"order-{suffix}",
        "payment_attempt_id": f"pa-{suffix}",
        "idempotency_key": f"idem-{suffix}",
        "amount_minor": 4599,
        "currency": "USD",
        "occurred_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device_id": f"dev-{suffix}",
        "payment_method_token": f"tok-{suffix}",
    }
    body.update(overrides)
    return {k: v for k, v in body.items() if v is not None}


@pytest.fixture
def live_client(integration_settings: Settings, redis_client: redis.Redis) -> Iterator[TestClient]:
    app = create_app(integration_settings)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
