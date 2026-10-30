import asyncio
import json
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, settings

from src.api.app import create_app
from src.api.settings import Settings

# psycopg's async driver needs a selector event loop on Windows.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Property tests run at least 100 examples each.
settings.register_profile(
    "default", max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile("default")

ISSUER = "https://test-issuer.invalid"
AUDIENCE = "fraud-api"
FIXED_NOW = datetime(2026, 10, 28, 12, 0, 30, tzinfo=UTC)
TENANT_MAP = {"merchant-a": "tenant-a", "merchant-b": "tenant-b"}
TEST_PSEUDONYM_KEY = "REMOVED-CREDENTIAL"


@pytest.fixture(scope="session")
def rsa_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="session")
def ec_key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


@pytest.fixture(scope="session")
def jwks(rsa_key: rsa.RSAPrivateKey, ec_key: ec.EllipticCurvePrivateKey) -> dict[str, Any]:
    rsa_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(rsa_key.public_key()))
    rsa_jwk.update({"kid": "rsa-1", "alg": "RS256", "use": "sig"})
    ec_jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(ec_key.public_key()))
    ec_jwk.update({"kid": "ec-1", "alg": "ES256", "use": "sig"})
    return {"keys": [rsa_jwk, ec_jwk]}


@pytest.fixture
def app_settings(tmp_path: Path, jwks: dict[str, Any]) -> Settings:
    jwks_path, tenants_path = tmp_path / "jwks.json", tmp_path / "tenants.json"
    jwks_path.write_text(json.dumps(jwks))
    tenants_path.write_text(json.dumps(TENANT_MAP))
    return Settings(
        jwt_issuer=ISSUER,
        jwt_audience=AUDIENCE,
        jwks_path=jwks_path,
        tenant_source="file",
        tenant_map_path=tenants_path,
        postgres_dsn="postgresql://unused",
        redis_url="redis://unused",
        pseudonym_key=TEST_PSEUDONYM_KEY,
    )


async def _ok() -> None:
    return None


@pytest.fixture
def client(app_settings: Settings) -> TestClient:
    app = create_app(
        app_settings,
        readiness_checks={"postgres": _ok, "redis": _ok},
        clock=lambda: FIXED_NOW,
        connect=False,
    )
    return TestClient(app, raise_server_exceptions=False)


TokenFactory = Callable[..., str]


@pytest.fixture
def make_token(rsa_key: rsa.RSAPrivateKey) -> TokenFactory:
    def factory(
        client_id: str = "merchant-a",
        scope: str = "fraud:score",
        *,
        key: Any = None,
        kid: str = "rsa-1",
        algorithm: str = "RS256",
        **overrides: Any,
    ) -> str:
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": client_id,
            "client_id": client_id,
            "scope": scope,
            "iat": now,
            "exp": now + 300,
        }
        claims.update(overrides)
        claims = {k: v for k, v in claims.items() if v is not None}
        return jwt.encode(
            claims, key if key is not None else rsa_key, algorithm=algorithm, headers={"kid": kid}
        )

    return factory


@pytest.fixture
def auth_header(make_token: TokenFactory) -> dict[str, str]:
    return {"Authorization": f"Bearer {make_token()}"}


def valid_score_request(**overrides: Any) -> dict[str, Any]:
    body = {
        "transaction_id": "order-1",
        "payment_attempt_id": "pa-1",
        "idempotency_key": "idem-0001",
        "amount_minor": 4599,
        "currency": "USD",
        "occurred_at": "2026-10-28T12:00:00Z",
    }
    body.update(overrides)
    return {k: v for k, v in body.items() if v is not None}
