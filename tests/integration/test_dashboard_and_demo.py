"""T-401..T-403: dev OIDC flow, dashboard RBAC/RLS, and the checkout demo against the real API."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
import urllib.parse
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.demo.app import FAIL_OPEN
from src.demo.app import create_app as create_demo
from src.devauth.oidc import create_app as create_oidc
from tests.integration.conftest import now_request

pytestmark = pytest.mark.integration

OIDC_ISSUER = "https://dev-oidc.test"
REDIRECT = "http://localhost:8000/dashboard/"


# ---------------------------------------------------------------- dev OIDC provider


@pytest.fixture
def oidc_client(tmp_path):
    users = {
        "users": [
            {
                "username": "ana",
                "tenant_id": "tenant-a",
                "roles": ["analyst"],
                "password_sha256": hashlib.sha256(b"pw").hexdigest(),
            }
        ]
    }
    path = tmp_path / "users.json"
    path.write_text(json.dumps(users))
    app = create_oidc(OIDC_ISSUER, path, {"fraud-dashboard": [REDIRECT]})
    return TestClient(app)


def _pkce() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(b"v" * 40).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge.rstrip(b"=").decode()


def _login(client: TestClient, challenge: str, password: str = "pw") -> httpx.Response:
    form = {
        "client_id": "fraud-dashboard",
        "redirect_uri": REDIRECT,
        "response_type": "code",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": "s1",
        "nonce": "n1",
        "username": "ana",
        "password": password,
    }
    return client.post("/authorize", data=form, follow_redirects=False)


def _code(response: httpx.Response) -> str:
    return urllib.parse.parse_qs(urllib.parse.urlparse(response.headers["location"]).query)["code"][
        0
    ]


def _exchange(client, code, verifier):
    return client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT,
            "client_id": "fraud-dashboard",
            "code_verifier": verifier,
        },
    )


def test_oidc_code_flow_with_pkce(oidc_client):
    verifier, challenge = _pkce()
    assert oidc_client.get("/.well-known/openid-configuration").json()["issuer"] == OIDC_ISSUER
    response = _login(oidc_client, challenge)
    assert response.status_code == 302
    tokens = _exchange(oidc_client, _code(response), verifier).json()
    jwks = jwt.PyJWKSet.from_dict(oidc_client.get("/jwks").json())
    claims = jwt.decode(
        tokens["access_token"],
        jwks.keys[0].key,
        algorithms=["RS256"],
        audience="fraud-dashboard",
        issuer=OIDC_ISSUER,
    )
    assert claims["tenant_id"] == "tenant-a" and claims["roles"] == ["analyst"]


def test_oidc_rejects_bad_verifier_reuse_and_foreign_redirect(oidc_client):
    verifier, challenge = _pkce()
    code = _code(_login(oidc_client, challenge))
    assert _exchange(oidc_client, code, "wrong-verifier").status_code == 400
    code = _code(_login(oidc_client, challenge))
    assert _exchange(oidc_client, code, verifier).status_code == 200
    assert _exchange(oidc_client, code, verifier).status_code == 400  # single use
    assert _login(oidc_client, challenge, password="nope").status_code == 401
    bad = oidc_client.get(
        "/authorize",
        params={
            "client_id": "fraud-dashboard",
            "redirect_uri": "https://evil.example/cb",
            "response_type": "code",
        },
    )
    assert bad.status_code == 400


# ---------------------------------------------------------------- dashboard API


@pytest.fixture
def dashboard(integration_settings, jwks, tmp_path, redis_client):
    jwks_path = tmp_path / "dashboard-jwks.json"
    jwks_path.write_text(json.dumps(jwks))
    settings = integration_settings.model_copy(
        update={
            "dashboard_enabled": True,
            "oidc_issuer": OIDC_ISSUER,
            "oidc_audience": "fraud-dashboard",
            "oidc_jwks_path": jwks_path,
        }
    )
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def user_token(rsa_key):
    def make(role: str | None, tenant: str | None = "tenant-a") -> dict[str, str]:
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": OIDC_ISSUER,
            "aud": "fraud-dashboard",
            "sub": f"u-{role}",
            "iat": now,
            "exp": now + 300,
            "preferred_username": str(role),
        }
        if role:
            claims["roles"] = [role]
        if tenant:
            claims["tenant_id"] = tenant
        token = jwt.encode(claims, rsa_key, algorithm="RS256", headers={"kid": "rsa-1"})
        return {"Authorization": f"Bearer {token}"}

    return make


def _seed(client: TestClient, make_token) -> dict[str, Any]:
    body = now_request()
    headers = {"Authorization": f"Bearer {make_token('merchant-a')}"}
    assert client.post("/api/v1/fraud/score", json=body, headers=headers).status_code == 200
    return body


def test_rbac(dashboard, user_token, make_token):
    body = _seed(dashboard, make_token)
    detail = f"/api/v1/dashboard/attempts/{body['payment_attempt_id']}"
    viewer, analyst, admin = user_token("viewer"), user_token("analyst"), user_token("admin")
    assert dashboard.get("/api/v1/dashboard/attempts", headers=viewer).status_code == 200
    assert dashboard.get("/api/v1/dashboard/metrics", headers=viewer).status_code == 200
    assert dashboard.get(detail, headers=viewer).status_code == 403
    assert dashboard.get(detail, headers=analyst).status_code == 200
    assert dashboard.get("/api/v1/dashboard/admin/config", headers=analyst).status_code == 403
    assert dashboard.get("/api/v1/dashboard/admin/config", headers=admin).status_code == 200
    assert dashboard.get("/api/v1/dashboard/attempts").status_code == 401
    assert (
        dashboard.get("/api/v1/dashboard/attempts", headers=user_token(None)).status_code == 403
    )  # no role claim
    assert (
        dashboard.get(
            "/api/v1/dashboard/attempts", headers=user_token("analyst", tenant=None)
        ).status_code
        == 403
    )
    # A merchant API token is not a dashboard token (different audience).
    merchant = {"Authorization": f"Bearer {make_token('merchant-a')}"}
    assert dashboard.get("/api/v1/dashboard/attempts", headers=merchant).status_code == 401


def test_tenant_isolation_and_label_states(dashboard, user_token, make_token, owner_conn):
    body = _seed(dashboard, make_token)
    attempt = body["payment_attempt_id"]
    other = user_token("analyst", tenant="tenant-b")
    assert dashboard.get(f"/api/v1/dashboard/attempts/{attempt}", headers=other).status_code == 404
    listed = dashboard.get("/api/v1/dashboard/attempts", headers=other).json()["items"]
    assert attempt not in {r["payment_attempt_id"] for r in listed}

    analyst = user_token("analyst")
    detail = dashboard.get(f"/api/v1/dashboard/attempts/{attempt}", headers=analyst).json()
    assert detail["label"] == "UNLABELED"
    assert body["device_id"] not in json.dumps(detail)  # raw identifiers are never shown
    owner_conn.execute(
        "INSERT INTO attempt_outcomes (tenant_id, payment_attempt_id, label, label_source, "
        "event_count, updated_at) VALUES ('tenant-a', %s, 'CONFIRMED_FRAUD', 'ISSUER', 1, now())",
        (attempt,),
    )
    detail = dashboard.get(f"/api/v1/dashboard/attempts/{attempt}", headers=analyst).json()
    assert detail["label"] == "CONFIRMED_FRAUD"
    metrics = dashboard.get("/api/v1/dashboard/metrics", headers=analyst).json()["labels"]
    assert metrics["confirmed_fraud"] >= 1 and metrics["unlabeled"] >= 0


def test_pagination_and_audit(dashboard, user_token, make_token):
    for _ in range(3):
        _seed(dashboard, make_token)
    admin = user_token("admin")
    page = dashboard.get("/api/v1/dashboard/attempts?limit=2", headers=admin).json()
    assert len(page["items"]) == 2 and page["next_cursor"]
    rest = dashboard.get(
        f"/api/v1/dashboard/attempts?limit=2&cursor={page['next_cursor']}", headers=admin
    ).json()
    assert not {r["payment_attempt_id"] for r in page["items"]} & {
        r["payment_attempt_id"] for r in rest["items"]
    }
    assert dashboard.get("/api/v1/dashboard/attempts?cursor=%%%", headers=admin).status_code == 422
    audit = dashboard.get("/api/v1/dashboard/admin/audit", headers=admin).json()["items"]
    assert {"list_attempts"} <= {a["action"] for a in audit}


def test_dashboard_ui_is_served(dashboard):
    assert "Fraud Monitoring" in dashboard.get("/dashboard/").text
    assert dashboard.get("/dashboard/config.json").json()["issuer"] == OIDC_ISSUER


# ---------------------------------------------------------------- checkout demo (T-403)


class _ForwardToTestClient(httpx.AsyncBaseTransport):
    """Routes the demo's HTTP calls into the real API app (with its lifespan running)."""

    def __init__(self, client: TestClient) -> None:
        self._client = client

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        content = await request.aread()
        r = await asyncio.to_thread(
            self._client.request,
            request.method,
            request.url.path,
            content=content,
            headers=dict(request.headers),
        )
        return httpx.Response(r.status_code, headers=r.headers, content=r.content)


class _Tokens:
    def __init__(self, token: str) -> None:
        self.token = token

    def get(self) -> str:
        return self.token


class _Demo:
    """Demo app with its lifespan running (it owns pooled HTTP clients)."""

    def __init__(self, api_client: TestClient, token: str) -> None:
        app = create_demo("http://api", _Tokens(token), _ForwardToTestClient(api_client), 2.0)
        self._client = TestClient(app)

    def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
        with self._client as client:
            return client.post(*args, **kwargs)

    def get(self, *args: Any, **kwargs: Any) -> httpx.Response:
        with self._client as client:
            return client.get(*args, **kwargs)


def test_demo_uses_real_api_and_shows_recommendation(live_client, make_token):
    demo = _Demo(live_client, make_token("merchant-a"))
    r = demo.post(
        "/checkout",
        json={"amount_minor": 4599, "device_id": "dev-x", "payment_method_token": "tok-x"},
    ).json()
    assert r["status"] == 200
    assert r["response"]["recommendation"] in {"PROCEED", "VERIFY", "RECOMMEND_DECLINE"}
    assert "simulated" in r["payment_decision"]


def test_demo_fails_open_on_outage(live_client, make_token):
    started = time.perf_counter()
    with TestClient(
        create_demo("http://api", _Tokens(make_token("merchant-a")), None, 0.2)
    ) as demo:
        r = demo.post(
            "/checkout",
            json={
                "amount_minor": 100,
                "device_id": "d",
                "payment_method_token": "t",
                "simulate_outage": True,
            },
        ).json()
    assert r["status"] is None and r["error"]
    assert r["payment_decision"] == FAIL_OPEN
    assert time.perf_counter() - started < 1.5  # bounded by the 200 ms client timeout


def test_demo_fails_open_when_scoring_unavailable(integration_settings, make_token, redis_client):
    settings = integration_settings.model_copy(update={"scorer_mode": "none"})
    with TestClient(create_app(settings), raise_server_exceptions=False) as api:
        r = (
            _Demo(api, make_token("merchant-a"))
            .post(
                "/checkout",
                json={"amount_minor": 100, "device_id": "d", "payment_method_token": "t"},
            )
            .json()
        )
    assert r["status"] == 503 and r["payment_decision"] == FAIL_OPEN


def test_demo_never_bypasses_auth(live_client):
    r = (
        _Demo(live_client, "not-a-valid-token")
        .post(
            "/checkout", json={"amount_minor": 100, "device_id": "d", "payment_method_token": "t"}
        )
        .json()
    )
    assert r["status"] == 401 and "not bypassed" in r["payment_decision"]


def test_demo_page(live_client, make_token):
    assert "Synthetic demo" in _Demo(live_client, make_token()).get("/").text


@pytest.fixture(autouse=True)
def _repo_root(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
