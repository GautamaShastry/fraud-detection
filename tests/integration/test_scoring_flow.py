"""T-102/T-104/T-105 against real PostgreSQL and Redis: lifecycle, idempotency, degradation."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.features.velocity import VelocityResult
from src.scoring.model import ScoringBundle
from src.scoring.stub import stub_bundle
from tests.integration.conftest import now_request
from tests.support.contract import assert_response_matches

pytestmark = pytest.mark.integration

SCORE = "/api/v1/fraud/score"


def _post(client: TestClient, headers: dict[str, str], body: dict[str, Any]):
    return client.post(SCORE, json=body, headers=headers)


def test_score_persists_and_returns_assessment(live_client, auth_header, owner_conn):
    body = now_request()
    response = _post(live_client, auth_header, body)
    assert response.status_code == 200, response.text
    assert_response_matches("POST", SCORE, response)
    data = response.json()
    assert data["scoring_mode"] == "ML"
    assert data["model_version"] == "stub-deterministic-0"  # labeled stub, never "real" ML
    row = owner_conn.execute(
        "SELECT a.status, s.assessment_id::text FROM attempts a JOIN assessments s "
        "USING (tenant_id, payment_attempt_id) WHERE a.payment_attempt_id = %s",
        (body["payment_attempt_id"],),
    ).fetchone()
    assert row == ("COMPLETED", data["assessment_id"])


def test_identical_retry_returns_same_assessment_without_recounting(live_client, auth_header):
    body = now_request()
    first = _post(live_client, auth_header, body).json()
    second = _post(live_client, auth_header, body).json()
    assert first == second
    # A new attempt with another token on the same device sees exactly 2 distinct methods.
    other = now_request(device_id=body["device_id"])
    third = _post(live_client, auth_header, other).json()
    assert third["assessment_id"] != first["assessment_id"]


def test_key_reuse_with_different_body_is_conflict(live_client, auth_header):
    body = now_request()
    assert _post(live_client, auth_header, body).status_code == 200
    changed = {**body, "amount_minor": body["amount_minor"] + 1}
    response = _post(live_client, auth_header, changed)
    assert response.status_code == 409
    assert response.json()["code"] == "IDEMPOTENCY_CONFLICT"
    assert_response_matches("POST", SCORE, response)


def test_attempt_reuse_with_different_key_is_conflict(live_client, auth_header):
    body = now_request()
    assert _post(live_client, auth_header, body).status_code == 200
    response = _post(live_client, auth_header, {**body, "idempotency_key": "idem-other-key"})
    assert response.status_code == 409


def test_parallel_identical_requests_produce_one_assessment(live_client, auth_header, owner_conn):
    assert _post(live_client, auth_header, now_request()).status_code == 200  # warm the pool
    body = now_request()
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: _post(live_client, auth_header, body), range(16)))
    statuses = {r.status_code for r in responses}
    # Concurrent duplicates either see the result or are told to retry; a claim holder that
    # misses the deadline answers 504 and releases its claim.
    assert statuses <= {200, 202, 504}, statuses
    ids = {r.json()["assessment_id"] for r in responses if r.status_code == 200}
    deadline = time.monotonic() + 5
    while True:
        final = _post(live_client, auth_header, body)
        if final.status_code == 200 or time.monotonic() > deadline:
            break
        time.sleep(0.05)
    assert final.status_code == 200
    ids.add(final.json()["assessment_id"])
    assert len(ids) == 1
    count = owner_conn.execute(
        "SELECT count(*) FROM assessments WHERE payment_attempt_id = %s",
        (body["payment_attempt_id"],),
    ).fetchone()
    assert count == (1,)
    for r in responses:
        assert_response_matches("POST", SCORE, r)


def test_get_by_attempt(live_client, auth_header):
    body = now_request()
    created = _post(live_client, auth_header, body).json()
    url = f"/api/v1/fraud/attempts/{body['payment_attempt_id']}/assessment"
    response = live_client.get(url, headers=auth_header)
    assert response.status_code == 200
    assert response.json() == created
    assert_response_matches(
        "GET", "/api/v1/fraud/attempts/{payment_attempt_id}/assessment", response
    )
    missing = live_client.get("/api/v1/fraud/attempts/pa-missing/assessment", headers=auth_header)
    assert missing.status_code == 404


def test_tenant_cannot_read_or_collide_with_other_tenant(live_client, make_token):
    a = {"Authorization": f"Bearer {make_token('merchant-a')}"}
    b = {"Authorization": f"Bearer {make_token('merchant-b')}"}
    body = now_request()
    created = _post(live_client, a, body).json()
    url = f"/api/v1/fraud/attempts/{body['payment_attempt_id']}/assessment"
    assert live_client.get(url, headers=b).status_code == 404
    # Tenant B reusing the same IDs is an unrelated attempt, not a conflict or a replay.
    other = _post(live_client, b, body)
    assert other.status_code == 200
    assert other.json()["assessment_id"] != created["assessment_id"]


def test_raw_identifiers_never_stored(live_client, auth_header, owner_conn):
    body = now_request(customer_id="cust-RAW-42", ip_address="203.0.113.99")
    assert _post(live_client, auth_header, body).status_code == 200
    row = owner_conn.execute(
        "SELECT row_to_json(a)::text FROM attempts a WHERE payment_attempt_id = %s",
        (body["payment_attempt_id"],),
    ).fetchone()
    stored = row[0]
    for raw in (body["device_id"], body["payment_method_token"], "cust-RAW-42", "203.0.113.99"):
        assert raw not in stored


def test_expired_lease_is_taken_over(live_client, auth_header, owner_conn):
    body = now_request()
    # Simulate a crashed request: claimed long ago, never completed.
    first = _post(live_client, auth_header, body)
    assert first.status_code == 200
    owner_conn.execute(
        "DELETE FROM assessments WHERE payment_attempt_id = %s", (body["payment_attempt_id"],)
    )
    owner_conn.execute(
        "UPDATE attempts SET status = 'IN_PROGRESS', lease_expires_at = now() - interval '1 s' "
        "WHERE payment_attempt_id = %s",
        (body["payment_attempt_id"],),
    )
    retry = _post(live_client, auth_header, body)
    assert retry.status_code == 200
    assert retry.json()["assessment_id"] != first.json()["assessment_id"]


def test_live_lease_returns_in_progress(live_client, auth_header, owner_conn):
    body = now_request()
    assert _post(live_client, auth_header, body).status_code == 200
    owner_conn.execute(
        "DELETE FROM assessments WHERE payment_attempt_id = %s", (body["payment_attempt_id"],)
    )
    owner_conn.execute(
        "UPDATE attempts SET status = 'IN_PROGRESS', lease_expires_at = now() + interval '30 s' "
        "WHERE payment_attempt_id = %s",
        (body["payment_attempt_id"],),
    )
    response = _post(live_client, auth_header, body)
    assert response.status_code == 202
    assert response.json()["status"] == "IN_PROGRESS"
    assert_response_matches("POST", SCORE, response)


# ---------------------------------------------------------------- fault injection (T-105)


@dataclass
class SlowScorer:
    model_version: str
    feature_names: tuple[str, ...]
    delay_s: float

    def predict(self, features: dict[str, float | None]) -> float:
        time.sleep(self.delay_s)
        return 0.1


@dataclass
class BrokenScorer:
    model_version: str
    feature_names: tuple[str, ...]

    def predict(self, features: dict[str, float | None]) -> float:
        raise RuntimeError("model crashed")


def _bundle(primary: Any = None, fallback: Any = "keep") -> ScoringBundle:
    base = stub_bundle()
    return ScoringBundle(
        primary=primary or base.primary,
        primary_policy=base.primary_policy,
        fallback=base.fallback if fallback == "keep" else fallback,
        fallback_policy=base.fallback_policy,
        signal_rules=base.signal_rules,
    )


def _client(settings: Any, **kwargs: Any) -> TestClient:
    return TestClient(create_app(settings, **kwargs), raise_server_exceptions=False)


async def _redis_down(attempt: Any) -> VelocityResult:
    raise ConnectionError("redis unavailable")


async def _redis_hangs(attempt: Any) -> VelocityResult:
    await asyncio.sleep(5)
    raise AssertionError("unreachable")


@pytest.mark.parametrize("velocity", [_redis_down, _redis_hangs], ids=["error", "timeout"])
def test_velocity_failure_degrades_to_fallback(integration_settings, auth_header, velocity):
    settings = integration_settings.model_copy(update={"server_deadline_s": 0.5})
    with _client(settings, velocity=velocity) as client:
        response = _post(client, auth_header, now_request())
    assert response.status_code == 200
    data = response.json()
    assert data["scoring_mode"] == "FALLBACK"
    assert data["recommendation"] != "RECOMMEND_DECLINE"
    assert_response_matches("POST", SCORE, response)


def test_primary_timeout_degrades_to_fallback(integration_settings, auth_header):
    slow = SlowScorer("slow-0", stub_bundle().primary.feature_names, 0.5)
    settings = integration_settings.model_copy(update={"server_deadline_s": 0.4})
    with _client(settings, bundle=_bundle(primary=slow)) as client:
        started = time.perf_counter()
        response = _post(client, auth_header, now_request())
        elapsed = time.perf_counter() - started
    assert response.status_code == 200
    assert response.json()["scoring_mode"] == "FALLBACK"
    assert elapsed < 0.5  # did not wait for the slow model


def test_primary_and_fallback_failure_is_explicit_unavailable(
    integration_settings, auth_header, owner_conn
):
    broken = BrokenScorer("broken-0", ())
    body = now_request()
    with _client(integration_settings, bundle=_bundle(primary=broken, fallback=broken)) as client:
        response = _post(client, auth_header, body)
    assert response.status_code == 503
    payload = response.json()
    assert payload["code"] == "SCORING_UNAVAILABLE" and "risk_score" not in payload
    assert_response_matches("POST", SCORE, response)
    status = owner_conn.execute(
        "SELECT status FROM attempts WHERE payment_attempt_id = %s", (body["payment_attempt_id"],)
    ).fetchone()
    assert status == ("UNAVAILABLE",)

    # After recovery, the same attempt is scored normally.
    with _client(integration_settings) as client:
        retry = _post(client, auth_header, body)
    assert retry.status_code == 200


def test_deadline_exceeded_returns_504(integration_settings, auth_header):
    settings = integration_settings.model_copy(update={"server_deadline_s": 0.05})
    slow = SlowScorer("slow-0", stub_bundle().fallback.feature_names, 0.5)  # type: ignore[union-attr]
    with _client(settings, bundle=_bundle(primary=slow, fallback=slow), velocity=_redis_hangs) as c:
        response = _post(c, auth_header, now_request())
    assert response.status_code == 504
    assert response.json()["code"] == "SCORING_DEADLINE_EXCEEDED"
    assert_response_matches("POST", SCORE, response)


def test_postgres_down_fails_open_quickly(integration_settings, auth_header):
    settings = integration_settings.model_copy(
        update={"postgres_dsn": "postgresql://fraud_app:x@127.0.0.1:1/none?connect_timeout=1"}
    )
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        started = time.perf_counter()
        response = _post(client, auth_header, now_request())
        elapsed = time.perf_counter() - started
    # Tenant mapping lives in PostgreSQL, so the caller cannot be resolved: explicit 503,
    # never an unauthenticated pass-through and never a fabricated score.
    assert response.status_code == 503
    assert response.json()["code"] in {"SERVICE_UNAVAILABLE", "SCORING_UNAVAILABLE"}
    assert elapsed < 3.0


def test_auth_still_enforced_when_dependencies_down(integration_settings):
    with _client(integration_settings, velocity=_redis_down) as client:
        response = client.post(SCORE, json=now_request())
    assert response.status_code == 401


def test_slow_tenant_lookup_counts_against_the_deadline(
    integration_settings, auth_header, owner_conn, monkeypatch
):
    """Review finding #3: authentication is inside the deadline, and no claim is taken once it
    has passed."""

    class SlowResolver:
        def __init__(self, inner: Any) -> None:
            self.inner = inner

        async def resolve(self, client_id: str) -> str | None:
            await asyncio.sleep(0.08)
            return await self.inner.resolve(client_id)

    # Isolate the request deadline from the lookup's own limit, which answers 503.
    monkeypatch.setattr("src.api.auth.TENANT_LOOKUP_TIMEOUT_S", 5.0)
    settings = integration_settings.model_copy(update={"server_deadline_s": 0.05})
    body = now_request()
    with _client(settings) as client:
        client.app.state.tenants = SlowResolver(client.app.state.tenants)  # type: ignore[attr-defined]
        response = _post(client, auth_header, body)
    assert response.status_code == 504
    assert response.json()["code"] == "SCORING_DEADLINE_EXCEEDED"
    assert (
        owner_conn.execute(
            "SELECT 1 FROM attempts WHERE payment_attempt_id = %s", (body["payment_attempt_id"],)
        ).fetchone()
        is None
    )
