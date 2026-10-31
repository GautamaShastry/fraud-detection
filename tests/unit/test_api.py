"""Behavior of the Phase 0 API skeleton: health, validation, clock skew, explicit unavailability."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from tests.conftest import FIXED_NOW, valid_score_request
from tests.support.contract import assert_response_matches

SCORE = "/api/v1/fraud/score"


def test_live(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert_response_matches("GET", "/health/live", response)


def test_ready_when_dependencies_up(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "checks": {"postgres": "ok", "redis": "ok"}}
    assert_response_matches("GET", "/health/ready", response)


async def _ok():
    return None


async def _down():
    raise ConnectionError("dependency down")


def test_not_ready_when_postgres_fails(app_settings):
    app = create_app(
        app_settings,
        readiness_checks={"postgres": _down},
        advisory_checks={"redis": _ok},
        connect=False,
    )
    with TestClient(app) as c:
        live, ready = c.get("/health/live"), c.get("/health/ready")
    assert live.status_code == 200  # liveness does not depend on dependencies
    assert ready.status_code == 503
    assert ready.json() == {
        "status": "unavailable",
        "checks": {"postgres": "unavailable", "redis": "ok"},
    }
    assert_response_matches("GET", "/health/ready", ready)


def test_redis_outage_degrades_but_keeps_the_instance_in_service(app_settings):
    """Review finding #2: scoring continues on the fallback, so Redis must not fail readiness."""
    app = create_app(
        app_settings,
        readiness_checks={"postgres": _ok},
        advisory_checks={"redis": _down},
        connect=False,
    )
    with TestClient(app) as c:
        ready = c.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "degraded",
        "checks": {"postgres": "ok", "redis": "unavailable"},
    }
    assert_response_matches("GET", "/health/ready", ready)


def test_hung_redis_check_cannot_fail_readiness(app_settings):
    import asyncio

    async def hang():
        await asyncio.sleep(10)

    settings = app_settings.model_copy(update={"readiness_timeout_s": 0.05})
    app = create_app(
        settings,
        readiness_checks={"postgres": _ok},
        advisory_checks={"redis": hang},
        connect=False,
    )
    with TestClient(app) as c:
        ready = c.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "degraded"


def test_default_checks_make_only_postgres_required(app_settings):
    app = create_app(app_settings, connect=False)
    assert set(app.state.readiness_checks) == {"postgres"}
    assert set(app.state.advisory_checks) == {"redis"}


def test_not_ready_when_check_hangs(app_settings):
    import asyncio

    async def hang():
        await asyncio.sleep(10)

    settings = app_settings.model_copy(update={"readiness_timeout_s": 0.05})
    app = create_app(settings, readiness_checks={"postgres": hang}, connect=False)
    with TestClient(app) as c:
        assert c.get("/health/ready").status_code == 503


def test_not_ready_with_no_checks(app_settings):
    app = create_app(app_settings, readiness_checks={}, connect=False)
    with TestClient(app) as c:
        assert c.get("/health/ready").status_code == 503


def test_score_returns_explicit_unavailable_not_a_fabricated_score(client, auth_header):
    response = client.post(SCORE, json=valid_score_request(), headers=auth_header)
    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "SCORING_UNAVAILABLE"
    assert body["retryable"] is False
    assert "risk_score" not in body
    assert_response_matches("POST", SCORE, response)


def test_validation_errors_are_problem_json_without_echo(client, auth_header):
    secret = "4111111111111111"
    response = client.post(
        SCORE, json=valid_score_request(card_number=secret, currency="usd"), headers=auth_header
    )
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    assert secret not in response.text
    fields = {e["field"] for e in response.json()["errors"]}
    assert fields == {"card_number", "currency"}
    assert_response_matches("POST", SCORE, response)


def test_invalid_json(client, auth_header):
    response = client.post(
        SCORE, content=b"{not json", headers={**auth_header, "content-type": "application/json"}
    )
    assert response.status_code == 422
    assert_response_matches("POST", SCORE, response)


@pytest.mark.parametrize(
    "occurred_at,ok",
    [
        ("2026-10-28T12:05:30Z", True),  # exactly at the 300 s future-skew bound
        ("2026-10-28T12:05:31Z", False),  # one second beyond it
        ("2026-10-27T12:00:30Z", True),  # exactly 24 h old
        ("2026-10-27T12:00:29Z", False),  # older than 24 h
    ],
)
def test_clock_skew_bounds(client, auth_header, occurred_at, ok):
    assert FIXED_NOW.isoformat() == "2026-10-28T12:00:30+00:00"
    response = client.post(
        SCORE, json=valid_score_request(occurred_at=occurred_at), headers=auth_header
    )
    if ok:
        assert response.status_code == 503
    else:
        assert response.status_code == 422
        assert response.json()["errors"][0]["field"] == "occurred_at"
        assert_response_matches("POST", SCORE, response)


def test_account_created_in_future_rejected(client, auth_header):
    body = valid_score_request(account_created_at="2027-01-01T00:00:00Z")
    response = client.post(SCORE, json=body, headers=auth_header)
    assert response.status_code == 422


def test_oversized_body_declared(client, auth_header):
    body = valid_score_request(device_id="d" * 20_000)
    response = client.post(SCORE, json=body, headers=auth_header)
    assert response.status_code == 413
    assert_response_matches("POST", SCORE, response)


def test_oversized_body_streamed_without_length(client, auth_header):
    def chunks():
        yield b'{"transaction_id": "'
        yield b"x" * 20_000
        yield b'"}'

    response = client.post(
        SCORE, content=chunks(), headers={**auth_header, "content-type": "application/json"}
    )
    assert response.status_code == 413
    assert response.json()["code"] == "PAYLOAD_TOO_LARGE"


def test_request_id_propagated_and_sanitized(client):
    assert (
        client.get("/health/live", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"]
        == "abc-123"
    )
    generated = client.get("/health/live", headers={"X-Request-ID": "bad id\n"}).headers[
        "x-request-id"
    ]
    assert generated != "bad id\n" and len(generated) == 36


def test_unknown_route_is_problem_json(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json()["code"] == "NOT_FOUND"


def test_generated_openapi_not_served(client):
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/docs").status_code == 404
