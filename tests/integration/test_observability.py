"""T-106/T-404: traces link the request path, logs/metrics/traces carry no raw identifiers, and
faults move the right indicators."""

from __future__ import annotations

import io
import json
import logging
import re
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.api.app import create_app
from src.telemetry.setup import JsonFormatter
from tests.integration.conftest import now_request
from tests.integration.test_scoring_flow import BrokenScorer, _bundle, _redis_down

pytestmark = pytest.mark.integration

EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
trace.set_tracer_provider(_provider)


def _metric(client: TestClient, name: str, **labels: str) -> float:
    text = client.get("/metrics").text
    want = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    total = 0.0
    for line in text.splitlines():
        m = re.match(rf"^{name}(?:\{{(.*)\}})? (\S+)$", line)
        if m and all(f'{k}="{v}"' in (m.group(1) or "") for k, v in labels.items()):
            total += float(m.group(2))
    assert want is not None
    return total


def test_trace_links_full_request_without_identifiers(live_client, auth_header):
    EXPORTER.clear()
    body = now_request(customer_id="cust-SECRET-1", ip_address="203.0.113.50")
    assert (
        live_client.post("/api/v1/fraud/score", json=body, headers=auth_header).status_code == 200
    )
    spans = EXPORTER.get_finished_spans()
    names = {s.name for s in spans}
    assert {"fraud.score", "postgres.claim_attempt", "scoring.assess", "postgres.commit"} <= names
    root = next(s for s in spans if s.name == "fraud.score")
    assert all(
        s.context.trace_id == root.context.trace_id
        for s in spans
        if s.name in {"postgres.claim_attempt", "scoring.assess", "postgres.commit"}
    )
    dumped = json.dumps([{"name": s.name, "attrs": dict(s.attributes or {})} for s in spans])
    for raw in (
        body["device_id"],
        body["payment_method_token"],
        "cust-SECRET-1",
        "203.0.113.50",
        body["payment_attempt_id"],
    ):
        assert raw not in dumped


def test_logs_never_contain_identifiers(integration_settings, auth_header, redis_client):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    try:
        with TestClient(
            create_app(integration_settings, velocity=_redis_down), raise_server_exceptions=False
        ) as client:
            root.addHandler(handler)  # after startup, which installs the app's own handler
            body = now_request(customer_id="cust-SECRET-2", ip_address="203.0.113.51")
            client.post("/api/v1/fraud/score", json=body, headers=auth_header)
            client.post(
                "/api/v1/fraud/score",
                json={**body, "card_number": "4111111111111111"},
                headers=auth_header,
            )
            logging.getLogger("probe").warning("probe", extra={"device_id": body["device_id"]})
    finally:
        root.removeHandler(handler)
    output = stream.getvalue()
    assert "Primary scoring failed" in output  # the degradation itself is visible
    for raw in (
        body["device_id"],
        body["payment_method_token"],
        "cust-SECRET-2",
        "203.0.113.51",
        "4111111111111111",
    ):
        assert raw not in output


def test_metrics_have_no_identifier_labels(live_client, auth_header):
    body = now_request()
    live_client.post("/api/v1/fraud/score", json=body, headers=auth_header)
    text = live_client.get("/metrics").text
    assert "fraud_score_requests_total" in text
    for raw in (body["payment_attempt_id"], body["device_id"], "tenant-a"):
        assert raw not in text


def test_fault_indicators(integration_settings, auth_header, make_token, redis_client):
    with TestClient(
        create_app(integration_settings, velocity=_redis_down), raise_server_exceptions=False
    ) as client:
        before = _metric(client, "fraud_primary_failures_total", reason="ConnectionError")
        fallback_before = _metric(client, "fraud_score_requests_total", outcome="fallback")
        client.post("/api/v1/fraud/score", json=now_request(), headers=auth_header)
        assert (
            _metric(client, "fraud_primary_failures_total", reason="ConnectionError") == before + 1
        )
        assert (
            _metric(client, "fraud_score_requests_total", outcome="fallback") == fallback_before + 1
        )

    broken = BrokenScorer("broken-0", ())
    with TestClient(
        create_app(integration_settings, bundle=_bundle(primary=broken, fallback=broken)),
        raise_server_exceptions=False,
    ) as client:
        before = _metric(client, "fraud_score_requests_total", outcome="unavailable")
        client.post("/api/v1/fraud/score", json=now_request(), headers=auth_header)
        assert _metric(client, "fraud_score_requests_total", outcome="unavailable") == before + 1

        # Security probe: forged and scope-less tokens are counted.
        probes_before = _metric(client, "fraud_auth_failures_total")
        client.post(
            "/api/v1/fraud/score",
            json=now_request(),
            headers={"Authorization": "Bearer forged.token.value"},
        )
        client.post(
            "/api/v1/fraud/score",
            json=now_request(),
            headers={"Authorization": f"Bearer {make_token(scope='fraud:outcomes')}"},
        )
        assert _metric(client, "fraud_auth_failures_total") == probes_before + 2


async def test_worker_backlog_gauge(database, live_client, make_token):
    from src.events import worker
    from src.storage.db import create_pool
    from src.telemetry import metrics

    headers = {"Authorization": f"Bearer {make_token(scope='fraud:outcomes')}"}
    for i in range(3):
        live_client.post(
            "/api/v1/fraud/outcomes",
            headers=headers,
            json={
                "event_id": f"evt-backlog-{i}-{now_request()['payment_attempt_id']}",
                "payment_attempt_id": "pa-backlog",
                "event_type": "AUTHORIZATION_APPROVED",
                "source": "PROCESSOR",
                "occurred_at": now_request()["occurred_at"],
            },
        )
    pool = create_pool(database["worker"], min_size=1, max_size=2)
    await pool.open()
    try:
        await worker.update_backlog(pool)
        assert metrics.INBOX_BACKLOG._value.get() >= 3
        await worker.process_inbox_once(pool, worker.WorkerConfig(batch_size=1000))
        await worker.update_backlog(pool)
        assert metrics.INBOX_BACKLOG._value.get() == 0
    finally:
        await pool.close()


def _unused(*_: Any) -> None:  # keep imports used by type checkers
    return None
