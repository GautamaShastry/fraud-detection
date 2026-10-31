"""T-201..T-204: durable outcomes, worker crash/retry, projections, archive and retention."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

from src.api.app import create_app
from src.events import worker
from src.events.archive import FileArchiveStore, export_batch, purge_expired, verify_object
from src.events.reconcile import reconcile
from src.storage.db import create_pool
from tests.integration.conftest import now_request
from tests.support.contract import assert_response_matches

pytestmark = pytest.mark.integration

OUTCOMES = "/api/v1/fraud/outcomes"


@pytest.fixture
def outcome_headers(make_token) -> dict[str, str]:
    return {"Authorization": f"Bearer {make_token(scope='fraud:score fraud:outcomes')}"}


def _event(attempt: str, **overrides: Any) -> dict[str, Any]:
    body = {
        "event_id": f"evt-{uuid.uuid4().hex[:12]}",
        "payment_attempt_id": attempt,
        "event_type": "AUTHORIZATION_APPROVED",
        "source": "PROCESSOR",
        "occurred_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    body.update(overrides)
    return body


@pytest.fixture
async def worker_pool(database):
    pool = create_pool(database["worker"], min_size=1, max_size=4)
    await pool.open()
    yield pool
    await pool.close()


@pytest.fixture(autouse=True)
def _clean_events(owner_conn):
    for table in (
        "archive_ledger",
        "archive_batches",
        "outbox",
        "attempt_outcomes",
        "outcome_events",
    ):
        owner_conn.execute(f"DELETE FROM {table}")  # noqa: S608 - fixed table names


def test_outcome_accepted_duplicate_and_conflict(live_client, outcome_headers, owner_conn):
    event = _event("pa-1")
    first = live_client.post(OUTCOMES, json=event, headers=outcome_headers)
    assert first.status_code == 202 and first.json()["status"] == "ACCEPTED"
    assert_response_matches("POST", OUTCOMES, first)
    again = live_client.post(OUTCOMES, json=event, headers=outcome_headers)
    assert again.status_code == 202 and again.json()["status"] == "DUPLICATE"
    altered = live_client.post(
        OUTCOMES, json={**event, "source": "ISSUER"}, headers=outcome_headers
    )
    assert altered.status_code == 409 and altered.json()["code"] == "EVENT_ID_CONFLICT"
    assert_response_matches("POST", OUTCOMES, altered)
    count = owner_conn.execute(
        "SELECT count(*) FROM outcome_events WHERE event_id = %s", (event["event_id"],)
    ).fetchone()
    assert count == (1,)


def test_outcomes_require_their_own_scope(live_client, auth_header):
    response = live_client.post(OUTCOMES, json=_event("pa-1"), headers=auth_header)
    assert response.status_code == 403


def test_outcome_not_acknowledged_when_db_down(integration_settings, outcome_headers):
    settings = integration_settings.model_copy(
        update={
            "tenant_source": "file",
            "postgres_dsn": "postgresql://x:y@127.0.0.1:1/none?connect_timeout=1",
        }
    )
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        response = client.post(OUTCOMES, json=_event("pa-1"), headers=outcome_headers)
    assert response.status_code == 503
    assert response.json()["code"] == "SERVICE_UNAVAILABLE"


async def test_worker_projects_labels_with_precedence(
    live_client, outcome_headers, worker_pool, owner_conn
):
    t = datetime.now(UTC)

    def at(minutes: int) -> str:
        return (t - timedelta(minutes=60 - minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")

    events = [
        _event("pa-L", event_type="AUTHORIZATION_APPROVED", source="PROCESSOR", occurred_at=at(1)),
        _event("pa-L", event_type="CONFIRMED_LEGITIMATE", source="MARKETPLACE", occurred_at=at(2)),
        _event("pa-L", event_type="CONFIRMED_FRAUD", source="ISSUER", occurred_at=at(3)),
        # A later, less authoritative claim must not override the issuer's confirmation.
        _event("pa-L", event_type="CONFIRMED_LEGITIMATE", source="MARKETPLACE", occurred_at=at(4)),
    ]
    for e in reversed(events):  # delivered out of order
        assert live_client.post(OUTCOMES, json=e, headers=outcome_headers).status_code == 202

    cfg = worker.WorkerConfig()
    assert await worker.process_inbox_once(worker_pool, cfg) == 4
    row = owner_conn.execute(
        "SELECT authorization_status, label, label_source, event_count FROM attempt_outcomes "
        "WHERE payment_attempt_id = 'pa-L'"
    ).fetchone()
    assert row == ("APPROVED", "CONFIRMED_FRAUD", "ISSUER", 4)

    retraction = _event(
        "pa-L", event_type="LABEL_RETRACTED", source="MANUAL_REVIEW", occurred_at=at(5)
    )
    live_client.post(OUTCOMES, json=retraction, headers=outcome_headers)
    await worker.process_inbox_once(worker_pool, cfg)
    label = owner_conn.execute(
        "SELECT label FROM attempt_outcomes WHERE payment_attempt_id = 'pa-L'"
    ).fetchone()
    assert label == (None,)


async def test_crash_after_claim_is_recovered_without_double_apply(
    live_client, outcome_headers, worker_pool, owner_conn
):
    live_client.post(OUTCOMES, json=_event("pa-C"), headers=outcome_headers)
    now = datetime.now(UTC)
    clock_now = [now]
    cfg = worker.WorkerConfig(clock=lambda: clock_now[0])

    claimed = await worker.claim_events(worker_pool, cfg)  # worker "crashes" right here
    assert len(claimed) == 1
    assert await worker.claim_events(worker_pool, cfg) == []  # lease still held

    clock_now[0] = now + cfg.lease + timedelta(seconds=1)  # lease expires
    assert await worker.process_inbox_once(worker_pool, cfg) == 1

    # The crashed worker comes back and tries to finish with its stale lease: no effect.
    assert await worker.process_event(worker_pool, claimed[0], cfg) is False
    row = owner_conn.execute(
        "SELECT status, delivery_attempts FROM outcome_events WHERE payment_attempt_id = 'pa-C'"
    ).fetchone()
    assert row == ("PROCESSED", 2)
    outbox = owner_conn.execute(
        "SELECT count(*) FROM outbox WHERE topic = 'outcome.processed'"
    ).fetchone()
    assert outbox == (1,)


async def test_poison_event_is_dead_lettered(
    live_client, outcome_headers, worker_pool, owner_conn, monkeypatch
):
    live_client.post(OUTCOMES, json=_event("pa-P"), headers=outcome_headers)

    async def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("projection bug")

    monkeypatch.setattr(worker, "recompute_projection", broken)
    now = [datetime.now(UTC)]
    cfg = worker.WorkerConfig(clock=lambda: now[0])
    for _ in range(worker.MAX_DELIVERY_ATTEMPTS):
        await worker.process_inbox_once(worker_pool, cfg)
        now[0] += timedelta(minutes=10)  # past any backoff
    row = owner_conn.execute(
        "SELECT status, last_error FROM outcome_events WHERE payment_attempt_id = 'pa-P'"
    ).fetchone()
    assert row == ("DEAD", "RuntimeError")
    # A failed attempt never left a half-applied projection behind.
    assert owner_conn.execute(
        "SELECT count(*) FROM attempt_outcomes WHERE payment_attempt_id = 'pa-P'"
    ).fetchone() == (0,)


class FlakyStore(FileArchiveStore):
    def __init__(self, root: Path, fail_on: str) -> None:
        super().__init__(root)
        self.fail_on = fail_on

    def put(self, key: str, data: bytes) -> None:
        if self.fail_on and key.endswith(self.fail_on):
            raise OSError("upload interrupted")
        super().put(key, data)


async def test_archive_export_verify_and_recover_from_interrupted_upload(
    live_client, auth_header, worker_pool, owner_conn, tmp_path, database
):
    for _ in range(3):
        assert (
            live_client.post(
                "/api/v1/fraud/score", json=now_request(), headers=auth_header
            ).status_code
            == 200
        )
    now = datetime.now(UTC)

    root = tmp_path / "archive"
    flaky = FlakyStore(root, fail_on=".manifest.json")
    with pytest.raises(OSError):
        await export_batch(worker_pool, flaky, now)
    # Nothing was committed: rows remain unpublished and no batch or ledger rows exist.
    assert owner_conn.execute(
        "SELECT count(*) FROM outbox WHERE published_at IS NULL"
    ).fetchone() == (3,)
    assert owner_conn.execute("SELECT count(*) FROM archive_ledger").fetchone() == (0,)

    store = FileArchiveStore(root)
    result = await export_batch(worker_pool, store, now)
    assert result.rows == 3
    assert (await export_batch(worker_pool, store, now)).rows == 0  # idempotent re-run

    manifests = [k for k in store.list() if k.endswith(".manifest.json")]
    assert len(manifests) == 1
    assert verify_object(store, manifests[0]) == {
        "batch_id": result.batch_ids[0],
        "sha256_ok": True,
        "row_count_ok": True,
    }
    manifest = json.loads(store.get(manifests[0]))
    assert manifest["row_count"] == 3

    with psycopg.connect(database["owner"], row_factory=dict_row) as conn:
        report = reconcile(conn, store)
    assert report["ok"], report["problems"]
    assert report["archive_orphan_objects"] == []
    assert report["outbox_unpublished"] == 0


async def test_retry_after_new_rows_and_a_new_day_reuses_the_same_objects(
    live_client, auth_header, worker_pool, tmp_path, database
):
    """Review finding #6: a retried export must not strand the first attempt's objects."""

    def score() -> None:
        response = live_client.post("/api/v1/fraud/score", json=now_request(), headers=auth_header)
        assert response.status_code == 200

    for _ in range(3):
        score()
    now = datetime.now(UTC)
    root = tmp_path / "archive"
    with pytest.raises(OSError):
        await export_batch(worker_pool, FlakyStore(root, fail_on=".manifest.json"), now)
    store = FileArchiveStore(root)
    stranded = store.list()
    assert len(stranded) == 1  # the parquet landed; the crash was before the manifest

    score()  # a new row arrives before the retry, and the retry runs on a later day

    first = await export_batch(worker_pool, store, now + timedelta(days=2))
    assert first.rows == 3  # exactly the original rows
    assert set(stranded) <= set(store.list())
    assert len(store.list()) == 2  # same parquet key, plus its manifest

    second = await export_batch(worker_pool, store, now + timedelta(days=2))
    assert second.rows == 1
    assert (await export_batch(worker_pool, store, now + timedelta(days=2))).rows == 0
    assert len(store.list()) == 4

    with psycopg.connect(database["owner"], row_factory=dict_row) as conn:
        report = reconcile(conn, store)
    assert report["archive_orphan_objects"] == [] and report["ok"], report["problems"]


async def test_reconciliation_fails_on_orphans_but_not_on_an_export_in_flight(
    live_client, auth_header, worker_pool, tmp_path, database
):
    assert (
        live_client.post("/api/v1/fraud/score", json=now_request(), headers=auth_header).status_code
        == 200
    )
    root = tmp_path / "archive"
    with pytest.raises(OSError):
        await export_batch(
            worker_pool, FlakyStore(root, fail_on=".manifest.json"), datetime.now(UTC)
        )
    store = FileArchiveStore(root)
    with psycopg.connect(database["owner"], row_factory=dict_row) as conn:
        in_flight = reconcile(conn, store)
    assert in_flight["archive_orphan_objects"] == []  # will be finished by the next run

    await export_batch(worker_pool, store, datetime.now(UTC))
    store.put("assessments/tenant=x/date=2026-01-01/stray.parquet", b"stray")
    with psycopg.connect(database["owner"], row_factory=dict_row) as conn:
        report = reconcile(conn, store)
    assert report["archive_orphan_objects"] == [
        "assessments/tenant=x/date=2026-01-01/stray.parquet"
    ]
    assert not report["ok"]
    assert "archive contains objects with no batch" in report["problems"]


async def test_retention_purges_only_archived_rows(
    live_client, auth_header, worker_pool, owner_conn, tmp_path
):
    archived, pending = now_request(), now_request()
    for body in (archived, pending):
        assert (
            live_client.post("/api/v1/fraud/score", json=body, headers=auth_header).status_code
            == 200
        )
    # Export only the first one.
    owner_conn.execute(
        "UPDATE outbox SET published_at = now() WHERE payload->>'payment_attempt_id' = %s",
        (pending["payment_attempt_id"],),
    )
    await export_batch(worker_pool, FileArchiveStore(tmp_path), datetime.now(UTC))
    owner_conn.execute(
        "UPDATE outbox SET published_at = NULL WHERE payload->>'payment_attempt_id' = %s",
        (pending["payment_attempt_id"],),
    )

    future = datetime.now(UTC) + timedelta(days=31)
    purged = await purge_expired(worker_pool, future, timedelta(days=30))
    assert purged["assessments"] == 1 and purged["attempts"] == 1
    remaining = {
        r[0]
        for r in owner_conn.execute(
            "SELECT payment_attempt_id FROM attempts WHERE payment_attempt_id = ANY(%s)",
            ([archived["payment_attempt_id"], pending["payment_attempt_id"]],),
        )
    }
    assert remaining == {pending["payment_attempt_id"]}  # never archived => never purged
