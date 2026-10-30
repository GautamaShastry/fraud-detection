"""Outcome inbox worker, archive exporter and retention job (T-202, T-204; I-02 provisional).

Inbox processing is at-least-once with idempotent effects:

1. claim: lease a batch of PENDING events with FOR UPDATE SKIP LOCKED (several workers can run)
2. process each event in its own transaction, under a per-attempt advisory lock:
   mark PROCESSED (only if this worker still holds the lease), recompute the attempt's
   projection from all processed events, enqueue an outbox row for archival
3. on failure: back off and retry; after MAX_DELIVERY_ATTEMPTS the event goes DEAD
   (dead-letter), visible to reconciliation and dashboards, never silently dropped

A crash at any point leaves either an expired lease (event is re-claimed) or a committed result
(re-processing is a no-op because the projection is a deterministic fold).

    python -m src.events.worker
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from prometheus_client import start_http_server

from src.events.archive import ArchiveStore, export_batch, purge_expired, store_from_uri
from src.storage.attempts import enqueue_outbox
from src.storage.db import Pool, create_pool
from src.storage.outcomes import recompute_projection
from src.telemetry import metrics
from src.telemetry.setup import configure_logging, configure_tracing, tracer

logger = logging.getLogger(__name__)

MAX_DELIVERY_ATTEMPTS = 5


@dataclass
class WorkerConfig:
    batch_size: int = 100
    lease: timedelta = timedelta(seconds=30)
    idle_sleep_s: float = 0.5
    retention: timedelta = timedelta(days=30)  # PRD section 8: ~30 days operational
    purge_every: timedelta = timedelta(hours=1)
    archive_batch: int = 200
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))


async def claim_events(pool: Pool, cfg: WorkerConfig) -> list[dict[str, Any]]:
    now = cfg.clock()
    token = uuid.uuid4()
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute(
            "UPDATE outcome_events e SET lease_token = %s, lease_until = %s, "
            "delivery_attempts = e.delivery_attempts + 1 "
            "FROM (SELECT tenant_id, event_id FROM outcome_events WHERE status = 'PENDING' "
            "      AND (lease_until IS NULL OR lease_until < %s) ORDER BY received_at "
            "      LIMIT %s FOR UPDATE SKIP LOCKED) picked "
            "WHERE e.tenant_id = picked.tenant_id AND e.event_id = picked.event_id "
            "RETURNING e.tenant_id, e.event_id, e.payment_attempt_id, e.transaction_id, "
            "e.event_type, e.source, e.occurred_at, e.received_at, e.reason_code, "
            "e.lease_token, e.delivery_attempts",
            (token, now + cfg.lease, now, cfg.batch_size),
        )
        return list(await cur.fetchall())


async def process_event(pool: Pool, event: dict[str, Any], cfg: WorkerConfig) -> bool:
    """Apply one claimed event. Returns False if the lease was lost (another worker owns it)."""
    now = cfg.clock()
    async with pool.connection() as conn, conn.transaction():
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"{event['tenant_id']}\x1f{event['payment_attempt_id']}",),
        )
        cur = await conn.execute(
            "UPDATE outcome_events SET status = 'PROCESSED', processed_at = %s, "
            "lease_until = NULL, last_error = NULL "
            "WHERE tenant_id = %s AND event_id = %s AND lease_token = %s AND status = 'PENDING' "
            "RETURNING 1",
            (now, event["tenant_id"], event["event_id"], event["lease_token"]),
        )
        if await cur.fetchone() is None:
            return False
        await recompute_projection(conn, event["tenant_id"], event["payment_attempt_id"], now)
        await enqueue_outbox(
            conn, event["tenant_id"], "outcome.processed", {"event_id": event["event_id"]}, now
        )
    return True


async def record_failure(pool: Pool, event: dict[str, Any], error: str, cfg: WorkerConfig) -> str:
    now = cfg.clock()
    dead = event["delivery_attempts"] >= MAX_DELIVERY_ATTEMPTS
    backoff = timedelta(seconds=min(2 ** event["delivery_attempts"], 300))
    async with pool.connection() as conn, conn.transaction():
        await conn.execute(
            "UPDATE outcome_events SET status = %s, last_error = %s, lease_until = %s "
            "WHERE tenant_id = %s AND event_id = %s AND lease_token = %s",
            (
                "DEAD" if dead else "PENDING",
                error[:500],
                None if dead else now + backoff,
                event["tenant_id"],
                event["event_id"],
                event["lease_token"],
            ),
        )
    return "dead" if dead else "retry"


async def process_inbox_once(pool: Pool, cfg: WorkerConfig) -> int:
    events = await claim_events(pool, cfg)
    for event in events:
        with tracer.start_as_current_span("worker.process_event"):
            try:
                applied = await process_event(pool, event, cfg)
                result = "processed" if applied else "lease_lost"
            except Exception as exc:  # noqa: BLE001 - failures are retried / dead-lettered
                logger.warning("Event processing failed (%s)", type(exc).__name__)
                result = await record_failure(pool, event, type(exc).__name__, cfg)
        metrics.WORKER_EVENTS.labels(result=result).inc()
    return len(events)


async def update_backlog(pool: Pool) -> None:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT (SELECT count(*) FROM outcome_events WHERE status = 'PENDING') AS inbox, "
            "(SELECT count(*) FROM outbox WHERE published_at IS NULL) AS outbox"
        )
        row = await cur.fetchone()
        await conn.rollback()
    if row:
        metrics.INBOX_BACKLOG.set(row["inbox"])
        metrics.OUTBOX_BACKLOG.set(row["outbox"])


async def run(pool: Pool, store: ArchiveStore, cfg: WorkerConfig, stop: asyncio.Event) -> None:
    last_purge = datetime.min.replace(tzinfo=UTC)
    while not stop.is_set():
        busy = False
        try:
            busy |= await process_inbox_once(pool, cfg) > 0
            exported = await export_batch(pool, store, cfg.clock(), cfg.archive_batch)
            busy |= exported.rows > 0
            if cfg.clock() - last_purge >= cfg.purge_every:
                purged = await purge_expired(pool, cfg.clock(), cfg.retention)
                logger.info("Retention purge %s", json.dumps(purged))
                last_purge = cfg.clock()
            await update_backlog(pool)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive through outages
            logger.error("Worker iteration failed (%s)", type(exc).__name__, exc_info=True)
            busy = False
        if not busy:
            try:
                await asyncio.wait_for(stop.wait(), cfg.idle_sleep_s)
            except TimeoutError:
                pass


async def main_async() -> None:
    configure_logging(os.environ.get("FRAUD_LOG_LEVEL", "INFO"))
    configure_tracing("fraud-worker")
    start_http_server(int(os.environ.get("FRAUD_METRICS_PORT", "9100")))
    pool = create_pool(os.environ["FRAUD_WORKER_POSTGRES_DSN"], min_size=1, max_size=4)
    await pool.open(wait=False)
    store = store_from_uri(os.environ["FRAUD_ARCHIVE_URI"])
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows
            pass
    logger.info("Worker started")
    try:
        await run(pool, store, WorkerConfig(), stop)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main_async())
