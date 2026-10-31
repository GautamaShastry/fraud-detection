"""Durable outcome inbox and the label projection (FR-08, T-201, T-202; I-09 provisional).

Label precedence (I-09 provisional, documented in docs/DECISIONS.md):

* Only CONFIRMED_FRAUD / CONFIRMED_LEGITIMATE events set a label. An approved authorization
  is never treated as evidence of legitimacy.
* A label from a more authoritative source replaces one from a less authoritative source
  (MANUAL_REVIEW > ISSUER > PROCESSOR > MARKETPLACE). Between equal sources the later
  occurred_at wins (ties broken by received_at, then event_id).
* LABEL_RETRACTED clears the current label if its source is at least as authoritative.

The projection is recomputed by folding all processed events of the attempt in a fixed order,
so it is idempotent and independent of delivery order and duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from src.storage.db import Conn, Pool, tenant_tx

SOURCE_RANK = {"MARKETPLACE": 0, "PROCESSOR": 1, "ISSUER": 2, "MANUAL_REVIEW": 3}


class InsertResult(StrEnum):
    ACCEPTED = "ACCEPTED"
    DUPLICATE = "DUPLICATE"
    CONFLICT = "CONFLICT"


async def insert_event(
    pool: Pool,
    tenant_id: str,
    event: dict[str, Any],
    payload_hash: bytes,
    received_at: datetime,
) -> tuple[InsertResult, datetime]:
    """Insert into the inbox. Returns once the row is committed (durable acceptance)."""
    async with tenant_tx(pool, tenant_id) as conn:
        cur = await conn.execute(
            "INSERT INTO outcome_events (tenant_id, event_id, payload_hash, payment_attempt_id, "
            "transaction_id, event_type, source, occurred_at, received_at, reason_code) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT DO NOTHING RETURNING received_at",
            (
                tenant_id,
                event["event_id"],
                payload_hash,
                event["payment_attempt_id"],
                event.get("transaction_id"),
                event["event_type"],
                event["source"],
                event["occurred_at"],
                received_at,
                event.get("reason_code"),
            ),
        )
        row = await cur.fetchone()
        if row:
            return InsertResult.ACCEPTED, row["received_at"]
        cur = await conn.execute(
            "SELECT payload_hash, received_at FROM outcome_events WHERE event_id = %s",
            (event["event_id"],),
        )
        existing = await cur.fetchone()
        assert existing is not None
        if bytes(existing["payload_hash"]) != payload_hash:
            return InsertResult.CONFLICT, existing["received_at"]
        return InsertResult.DUPLICATE, existing["received_at"]


@dataclass
class Projection:
    authorization_status: str | None = None
    authorization_occurred_at: datetime | None = None
    chargeback_reported_at: datetime | None = None
    label: str | None = None
    label_source: str | None = None
    label_occurred_at: datetime | None = None
    label_event_id: str | None = None
    event_count: int = 0


def fold(events: list[dict[str, Any]]) -> Projection:
    """Deterministic projection of an attempt's outcome events (any order in, same result)."""
    p = Projection()
    ordered = sorted(events, key=lambda e: (e["occurred_at"], e["received_at"], e["event_id"]))
    for e in ordered:
        p.event_count += 1
        kind, source, at = e["event_type"], e["source"], e["occurred_at"]
        if kind in ("AUTHORIZATION_APPROVED", "AUTHORIZATION_DECLINED"):
            p.authorization_status = "APPROVED" if kind == "AUTHORIZATION_APPROVED" else "DECLINED"
            p.authorization_occurred_at = at
        elif kind == "CHARGEBACK_REPORTED":
            p.chargeback_reported_at = p.chargeback_reported_at or at
        elif kind in ("CONFIRMED_FRAUD", "CONFIRMED_LEGITIMATE"):
            if (
                p.label is None
                or SOURCE_RANK[source] >= SOURCE_RANK[p.label_source or "MARKETPLACE"]
            ):
                p.label, p.label_source = kind, source
                p.label_occurred_at, p.label_event_id = at, e["event_id"]
        elif kind == "LABEL_RETRACTED":
            if p.label_source is not None and SOURCE_RANK[source] >= SOURCE_RANK[p.label_source]:
                p.label = p.label_source = p.label_occurred_at = p.label_event_id = None
    return p


async def recompute_projection(
    conn: Conn, tenant_id: str, payment_attempt_id: str, now: datetime
) -> Projection:
    cur = await conn.execute(
        "SELECT event_id, event_type, source, occurred_at, received_at FROM outcome_events "
        "WHERE tenant_id = %s AND payment_attempt_id = %s AND status = 'PROCESSED'",
        (tenant_id, payment_attempt_id),
    )
    projection = fold(list(await cur.fetchall()))
    await conn.execute(
        "INSERT INTO attempt_outcomes (tenant_id, payment_attempt_id, authorization_status, "
        "authorization_occurred_at, chargeback_reported_at, label, label_source, "
        "label_occurred_at, label_event_id, event_count, updated_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (tenant_id, payment_attempt_id) DO UPDATE SET "
        "authorization_status = EXCLUDED.authorization_status, "
        "authorization_occurred_at = EXCLUDED.authorization_occurred_at, "
        "chargeback_reported_at = EXCLUDED.chargeback_reported_at, label = EXCLUDED.label, "
        "label_source = EXCLUDED.label_source, label_occurred_at = EXCLUDED.label_occurred_at, "
        "label_event_id = EXCLUDED.label_event_id, event_count = EXCLUDED.event_count, "
        "updated_at = EXCLUDED.updated_at",
        (
            tenant_id,
            payment_attempt_id,
            projection.authorization_status,
            projection.authorization_occurred_at,
            projection.chargeback_reported_at,
            projection.label,
            projection.label_source,
            projection.label_occurred_at,
            projection.label_event_id,
            projection.event_count,
            now,
        ),
    )
    return projection
