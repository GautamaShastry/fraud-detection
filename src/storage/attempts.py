"""Attempt claim state machine and immutable assessment persistence (T-102, T-104, I-03).

Lifecycle of one (tenant, payment_attempt_id):

    claim ──► IN_PROGRESS ──complete──► COMPLETED   (assessment row written, immutable)
                  │  ▲
     unavailable  │  │ reclaim (lease expired, or after UNAVAILABLE)
                  ▼  │
              UNAVAILABLE

* A claim is a committed row with a random claim_token and a lease. Only the holder of the
  current token may complete it; a lease that expires (crashed or stalled request) can be
  taken over by a retry.
* Reusing an idempotency key or attempt ID with a different request fingerprint is a conflict.
* An identical retry of a COMPLETED attempt returns the stored assessment unchanged.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from src.domain.models import (
    Assessment,
    Attempt,
    AttemptStatus,
    Recommendation,
    RiskSignal,
    ScoringMode,
)
from src.storage.db import Conn, Pool, tenant_tx


class ClaimOutcome(StrEnum):
    CLAIMED = "CLAIMED"
    COMPLETED = "COMPLETED"
    IN_PROGRESS = "IN_PROGRESS"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class Claim:
    outcome: ClaimOutcome
    claim_token: uuid.UUID | None = None
    assessment: Assessment | None = None
    retry_after_ms: int | None = None


class LostClaimError(Exception):
    """The lease expired and another request took over the attempt."""


_CLAIM_SQL = (
    "SELECT * FROM fraud_claim_attempt(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
    "%s, %s, %s, %s)"
)
_COMPLETE_SQL = (
    "SELECT * FROM fraud_complete_attempt(%s, %s, %s, %s, %s, %s, %s::smallint, %s, %s, "
    "%s::jsonb, %s::jsonb, %s, %s, %s, %s::real, %s::jsonb)"
)


async def claim_attempt(pool: Pool, attempt: Attempt, now: datetime, lease: timedelta) -> Claim:
    """One autocommit round trip to fraud_claim_attempt (migration 0005)."""
    token = uuid.uuid4()
    a = attempt
    async with pool.connection() as conn:
        cur = await conn.execute(
            _CLAIM_SQL,
            (
                a.tenant_id,
                a.payment_attempt_id,
                a.idempotency_key,
                a.transaction_id,
                a.fingerprint,
                token,
                now,
                now + lease,
                a.amount_minor,
                a.currency,
                a.occurred_at,
                a.received_at,
                a.customer_ref,
                a.device_ref,
                a.payment_method_ref,
                a.ip_ref,
                a.shipping_ref,
                a.account_created_at,
            ),
        )
        row = await cur.fetchone()
    if row is None:
        raise RuntimeError("COMPLETED attempt without assessment")
    outcome = row["outcome"]
    if outcome == "CLAIMED":
        return Claim(ClaimOutcome.CLAIMED, claim_token=token)
    if outcome == "CONFLICT":
        return Claim(ClaimOutcome.CONFLICT)
    if outcome == "COMPLETED":
        return Claim(ClaimOutcome.COMPLETED, assessment=assessment_from_json(row["assessment"]))
    if outcome == "IN_PROGRESS":
        remaining = (row["lease_expires_at"] - now) / timedelta(milliseconds=1)
        return Claim(ClaimOutcome.IN_PROGRESS, retry_after_ms=int(min(max(remaining, 10), 500)))
    # RETRY: the row vanished between insert and lookup (retention purge).
    return Claim(ClaimOutcome.IN_PROGRESS, retry_after_ms=50)


async def complete_attempt(
    pool: Pool, assessment: Assessment, claim_token: uuid.UUID, now: datetime
) -> Assessment:
    """Commit the immutable assessment and its outbox row in one round trip.

    Nothing is written unless this request still holds the claim. Returns the assessment that
    is now authoritative (this one, or one a concurrent request committed first).
    """
    s = assessment
    async with pool.connection() as conn:
        cur = await conn.execute(
            _COMPLETE_SQL,
            (
                s.tenant_id,
                s.payment_attempt_id,
                claim_token,
                now,
                s.assessment_id,
                s.transaction_id,
                s.risk_score,
                s.recommendation.value,
                s.scoring_mode.value,
                json.dumps([sig.to_json() for sig in s.risk_signals]),
                json.dumps(s.features),
                s.model_version,
                s.policy_version,
                s.evaluated_at,
                s.scoring_latency_ms,
                json.dumps(
                    {
                        "assessment_id": str(s.assessment_id),
                        "payment_attempt_id": s.payment_attempt_id,
                    }
                ),
            ),
        )
        row = await cur.fetchone()
    if row is not None and row["outcome"] == "RECORDED":
        return assessment
    if row is not None and row["outcome"] == "EXISTING":
        return assessment_from_json(row["assessment"])
    raise LostClaimError(assessment.payment_attempt_id)


def assessment_from_json(d: dict[str, Any]) -> Assessment:
    """Assessment from to_jsonb(assessments row)."""
    return assessment_from_row(
        {
            **d,
            "assessment_id": uuid.UUID(d["assessment_id"]),
            "evaluated_at": datetime.fromisoformat(d["evaluated_at"]),
        }
    )


async def enqueue_outbox(
    conn: Conn, tenant_id: str, topic: str, payload: dict[str, Any], now: datetime
) -> None:
    """Transactional outbox: written in the same transaction as the state change."""
    await conn.execute(
        "INSERT INTO outbox (tenant_id, topic, payload, created_at) VALUES (%s, %s, %s, %s)",
        (tenant_id, topic, json.dumps(payload), now),
    )


async def mark_unavailable(
    pool: Pool, attempt: Attempt, claim_token: uuid.UUID, now: datetime
) -> None:
    async with tenant_tx(pool, attempt.tenant_id) as conn:
        await conn.execute(
            "UPDATE attempts SET status = 'UNAVAILABLE', updated_at = %s "
            "WHERE payment_attempt_id = %s AND claim_token = %s AND status = 'IN_PROGRESS'",
            (now, attempt.payment_attempt_id, claim_token),
        )


@dataclass(frozen=True)
class AttemptState:
    status: AttemptStatus
    lease_expires_at: datetime
    assessment: Assessment | None


async def get_attempt_state(
    pool: Pool, tenant_id: str, payment_attempt_id: str
) -> AttemptState | None:
    async with tenant_tx(pool, tenant_id) as conn:
        cur = await conn.execute(
            "SELECT status, lease_expires_at FROM attempts WHERE payment_attempt_id = %s",
            (payment_attempt_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return AttemptState(
            AttemptStatus(row["status"]),
            row["lease_expires_at"],
            await _fetch_assessment(conn, payment_attempt_id),
        )


async def _fetch_assessment(conn: Conn, payment_attempt_id: str) -> Assessment | None:
    cur = await conn.execute(
        "SELECT * FROM assessments WHERE payment_attempt_id = %s", (payment_attempt_id,)
    )
    row = await cur.fetchone()
    return assessment_from_row(row) if row else None


def assessment_from_row(row: dict[str, Any]) -> Assessment:
    return Assessment(
        assessment_id=row["assessment_id"],
        tenant_id=row["tenant_id"],
        payment_attempt_id=row["payment_attempt_id"],
        transaction_id=row["transaction_id"],
        risk_score=row["risk_score"],
        recommendation=Recommendation(row["recommendation"]),
        risk_signals=tuple(RiskSignal(**s) for s in row["risk_signals"]),
        scoring_mode=ScoringMode(row["scoring_mode"]),
        model_version=row["model_version"],
        policy_version=row["policy_version"],
        evaluated_at=row["evaluated_at"],
        scoring_latency_ms=row["scoring_latency_ms"],
        features=row["features"],
    )


class PostgresTenantResolver:
    """Trusted client -> tenant mapping from api_clients (T-101), cached briefly."""

    def __init__(self, pool: Pool, ttl_s: float = 30.0) -> None:
        self._pool = pool
        self._ttl_s = ttl_s
        self._cache: dict[str, tuple[float, str | None]] = {}

    async def resolve(self, client_id: str) -> str | None:
        import time

        now = time.monotonic()
        hit = self._cache.get(client_id)
        if hit and now - hit[0] < self._ttl_s:
            return hit[1]
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT tenant_id FROM api_clients WHERE client_id = %s AND active", (client_id,)
            )
            row = await cur.fetchone()
            await conn.rollback()
        tenant = row["tenant_id"] if row else None
        self._cache[client_id] = (now, tenant)
        return tenant
