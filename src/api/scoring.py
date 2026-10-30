"""Scoring endpoints (FR-02, FR-03, FR-04, FR-07; T-102..T-105).

POST /score flow, bounded by the 170 ms server deadline:

  1. authenticate, resolve tenant, validate (incl. clock skew)
  2. claim the attempt in PostgreSQL           -> 409 conflict / 202 in progress / 200 replay
  3. score: velocity + primary, else fallback   -> explicit 503 if neither succeeds
  4. commit the immutable assessment (I-03)     -> only then 200

Any dependency failure answers 503/504 so the marketplace continues to ordinary payment
authorization. A risk score is never returned unless it was durably recorded.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from opentelemetry.trace import Status, StatusCode

from src.api.auth import SCOPE_SCORE, Principal, require_scope
from src.api.problems import ProblemError
from src.api.schemas import ExternalId, ScoreRequest, clock_errors
from src.api.settings import Settings
from src.domain.identity import Pseudonymizer
from src.domain.models import Attempt, AttemptStatus
from src.scoring.engine import ScoringEngine
from src.storage import attempts as repo
from src.telemetry import metrics
from src.telemetry.setup import tracer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scoring"])


def _unavailable(detail: str, *, retryable: bool) -> ProblemError:
    return ProblemError(
        503, "SCORING_UNAVAILABLE", "Scoring unavailable", detail, retryable=retryable
    )


def build_attempt(
    tenant_id: str, body: ScoreRequest, received_at: datetime, pseudo: Pseudonymizer
) -> Attempt:
    def ref(kind: str, value: Any) -> str | None:
        return pseudo.ref(tenant_id, kind, None if value is None else str(value))

    return Attempt(
        tenant_id=tenant_id,
        payment_attempt_id=body.payment_attempt_id,
        idempotency_key=body.idempotency_key,
        transaction_id=body.transaction_id,
        fingerprint=pseudo.fingerprint(tenant_id, body.model_dump(mode="json")),
        amount_minor=body.amount_minor,
        currency=body.currency,
        occurred_at=body.occurred_at,
        received_at=received_at,
        customer_ref=ref("customer", body.customer_id),
        device_ref=ref("device", body.device_id),
        payment_method_ref=ref("payment_method", body.payment_method_token),
        ip_ref=ref("ip", body.ip_address),
        shipping_ref=ref("shipping", body.shipping_destination_ref),
        account_created_at=body.account_created_at,
    )


def _record(outcome: str, started: float) -> None:
    metrics.SCORE_REQUESTS.labels(outcome=outcome).inc()
    metrics.SCORE_LATENCY.labels(outcome=outcome).observe(time.perf_counter() - started)


@router.post("/api/v1/fraud/score")
async def score(
    request: Request,
    principal: Annotated[Principal, Depends(require_scope(SCOPE_SCORE))],
    body: ScoreRequest,
) -> JSONResponse:
    started = time.perf_counter()
    settings: Settings = request.app.state.settings
    clock: Callable[[], datetime] = request.app.state.clock
    now = clock()

    errors = clock_errors(
        body,
        now,
        max_future_skew_s=settings.max_future_skew_s,
        max_attempt_age_s=settings.max_attempt_age_s,
    )
    if errors:
        _record("rejected", started)
        raise ProblemError(422, "VALIDATION_ERROR", "Request validation failed", errors=errors)

    engine: ScoringEngine | None = request.app.state.engine
    pool = request.app.state.pool
    if engine is None or pool is None:
        _record("unavailable", started)
        raise _unavailable(
            "No validated scorer is deployed. Continue checkout without an assessment.",
            retryable=False,
        )

    attempt = build_attempt(principal.tenant_id, body, now, request.app.state.pseudonymizer)
    held: dict[str, uuid.UUID] = {}
    with tracer.start_as_current_span("fraud.score") as span:
        span.set_attribute("tenant.id", principal.tenant_id)
        try:
            async with asyncio.timeout(settings.server_deadline_s):
                outcome, response = await _score_flow(
                    request, attempt, engine, settings, clock, held
                )
        except TimeoutError:
            outcome = "deadline"
            span.set_status(Status(StatusCode.ERROR, "deadline exceeded"))
            response = None
            if "claim" in held:
                await _release_claim(pool, attempt, held["claim"], clock)
        except ProblemError as problem:
            # Conflicts and unavailability are outcomes too; record them before re-raising.
            outcome = {409: "conflict", 503: "unavailable"}.get(problem.status, "rejected")
            span.set_attribute("fraud.outcome", outcome)
            _record(outcome, started)
            raise
        span.set_attribute("fraud.outcome", outcome)

    _record(outcome, started)
    if response is None:
        raise ProblemError(
            504,
            "SCORING_DEADLINE_EXCEEDED",
            "Scoring deadline exceeded",
            "Continue checkout without an assessment, or retry with the same request.",
            retryable=True,
        )
    return response


async def _score_flow(
    request: Request,
    attempt: Attempt,
    engine: ScoringEngine,
    settings: Settings,
    clock: Callable[[], datetime],
    held: dict[str, uuid.UUID],
) -> tuple[str, JSONResponse]:
    pool = request.app.state.pool
    try:
        with tracer.start_as_current_span("postgres.claim_attempt"), _Timed("postgres", "claim"):
            claim = await repo.claim_attempt(
                pool, attempt, attempt.received_at, timedelta(seconds=settings.attempt_lease_s)
            )
    except TimeoutError:
        raise
    except Exception as exc:
        logger.error("Attempt claim failed (%s)", type(exc).__name__, exc_info=True)
        metrics.DEPENDENCY_ERRORS.labels(dependency="postgres", operation="claim").inc()
        raise _unavailable("Assessment store unavailable.", retryable=True) from exc

    if claim.outcome == repo.ClaimOutcome.CONFLICT:
        raise ProblemError(
            409,
            "IDEMPOTENCY_CONFLICT",
            "Idempotency conflict",
            "This idempotency key or payment attempt was already used with a different request.",
        )
    if claim.outcome == repo.ClaimOutcome.IN_PROGRESS:
        return "in_progress", _in_progress(attempt.payment_attempt_id, claim.retry_after_ms or 50)
    if claim.outcome == repo.ClaimOutcome.COMPLETED:
        assert claim.assessment is not None
        return "replay", JSONResponse(claim.assessment.to_response())

    assert claim.claim_token is not None
    held["claim"] = claim.claim_token
    with tracer.start_as_current_span("scoring.assess"):
        assessment = await engine.assess(attempt)
    if assessment is None:
        try:
            await repo.mark_unavailable(pool, attempt, claim.claim_token, clock())
        except Exception:  # noqa: BLE001 - the lease expires on its own
            logger.warning("Could not mark attempt unavailable")
        raise _unavailable(
            "Neither primary nor fallback scoring produced a valid assessment.", retryable=True
        )

    try:
        with (
            tracer.start_as_current_span("postgres.commit"),
            _Timed("postgres", "commit"),
        ):
            stored = await repo.complete_attempt(pool, assessment, claim.claim_token, clock())
    except repo.LostClaimError:
        return "in_progress", _in_progress(attempt.payment_attempt_id, 50)
    except TimeoutError:
        raise
    except Exception as exc:
        logger.error("Assessment commit failed (%s)", type(exc).__name__, exc_info=True)
        metrics.DEPENDENCY_ERRORS.labels(dependency="postgres", operation="commit").inc()
        raise _unavailable("Assessment could not be recorded.", retryable=True) from exc

    outcome = (
        "replay"
        if stored.assessment_id != assessment.assessment_id
        else (stored.scoring_mode.value.lower())
    )
    return outcome, JSONResponse(stored.to_response())


async def _release_claim(
    pool: Any, attempt: Attempt, token: uuid.UUID, clock: Callable[[], datetime]
) -> None:
    """After a missed deadline, release the claim so an immediate retry can re-evaluate
    instead of waiting for the lease to expire. A no-op if the commit already landed."""
    try:
        await asyncio.wait_for(repo.mark_unavailable(pool, attempt, token, clock()), 0.5)
    except Exception:  # noqa: BLE001 - the lease still expires on its own
        logger.warning("Could not release claim after deadline")


def _in_progress(payment_attempt_id: str, retry_after_ms: int) -> JSONResponse:
    return JSONResponse(
        {
            "status": "IN_PROGRESS",
            "payment_attempt_id": payment_attempt_id,
            "retry_after_ms": retry_after_ms,
        },
        status_code=202,
    )


class _Timed:
    def __init__(self, dependency: str, operation: str) -> None:
        self._labels = metrics.DEPENDENCY_LATENCY.labels(dependency=dependency, operation=operation)

    def __enter__(self) -> None:
        self._start = time.perf_counter()

    def __exit__(self, *exc: object) -> None:
        self._labels.observe(time.perf_counter() - self._start)


@router.get("/api/v1/fraud/attempts/{payment_attempt_id}/assessment")
async def get_assessment(
    request: Request,
    payment_attempt_id: ExternalId,
    principal: Annotated[Principal, Depends(require_scope(SCOPE_SCORE))],
) -> JSONResponse:
    pool = request.app.state.pool
    if pool is None:
        raise ProblemError(503, "SERVICE_UNAVAILABLE", "Service unavailable", retryable=True)
    try:
        state = await repo.get_attempt_state(pool, principal.tenant_id, payment_attempt_id)
    except Exception as exc:
        raise ProblemError(
            503, "SERVICE_UNAVAILABLE", "Service unavailable", retryable=True
        ) from exc
    now = request.app.state.clock()
    if state is not None and state.assessment is not None:
        return JSONResponse(state.assessment.to_response())
    if (
        state is not None
        and state.status == AttemptStatus.IN_PROGRESS
        and state.lease_expires_at > now
    ):
        remaining = int((state.lease_expires_at - now) / timedelta(milliseconds=1))
        return _in_progress(payment_attempt_id, min(max(remaining, 10), 500))
    raise ProblemError(404, "NOT_FOUND", "Not found", "No assessment established for this attempt")
