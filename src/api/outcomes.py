"""POST /api/v1/fraud/outcomes (FR-08, T-201).

Answers 202 only after the event is committed to the PostgreSQL inbox. A retry with the same
event_id and body is acknowledged as DUPLICATE with no further effect; the same event_id with a
different body is a 409. If the inbox is unavailable the event is not acknowledged (503).
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from src.api.auth import SCOPE_OUTCOMES, Principal, require_scope
from src.api.problems import ProblemError
from src.api.schemas import OutcomeEvent
from src.domain.models import format_utc
from src.storage.outcomes import InsertResult, insert_event
from src.telemetry import metrics

logger = logging.getLogger(__name__)

router = APIRouter(tags=["outcomes"])


@router.post("/api/v1/fraud/outcomes")
async def report_outcome(
    request: Request,
    principal: Annotated[Principal, Depends(require_scope(SCOPE_OUTCOMES))],
    body: OutcomeEvent,
) -> JSONResponse:
    settings = request.app.state.settings
    now = request.app.state.clock()
    if body.occurred_at > now + timedelta(seconds=settings.max_future_skew_s):
        metrics.OUTCOME_EVENTS.labels(result="rejected").inc()
        raise ProblemError(
            422,
            "VALIDATION_ERROR",
            "Request validation failed",
            errors=[{"field": "occurred_at", "message": "is too far in the future"}],
        )

    pool = request.app.state.pool
    if pool is None:
        raise ProblemError(
            503,
            "SERVICE_UNAVAILABLE",
            "Service unavailable",
            "Event not accepted; retry later.",
            retryable=True,
        )

    payload = body.model_dump(mode="json")
    payload_hash = request.app.state.pseudonymizer.fingerprint(principal.tenant_id, payload)
    event = body.model_dump()
    try:
        result, received_at = await insert_event(
            pool, principal.tenant_id, event, payload_hash, now
        )
    except Exception as exc:
        logger.error("Outcome inbox insert failed (%s)", type(exc).__name__)
        metrics.OUTCOME_EVENTS.labels(result="unavailable").inc()
        raise ProblemError(
            503,
            "SERVICE_UNAVAILABLE",
            "Service unavailable",
            "Event not accepted; retry later.",
            retryable=True,
        ) from exc

    metrics.OUTCOME_EVENTS.labels(result=result.value.lower()).inc()
    if result == InsertResult.CONFLICT:
        raise ProblemError(
            409,
            "EVENT_ID_CONFLICT",
            "Event ID conflict",
            "This event_id was already used with a different body.",
        )
    return JSONResponse(
        {"event_id": body.event_id, "status": result.value, "received_at": format_utc(received_at)},
        status_code=202,
    )
