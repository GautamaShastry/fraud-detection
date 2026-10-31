"""Request models mirroring contracts/openapi.yaml (FR-02)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, IPvAnyAddress, StringConstraints

ID_PATTERN = r"^[A-Za-z0-9._:-]+$"
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?(Z|\+00:00)$")


def _require_utc_string(value: Any) -> Any:
    if not isinstance(value, str) or not _UTC_TIMESTAMP.match(value):
        raise ValueError("must be an RFC 3339 UTC timestamp ending in Z or +00:00")
    return value


ExternalId = Annotated[str, StringConstraints(min_length=1, max_length=128, pattern=ID_PATTERN)]
UtcTimestamp = Annotated[datetime, BeforeValidator(_require_utc_string)]


class ScoreRequest(BaseModel):
    # Unknown fields (including any attempt to assert tenant_id) are rejected.
    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: ExternalId
    payment_attempt_id: ExternalId
    idempotency_key: Annotated[
        str, StringConstraints(min_length=8, max_length=128, pattern=ID_PATTERN)
    ]
    amount_minor: Annotated[int, Field(strict=True, ge=0, le=1_000_000_000_000)]
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
    occurred_at: UtcTimestamp

    customer_id: ExternalId | None = None
    device_id: ExternalId | None = None
    payment_method_token: ExternalId | None = None
    ip_address: IPvAnyAddress | None = None
    shipping_destination_ref: ExternalId | None = None
    account_created_at: UtcTimestamp | None = None


def clock_errors(
    req: ScoreRequest, now: datetime, *, max_future_skew_s: int, max_attempt_age_s: int
) -> list[dict[str, str]]:
    """Semantic timestamp checks that depend on server time (I-10 provisional bounds)."""
    errors = []
    future_limit = now + timedelta(seconds=max_future_skew_s)
    if req.occurred_at > future_limit:
        errors.append({"field": "occurred_at", "message": "is too far in the future"})
    if req.occurred_at < now - timedelta(seconds=max_attempt_age_s):
        errors.append({"field": "occurred_at", "message": "is older than the accepted window"})
    if req.account_created_at is not None and req.account_created_at > future_limit:
        errors.append({"field": "account_created_at", "message": "is in the future"})
    return errors


class OutcomeEventType(StrEnum):
    AUTHORIZATION_APPROVED = "AUTHORIZATION_APPROVED"
    AUTHORIZATION_DECLINED = "AUTHORIZATION_DECLINED"
    CHARGEBACK_REPORTED = "CHARGEBACK_REPORTED"
    CONFIRMED_FRAUD = "CONFIRMED_FRAUD"
    CONFIRMED_LEGITIMATE = "CONFIRMED_LEGITIMATE"
    LABEL_RETRACTED = "LABEL_RETRACTED"


class OutcomeSource(StrEnum):
    MARKETPLACE = "MARKETPLACE"
    PROCESSOR = "PROCESSOR"
    ISSUER = "ISSUER"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class OutcomeEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: ExternalId
    payment_attempt_id: ExternalId
    transaction_id: ExternalId | None = None
    event_type: OutcomeEventType
    source: OutcomeSource
    occurred_at: UtcTimestamp
    reason_code: Annotated[str, StringConstraints(max_length=64, pattern=ID_PATTERN)] | None = None
