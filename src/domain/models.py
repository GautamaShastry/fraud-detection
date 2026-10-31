"""Core domain types shared by the API, scoring and storage layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID


class Recommendation(StrEnum):
    PROCEED = "PROCEED"
    VERIFY = "VERIFY"
    RECOMMEND_DECLINE = "RECOMMEND_DECLINE"


class ScoringMode(StrEnum):
    ML = "ML"
    FALLBACK = "FALLBACK"


class AttemptStatus(StrEnum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class RiskSignal:
    code: str
    description: str | None = None

    def to_json(self) -> dict[str, str]:
        out = {"code": self.code}
        if self.description is not None:
            out["description"] = self.description
        return out


@dataclass(frozen=True)
class Attempt:
    """A validated scoring request with identifiers already pseudonymized."""

    tenant_id: str
    payment_attempt_id: str
    idempotency_key: str
    transaction_id: str
    fingerprint: bytes
    amount_minor: int
    currency: str
    occurred_at: datetime
    received_at: datetime
    customer_ref: str | None = None
    device_ref: str | None = None
    payment_method_ref: str | None = None
    ip_ref: str | None = None
    shipping_ref: str | None = None
    account_created_at: datetime | None = None


@dataclass(frozen=True)
class Assessment:
    assessment_id: UUID
    tenant_id: str
    payment_attempt_id: str
    transaction_id: str
    risk_score: int
    recommendation: Recommendation
    risk_signals: tuple[RiskSignal, ...]
    scoring_mode: ScoringMode
    model_version: str
    policy_version: str
    evaluated_at: datetime
    scoring_latency_ms: float
    features: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.risk_score <= 100:
            raise ValueError("risk_score must be within 0-100")
        if (
            self.scoring_mode == ScoringMode.FALLBACK
            and self.recommendation == Recommendation.RECOMMEND_DECLINE
        ):
            raise ValueError("Fallback assessments may not recommend decline (I-08)")

    def to_response(self) -> dict[str, Any]:
        return {
            "assessment_id": str(self.assessment_id),
            "transaction_id": self.transaction_id,
            "payment_attempt_id": self.payment_attempt_id,
            "risk_score": self.risk_score,
            "recommendation": self.recommendation.value,
            "risk_signals": [s.to_json() for s in self.risk_signals],
            "scoring_mode": self.scoring_mode.value,
            "model_version": self.model_version,
            "policy_version": self.policy_version,
            "evaluated_at": format_utc(self.evaluated_at),
        }


def format_utc(ts: datetime) -> str:
    """RFC 3339 UTC with millisecond precision, matching the contract's UtcTimestamp."""
    if ts.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
