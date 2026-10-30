"""Deadline-bounded scoring with explicit degradation (FR-06, FR-07, T-105).

Budgets are provisional and configurable (PRD 4.1): primary path (velocity lookup + primary
model) 110 ms, fallback 40 ms. Outcomes:

* ML        primary model produced a valid output within budget
* FALLBACK  primary failed or timed out, or velocity was unavailable, and the validated
            fallback produced a valid output within its budget (never recommends decline)
* None      neither could; the caller returns an explicit unavailable response.
            A risk score is never fabricated.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

from src.domain.models import Assessment, Attempt, Recommendation, ScoringMode
from src.features.velocity import VelocityResult
from src.scoring.features import FeatureInputs, build_features
from src.scoring.model import Scorer, ScoringBundle, ScoringPolicy
from src.telemetry import metrics

logger = logging.getLogger(__name__)

VelocityLookup = Callable[[Attempt], Awaitable[VelocityResult]]


@dataclass(frozen=True)
class Budgets:
    primary_s: float = 0.110
    fallback_s: float = 0.040


async def _predict(scorer: Scorer, features: dict[str, float | None], timeout_s: float) -> float:
    if timeout_s <= 0:
        raise TimeoutError
    if getattr(scorer, "inline", False):
        # Compiled models take ~1 ms of CPU: run inline (a thread hop costs more and, under
        # load, queues behind abandoned work) and enforce the budget by elapsed time.
        started = time.perf_counter()
        output = scorer.predict(features)
        if time.perf_counter() - started > timeout_s:
            raise TimeoutError
    else:
        # Slow (uncompiled) models run off the event loop so the deadline can preempt waiting.
        output = await asyncio.wait_for(asyncio.to_thread(scorer.predict, features), timeout_s)
    if not 0.0 <= output <= 1.0:
        raise ValueError(f"model output out of range: {output}")
    return output


class ScoringEngine:
    def __init__(
        self,
        bundle: ScoringBundle,
        velocity: VelocityLookup,
        clock: Callable[[], datetime],
        budgets: Budgets | None = None,
    ) -> None:
        self.bundle = bundle
        self._velocity = velocity
        self._clock = clock
        self.budgets = budgets or Budgets()

    def _inputs(self, attempt: Attempt, f01: int | None) -> FeatureInputs:
        return FeatureInputs(
            amount_minor=attempt.amount_minor,
            occurred_at=attempt.occurred_at,
            account_created_at=attempt.account_created_at,
            f01=f01,
            has_customer=attempt.customer_ref is not None,
            has_device=attempt.device_ref is not None,
            has_payment_method=attempt.payment_method_ref is not None,
            has_ip=attempt.ip_ref is not None,
            has_shipping=attempt.shipping_ref is not None,
        )

    async def assess(self, attempt: Attempt) -> Assessment | None:
        started = time.perf_counter()
        primary_deadline = started + self.budgets.primary_s

        try:
            velocity = await asyncio.wait_for(
                self._velocity(attempt), max(primary_deadline - time.perf_counter(), 0.001)
            )
            features = build_features(self._inputs(attempt, velocity.value))
            output = await _predict(
                self.bundle.primary, features, primary_deadline - time.perf_counter()
            )
            return self._assessment(
                attempt,
                features,
                output,
                ScoringMode.ML,
                self.bundle.primary,
                self.bundle.primary_policy,
                started,
            )
        except Exception as exc:  # noqa: BLE001 - every primary failure degrades explicitly
            reason = "timeout" if isinstance(exc, TimeoutError) else type(exc).__name__
            metrics.PRIMARY_FAILURES.labels(reason=reason).inc()
            logger.warning("Primary scoring failed (%s); trying fallback", reason)

        return await self._fallback(attempt, started)

    async def _fallback(self, attempt: Attempt, started: float) -> Assessment | None:
        bundle = self.bundle
        if bundle.fallback is None or bundle.fallback_policy is None:
            return None
        # Velocity is excluded: the fallback must not depend on Redis.
        features = build_features(self._inputs(attempt, None))
        try:
            output = await _predict(bundle.fallback, features, self.budgets.fallback_s)
        except Exception as exc:  # noqa: BLE001
            reason = "timeout" if isinstance(exc, TimeoutError) else type(exc).__name__
            metrics.FALLBACK_FAILURES.labels(reason=reason).inc()
            logger.warning("Fallback scoring failed (%s)", reason)
            return None
        return self._assessment(
            attempt,
            features,
            output,
            ScoringMode.FALLBACK,
            bundle.fallback,
            bundle.fallback_policy,
            started,
        )

    def _assessment(
        self,
        attempt: Attempt,
        features: dict[str, float | None],
        output: float,
        mode: ScoringMode,
        scorer: Scorer,
        policy: ScoringPolicy,
        started: float,
    ) -> Assessment:
        score = policy.risk_score(output)
        recommendation = policy.recommend(output)
        if mode == ScoringMode.FALLBACK and recommendation == Recommendation.RECOMMEND_DECLINE:
            recommendation = Recommendation.VERIFY  # defense in depth; policy forbids it too
        return Assessment(
            assessment_id=uuid.uuid4(),
            tenant_id=attempt.tenant_id,
            payment_attempt_id=attempt.payment_attempt_id,
            transaction_id=attempt.transaction_id,
            risk_score=score,
            recommendation=recommendation,
            risk_signals=self.bundle.signals(features),
            scoring_mode=mode,
            model_version=scorer.model_version,
            policy_version=policy.version,
            evaluated_at=self._clock(),
            scoring_latency_ms=(time.perf_counter() - started) * 1000,
            features={k: v for k, v in features.items()},
        )
