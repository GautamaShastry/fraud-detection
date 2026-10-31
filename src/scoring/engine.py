"""Deadline-bounded scoring with explicit degradation (FR-06, FR-07, T-105).

Budgets are provisional and configurable (PRD 4.1): primary path (velocity lookup + primary
model) up to 110 ms, fallback up to 40 ms. They are limits inside the caller's single request
deadline, not separate clocks: the primary path gets only what is left after setting aside the
fallback budget and `commit_reserve_s`, so a slow start or a slow primary cannot use up the time
the fallback and the commit need. Outcomes:

* ML        primary model produced a valid output within budget
* FALLBACK  primary failed or timed out, or velocity was unavailable, and the validated
            fallback produced a valid output within its budget (never recommends decline)
* None      neither could; the caller returns an explicit unavailable response.
            A risk score is never fabricated.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

from src.domain.bounded import within
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
    commit_reserve_s: float = 0.020  # Left for recording the assessment after scoring


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

    def warm(self) -> None:
        """Run the scoring path once, so the first real request doesn't pay for first-use costs.

        Touches no storage and no Redis. Slow (uncompiled) models are skipped: they would only
        delay startup, and they are not fit for the deadline anyway.
        """
        now = self._clock()
        attempt = Attempt(
            tenant_id="warmup",
            payment_attempt_id="warmup",
            idempotency_key="warmup",
            transaction_id="warmup",
            fingerprint=b"warmup",
            amount_minor=1000,
            currency="USD",
            occurred_at=now,
            received_at=now,
        )
        b = self.bundle
        paths = (
            (b.primary, b.primary_policy, 1, ScoringMode.ML),
            (b.fallback, b.fallback_policy, None, ScoringMode.FALLBACK),
        )
        for scorer, policy, f01, mode in paths:
            if scorer is None or policy is None or not getattr(scorer, "inline", False):
                continue
            features = build_features(self._inputs(attempt, f01))
            output = scorer.predict(features)
            self._assessment(attempt, features, output, mode, scorer, policy, time.perf_counter())

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

    async def assess(self, attempt: Attempt, deadline: float | None = None) -> Assessment | None:
        """Score within `deadline` (a time.perf_counter() value; None means budgets only).

        Raises TimeoutError if the deadline leaves no time for the fallback.
        """
        started = time.perf_counter()
        scoring_ends = math.inf if deadline is None else deadline - self.budgets.commit_reserve_s
        primary_deadline = min(
            started + self.budgets.primary_s, scoring_ends - self.budgets.fallback_s
        )

        try:
            if primary_deadline <= started:
                raise TimeoutError  # No time to spare: go straight to the fallback.
            velocity = await within(
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

        return await self._fallback(attempt, started, scoring_ends)

    async def _fallback(
        self, attempt: Attempt, started: float, scoring_ends: float
    ) -> Assessment | None:
        bundle = self.bundle
        if bundle.fallback is None or bundle.fallback_policy is None:
            return None
        # Velocity is excluded: the fallback must not depend on Redis.
        features = build_features(self._inputs(attempt, None))
        budget = min(self.budgets.fallback_s, scoring_ends - time.perf_counter())
        try:
            output = await _predict(bundle.fallback, features, budget)
        except Exception as exc:  # noqa: BLE001
            reason = "timeout" if isinstance(exc, TimeoutError) else type(exc).__name__
            metrics.FALLBACK_FAILURES.labels(reason=reason).inc()
            logger.warning("Fallback scoring failed (%s)", reason)
            if isinstance(exc, TimeoutError) and budget < self.budgets.fallback_s:
                raise  # The request's deadline ran out, not the fallback's own budget.
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
