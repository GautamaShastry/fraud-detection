"""Review finding #3: budgets are limits inside one request deadline, not separate clocks.

Durations are large so these stay reliable on a busy machine.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from src.domain.models import Attempt, ScoringMode
from src.features.velocity import VelocityResult
from src.scoring.engine import Budgets, ScoringEngine
from src.scoring.stub import STUB_FALLBACK_VERSION, stub_bundle

NOW = datetime(2026, 1, 1, tzinfo=UTC)
ATTEMPT = Attempt(
    tenant_id="t",
    payment_attempt_id="pa",
    idempotency_key="idem",
    transaction_id="tx",
    fingerprint=b"f",
    amount_minor=100,
    currency="USD",
    occurred_at=NOW,
    received_at=NOW,
)


class Velocity:
    def __init__(self, hang: bool = False) -> None:
        self.hang = hang
        self.calls = 0

    async def __call__(self, attempt: Attempt) -> VelocityResult:
        self.calls += 1
        if self.hang:
            await asyncio.sleep(30)
        return VelocityResult(1)


@dataclass
class SlowScorer:
    model_version: str
    feature_names: tuple[str, ...]
    delay_s: float = 1.0

    def predict(self, features: dict[str, float | None]) -> float:
        time.sleep(self.delay_s)
        return 0.1


def _engine(velocity: Velocity, budgets: Budgets, bundle=None) -> ScoringEngine:
    return ScoringEngine(bundle or stub_bundle(), velocity, lambda: NOW, budgets)


async def test_no_deadline_uses_the_primary():
    assessment = await _engine(Velocity(), Budgets()).assess(ATTEMPT)
    assert assessment is not None and assessment.scoring_mode == ScoringMode.ML


async def test_slow_primary_leaves_time_for_the_fallback_and_the_commit():
    # The claim has used part of the deadline, so the primary may not use its full budget.
    budgets = Budgets(primary_s=5.0, fallback_s=0.4, commit_reserve_s=0.2)
    started = time.perf_counter()
    assessment = await _engine(Velocity(hang=True), budgets).assess(ATTEMPT, started + 1.0)
    elapsed = time.perf_counter() - started
    assert assessment is not None and assessment.scoring_mode == ScoringMode.FALLBACK
    assert 0.3 <= elapsed < 0.8  # cut at 1.0 - 0.2 - 0.4, finished before the commit reserve


async def test_too_little_time_for_the_primary_goes_straight_to_the_fallback():
    velocity = Velocity()
    budgets = Budgets(primary_s=5.0, fallback_s=5.0, commit_reserve_s=0.5)
    assessment = await _engine(velocity, budgets).assess(ATTEMPT, time.perf_counter() + 2.0)
    assert assessment is not None and assessment.scoring_mode == ScoringMode.FALLBACK
    assert velocity.calls == 0  # Redis is not asked: no F01 window update, no breaker effect


async def test_no_time_at_all_is_a_deadline_miss():
    with pytest.raises(TimeoutError):
        await _engine(Velocity(), Budgets()).assess(ATTEMPT, time.perf_counter())


async def test_fallback_cut_short_by_the_deadline_is_a_deadline_miss():
    base = stub_bundle()
    slow = SlowScorer(STUB_FALLBACK_VERSION, base.fallback.feature_names)  # type: ignore[union-attr]
    bundle = _replace_fallback(base, slow)
    budgets = Budgets(primary_s=5.0, fallback_s=5.0, commit_reserve_s=0.05)
    with pytest.raises(TimeoutError):
        await _engine(Velocity(hang=True), budgets, bundle).assess(
            ATTEMPT, time.perf_counter() + 0.1
        )


async def test_fallback_exceeding_its_own_budget_is_unavailable_not_a_deadline_miss():
    base = stub_bundle()
    slow = SlowScorer(STUB_FALLBACK_VERSION, base.fallback.feature_names)  # type: ignore[union-attr]
    bundle = _replace_fallback(base, slow)
    budgets = Budgets(primary_s=0.01, fallback_s=0.05, commit_reserve_s=0.0)
    assert await _engine(Velocity(hang=True), budgets, bundle).assess(ATTEMPT) is None


def _replace_fallback(base, fallback):
    from dataclasses import replace

    return replace(base, fallback=fallback)


class _Inline:
    inline = True

    def __init__(self, base) -> None:
        self.model_version, self.feature_names = base.model_version, base.feature_names
        self.calls = 0

    def predict(self, features: dict[str, float | None]) -> float:
        self.calls += 1
        return 0.1


def test_warm_runs_both_models_without_touching_velocity():
    from dataclasses import replace

    base = stub_bundle()
    primary, fallback = _Inline(base.primary), _Inline(base.fallback)
    velocity = Velocity()
    engine = _engine(velocity, Budgets(), replace(base, primary=primary, fallback=fallback))
    engine.warm()
    assert primary.calls == 1 and fallback.calls == 1
    assert velocity.calls == 0


def test_warm_skips_slow_models():
    base = stub_bundle()
    slow = SlowScorer("slow", base.primary.feature_names, delay_s=5.0)
    from dataclasses import replace

    started = time.perf_counter()
    _engine(Velocity(), Budgets(), replace(base, primary=slow, fallback=None)).warm()
    assert time.perf_counter() - started < 1.0
