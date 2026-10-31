"""Review finding #5: one slow call or a request's own deadline must not open the breaker."""

import asyncio

import pytest

from src.features.velocity import CircuitBreaker, VelocityResult, VelocityUnavailableError


class Redis:
    def __init__(self) -> None:
        self.calls = 0

    async def fails(self) -> VelocityResult:
        self.calls += 1
        raise ConnectionError

    async def works(self) -> VelocityResult:
        self.calls += 1
        return VelocityResult(1)

    async def hangs(self) -> VelocityResult:
        self.calls += 1
        await asyncio.sleep(10)
        raise AssertionError("unreachable")


def _breaker(now: list[float], **kwargs) -> CircuitBreaker:
    return CircuitBreaker(open_s=2.0, failure_threshold=3, clock=lambda: now[0], **kwargs)


async def _fail(breaker: CircuitBreaker, redis: Redis, times: int) -> None:
    for _ in range(times):
        with pytest.raises(ConnectionError):
            await breaker.call(redis.fails)


async def test_a_single_failure_does_not_open_the_breaker():
    breaker, redis = _breaker([0.0]), Redis()
    await _fail(breaker, redis, 1)
    assert not breaker.is_open
    assert (await breaker.call(redis.works)).value == 1


async def test_opens_after_consecutive_failures_and_fails_fast():
    breaker, redis = _breaker([0.0]), Redis()
    await _fail(breaker, redis, 3)
    assert breaker.is_open
    with pytest.raises(VelocityUnavailableError):
        await breaker.call(redis.works)
    assert redis.calls == 3  # the fast-fail did not touch Redis


async def test_a_success_resets_the_count():
    breaker, redis = _breaker([0.0]), Redis()
    for _ in range(5):
        await _fail(breaker, redis, 2)
        await breaker.call(redis.works)
    assert not breaker.is_open  # intermittent hiccups never trip it


async def test_deadline_cancellation_does_not_count():
    breaker, redis = _breaker([0.0]), Redis()
    for _ in range(10):
        task = asyncio.create_task(breaker.call(redis.hangs))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not breaker.is_open


async def test_cancellation_through_wait_for_does_not_count():
    # How the scoring engine actually cuts off the velocity call.
    breaker, redis = _breaker([0.0], call_timeout_s=10.0), Redis()
    for _ in range(5):
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(breaker.call(redis.hangs), 0.001)
    assert not breaker.is_open


async def test_hung_redis_opens_through_the_call_timeout():
    breaker, redis = _breaker([0.0], call_timeout_s=0.001), Redis()
    for _ in range(3):
        with pytest.raises(TimeoutError):
            await breaker.call(redis.hangs)
    assert breaker.is_open


async def test_recovers_after_the_open_period():
    now = [0.0]
    breaker, redis = _breaker(now), Redis()
    await _fail(breaker, redis, 3)
    now[0] = 2.5
    assert (await breaker.call(redis.works)).value == 1
    assert not breaker.is_open


async def test_failure_after_the_open_period_reopens_immediately():
    now = [0.0]
    breaker, redis = _breaker(now), Redis()
    await _fail(breaker, redis, 3)
    now[0] = 2.5
    await _fail(breaker, redis, 1)  # one probe, not three more timeouts
    assert breaker.is_open
