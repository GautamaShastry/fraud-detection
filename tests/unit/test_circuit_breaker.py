import pytest

from src.features.velocity import CircuitBreaker, VelocityResult, VelocityUnavailableError


async def test_breaker_opens_after_failure_and_recovers():
    now = [0.0]
    breaker = CircuitBreaker(open_s=2.0, clock=lambda: now[0])
    calls = []

    async def failing():
        calls.append("fail")
        raise ConnectionError

    async def working():
        calls.append("ok")
        return VelocityResult(1)

    with pytest.raises(ConnectionError):
        await breaker.call(failing)
    with pytest.raises(VelocityUnavailableError):  # fast-fail, Redis not touched
        await breaker.call(working)
    assert calls == ["fail"]
    now[0] = 2.5
    assert (await breaker.call(working)).value == 1
    assert calls == ["fail", "ok"]
