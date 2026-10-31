"""F01: distinct tokenized payment methods per device in the last five minutes (FR-05, T-103).

The count includes the current unique attempt. One Lua script makes the update atomic:

* Retries of the same attempt return the value recorded for that attempt and change nothing
  (a per-attempt dedupe key holds the result for the idempotency horizon).
* The window is a sorted set of payment-method pseudonyms scored by last-seen time, so the
  same token seen twice counts once.
* Time is the server receive time (I-10), passed in so tests can inject a clock.
* Missing device or payment-method IDs yield a missing feature; nothing is fabricated.

I-11 is open: capacity and Lua cost still need benchmarking.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import redis.asyncio as aioredis

F01_NAME = "device_distinct_payment_methods_5m"
WINDOW_MS = 5 * 60 * 1000
DEDUPE_TTL_S = 48 * 3600  # Matches the ~48 h idempotency horizon (PRD section 8)

# KEYS[1] dedupe key, KEYS[2] window zset
# ARGV[1] now_ms, ARGV[2] window_ms, ARGV[3] member, ARGV[4] dedupe ttl s
_F01_LUA = """
local recorded = redis.call('GET', KEYS[1])
if recorded then
  return {tonumber(recorded), 1}
end
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local prev = redis.call('ZSCORE', KEYS[2], ARGV[3])
if (not prev) or tonumber(prev) < now then
  redis.call('ZADD', KEYS[2], now, ARGV[3])
end
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now - window)
local count = redis.call('ZCOUNT', KEYS[2], now - window + 1, '+inf')
redis.call('PEXPIRE', KEYS[2], window + 60000)
redis.call('SET', KEYS[1], count, 'EX', tonumber(ARGV[4]))
return {count, 0}
"""


@dataclass(frozen=True)
class VelocityResult:
    value: int | None  # None = missing (no device or payment method on the attempt)
    replayed: bool = False


class VelocityStore:
    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client
        self._script = client.register_script(_F01_LUA)

    async def warm(self) -> None:
        """Open a connection and load the script so the first request pays neither."""
        await self._client.ping()
        await self._client.script_load(_F01_LUA)

    async def f01(
        self,
        tenant_id: str,
        payment_attempt_id: str,
        device_ref: str | None,
        payment_method_ref: str | None,
        now_ms: int,
    ) -> VelocityResult:
        if device_ref is None or payment_method_ref is None:
            return VelocityResult(None)
        # {tenant} hash tag keeps a tenant's keys in one cluster slot; all keys are namespaced.
        dedupe = f"t:{{{tenant_id}}}:f01:seen:{payment_attempt_id}"
        window = f"t:{{{tenant_id}}}:f01:dev:{device_ref}"
        count, replayed = await self._script(
            keys=[dedupe, window], args=[now_ms, WINDOW_MS, payment_method_ref, DEDUPE_TTL_S]
        )
        return VelocityResult(int(count), bool(replayed))


class VelocityUnavailableError(Exception):
    """Velocity is temporarily skipped (circuit open) after a Redis failure."""


class CircuitBreaker:
    """Fast-fails calls for `open_s` after `failure_threshold` consecutive failures, so a dead
    Redis costs a few timeouts, not one per request. Callers then degrade to the fallback scorer
    immediately. Any success resets the count, so isolated slow calls never move all traffic to
    the fallback. After `open_s` calls are let through again; one more failure reopens it.

    Being cancelled by the request's deadline is not a Redis failure: the request may have spent
    its time elsewhere. A hung Redis is still detected, because `call_timeout_s` is shorter than
    the primary budget and counts as a failure.
    """

    def __init__(
        self,
        open_s: float = 2.0,
        failure_threshold: int = 5,
        call_timeout_s: float = 0.08,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._open_s = open_s
        self._failure_threshold = failure_threshold
        self._call_timeout_s = call_timeout_s
        self._clock = clock
        self._open_until = 0.0
        self._failures = 0

    @property
    def is_open(self) -> bool:
        return self._clock() < self._open_until

    async def call(self, fn: Callable[[], Awaitable[VelocityResult]]) -> VelocityResult:
        if self.is_open:
            raise VelocityUnavailableError("velocity circuit open")
        try:
            async with asyncio.timeout(self._call_timeout_s):
                result = await fn()
        except Exception:  # Cancellation is a BaseException, so it does not count.
            self._failures += 1
            if self._failures >= self._failure_threshold:
                self._open_until = self._clock() + self._open_s
            raise
        self._failures = 0
        return result
