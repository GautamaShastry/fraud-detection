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
    """Fast-fails calls for `open_s` after a failure, so a dead Redis costs one timeout, not
    one timeout per request. Callers then degrade to the fallback scorer immediately."""

    def __init__(self, open_s: float = 2.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._open_s = open_s
        self._clock = clock
        self._open_until = 0.0

    @property
    def is_open(self) -> bool:
        return self._clock() < self._open_until

    async def call(self, fn: Callable[[], Awaitable[VelocityResult]]) -> VelocityResult:
        if self.is_open:
            raise VelocityUnavailableError("velocity circuit open")
        try:
            return await fn()
        except BaseException:
            # Includes cancellation by the primary deadline: a hung Redis also opens it.
            self._open_until = self._clock() + self._open_s
            raise
