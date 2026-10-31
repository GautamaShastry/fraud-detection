"""Review findings #1 and #10: JWKS refresh is rate-limited, revocation is bounded by key age, and
only a real download leaves the event loop."""

import asyncio
import secrets
import threading
import time
from typing import Any

import pytest

from src.api.auth import RemoteKeys, TokenVerifier
from src.api.problems import ProblemError
from tests.conftest import AUDIENCE, ISSUER


class FakeIssuer:
    """Stands in for the JWKS endpoint; counts fetches."""

    def __init__(self, jwks: dict[str, Any], fail: bool = False, delay_s: float = 0.0) -> None:
        self.jwks, self.fail, self.delay_s = jwks, fail, delay_s
        self.calls = 0

    def __call__(self, url: str, timeout_s: float) -> dict[str, Any]:
        self.calls += 1
        time.sleep(self.delay_s)
        if self.fail:
            raise OSError("issuer unreachable")
        return self.jwks


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _keys(issuer: FakeIssuer, clock: Clock) -> RemoteKeys:
    return RemoteKeys("https://issuer.test/jwks", min_refresh_s=60, fetch=issuer, clock=clock)


def test_forged_kid_flood_causes_at_most_one_fetch(jwks, make_token):
    issuer, clock = FakeIssuer(jwks), Clock()
    verifier = TokenVerifier(issuer=ISSUER, audience=AUDIENCE, keys=_keys(issuer, clock))
    for _ in range(200):
        with pytest.raises(ProblemError) as exc:
            verifier.verify(make_token(kid=f"forged-{secrets.token_hex(4)}"))
        assert exc.value.status == 401
    assert issuer.calls == 1


def test_known_kid_is_served_from_cache(jwks, make_token):
    issuer, clock = FakeIssuer(jwks), Clock()
    verifier = TokenVerifier(issuer=ISSUER, audience=AUDIENCE, keys=_keys(issuer, clock))
    for _ in range(50):
        assert verifier.verify(make_token()).client_id == "merchant-a"
    assert issuer.calls == 1  # the first lookup filled the cache; no more fetches


def test_rotation_is_picked_up_after_the_refresh_interval(jwks, make_token):
    rotated = {"keys": [k for k in jwks["keys"] if k["kid"] == "ec-1"]}
    issuer, clock = FakeIssuer(rotated), Clock()
    keys = _keys(issuer, clock)
    assert keys.get("ec-1") is not None
    assert keys.get("rsa-1") is None  # not published yet

    issuer.jwks = jwks  # the issuer starts publishing rsa-1
    clock.now += 30
    assert keys.get("rsa-1") is None and issuer.calls == 1  # still inside the interval
    clock.now += 31
    assert keys.get("rsa-1") is not None and issuer.calls == 2


def test_failed_fetches_are_rate_limited_and_keep_cached_keys(jwks):
    issuer, clock = FakeIssuer(jwks), Clock()
    keys = _keys(issuer, clock)
    assert keys.get("rsa-1") is not None
    issuer.fail = True
    clock.now += 61
    for _ in range(100):
        assert keys.get(f"unknown-{secrets.token_hex(3)}") is None
    assert issuer.calls == 2  # one failed attempt, not one per request
    assert keys.get("rsa-1") is not None  # a failed refresh never drops known keys


def test_concurrent_unknown_kids_share_one_fetch(jwks):
    issuer = FakeIssuer(jwks, delay_s=0.05)
    keys = RemoteKeys("https://issuer.test/jwks", min_refresh_s=60, fetch=issuer)
    threads = [threading.Thread(target=keys.get, args=(f"k-{i}",)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert issuer.calls == 1


def test_keys_removed_by_the_issuer_stop_validating(jwks):
    issuer, clock = FakeIssuer(jwks), Clock()
    keys = _keys(issuer, clock)
    assert keys.get("ec-1") is not None
    issuer.jwks = {"keys": [k for k in jwks["keys"] if k["kid"] == "rsa-1"]}
    clock.now += 61
    keys.get("unknown")  # triggers the refresh
    assert keys.get("ec-1") is None  # revoked key is no longer trusted


def _spawned_keys(issuer: FakeIssuer, clock: Clock) -> tuple[RemoteKeys, list]:
    pending: list = []
    keys = RemoteKeys(
        "https://issuer.test/jwks",
        min_refresh_s=60,
        max_age_s=300,
        fetch=issuer,
        clock=clock,
        spawn=pending.append,
    )
    return keys, pending


def test_revoked_key_stops_validating_after_max_age_without_any_unknown_kid(jwks):
    issuer, clock = FakeIssuer(jwks), Clock()
    keys, pending = _spawned_keys(issuer, clock)
    assert keys.get("ec-1") is not None

    issuer.jwks = {"keys": [k for k in jwks["keys"] if k["kid"] == "rsa-1"]}  # ec-1 revoked
    clock.now += 299
    assert keys.get("ec-1") is not None and not pending  # still within max age

    clock.now += 2
    assert keys.get("ec-1") is not None  # answered from cache; the refresh is in the background
    assert len(pending) == 1
    pending.pop()()
    assert keys.get("ec-1") is None


def test_stale_lookups_never_wait_and_start_one_refresh(jwks):
    issuer, clock = FakeIssuer(jwks), Clock()
    keys, pending = _spawned_keys(issuer, clock)
    assert keys.get("ec-1") is not None
    clock.now += 301
    for _ in range(100):
        assert not keys.needs_io("ec-1")
        assert keys.get("ec-1") is not None
    assert len(pending) == 1
    assert issuer.calls == 1  # the refresh has not run, so nothing blocked on the issuer


def test_failed_background_refresh_keeps_keys_and_stays_rate_limited(jwks):
    issuer, clock = FakeIssuer(jwks), Clock()
    keys, pending = _spawned_keys(issuer, clock)
    assert keys.get("ec-1") is not None
    issuer.fail = True
    clock.now += 301
    assert keys.get("ec-1") is not None
    pending.pop()()
    assert keys.get("ec-1") is not None  # the issuer being down does not reject live traffic
    assert not pending  # inside min_refresh_s: no new attempt
    clock.now += 61
    assert keys.get("ec-1") is not None
    assert len(pending) == 1  # still stale, so it tries again


def _forbid_threads(monkeypatch) -> list:
    hops: list = []

    async def to_thread(fn, *args):
        hops.append(fn)
        return fn(*args)

    monkeypatch.setattr(asyncio, "to_thread", to_thread)
    return hops


async def test_cached_keys_verify_without_a_thread_hop(jwks, make_token, monkeypatch):
    hops = _forbid_threads(monkeypatch)
    issuer, clock = FakeIssuer(jwks), Clock()
    verifier = TokenVerifier(issuer=ISSUER, audience=AUDIENCE, keys=_keys(issuer, clock))

    token = make_token()
    assert (await verifier.averify(token)).client_id == "merchant-a"  # first use must download
    assert len(hops) == 1
    for _ in range(20):
        assert (await verifier.averify(token)).client_id == "merchant-a"
    assert len(hops) == 1


async def test_unknown_kid_between_refreshes_is_rejected_without_a_thread_hop(
    jwks, make_token, monkeypatch
):
    hops = _forbid_threads(monkeypatch)
    issuer, clock = FakeIssuer(jwks), Clock()
    keys = _keys(issuer, clock)
    assert keys.get("rsa-1") is not None
    verifier = TokenVerifier(issuer=ISSUER, audience=AUDIENCE, keys=keys)
    with pytest.raises(ProblemError) as exc:
        await verifier.averify(make_token(kid="forged"))
    assert exc.value.status == 401
    assert not hops


async def test_malformed_token_is_rejected_without_a_download(jwks, monkeypatch):
    hops = _forbid_threads(monkeypatch)
    issuer, clock = FakeIssuer(jwks), Clock()
    verifier = TokenVerifier(issuer=ISSUER, audience=AUDIENCE, keys=_keys(issuer, clock))
    with pytest.raises(ProblemError):
        await verifier.averify("not-a-token")
    assert not hops and issuer.calls == 0
