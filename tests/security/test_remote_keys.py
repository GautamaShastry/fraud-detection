"""Review finding #1: unknown `kid`s must not let an unauthenticated caller force JWKS fetches."""

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
