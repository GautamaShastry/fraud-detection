"""Bearer token verification (FR-01 merchant API, FR-10 dashboard).

Merchant calls use OAuth 2.0 client credentials. The tenant is derived from the validated client
identity through the trusted api_clients mapping; it is never read from the request payload.
Any failure, including an unavailable mapping store, rejects the request: auth never fails open.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import urllib.request
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import jwt
from fastapi import Request

from src.api.problems import ProblemError
from src.telemetry import metrics

logger = logging.getLogger(__name__)

ALLOWED_ALGORITHMS = ("RS256", "ES256")
REQUIRED_CLAIMS = ("exp", "iat", "iss", "aud", "sub")

TENANT_LOOKUP_TIMEOUT_S = 0.1

SCOPE_SCORE = "fraud:score"
SCOPE_OUTCOMES = "fraud:outcomes"
SCOPE_READ = "fraud:read"


@dataclass(frozen=True)
class VerifiedToken:
    client_id: str
    scopes: frozenset[str]
    claims: Mapping[str, Any]


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    client_id: str
    scopes: frozenset[str]


def unauthenticated(detail: str) -> ProblemError:
    metrics.AUTH_FAILURES.labels(reason="unauthenticated").inc()
    return ProblemError(
        401,
        "UNAUTHENTICATED",
        "Authentication required",
        detail,
        headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
    )


def forbidden(detail: str) -> ProblemError:
    return ProblemError(403, "FORBIDDEN", "Forbidden", detail)


class KeysUnavailableError(Exception):
    """Signing keys can't be determined right now: the caller should retry, not be rejected."""


class KeySource(Protocol):
    def needs_io(self, kid: str) -> bool: ...

    def get(self, kid: str) -> jwt.PyJWK | None: ...


class StaticKeys:
    def __init__(self, jwks: Mapping[str, Any]) -> None:
        self._keys: dict[str, jwt.PyJWK] = {}
        for key in jwt.PyJWKSet.from_dict(dict(jwks)).keys:
            if key.key_id is None:
                raise ValueError("Every JWKS key must have a kid")
            self._keys[key.key_id] = key

    @classmethod
    def from_file(cls, path: Path) -> StaticKeys:
        return cls(json.loads(path.read_text()))

    @property
    def keys_by_id(self) -> dict[str, jwt.PyJWK]:
        return dict(self._keys)

    def needs_io(self, kid: str) -> bool:
        return False

    def get(self, kid: str) -> jwt.PyJWK | None:
        return self._keys.get(kid)


def _fetch_jwks(url: str, timeout_s: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310 - configured URL
        data: dict[str, Any] = json.loads(resp.read())
        return data


def _in_background(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, name="jwks-refresh", daemon=True).start()


class RemoteKeys:
    """JWKS fetched from the issuer and cached in memory.

    Key lookup happens before the signature is checked, so the `kid` comes from an
    unauthenticated caller. An unknown kid therefore triggers at most one fetch per
    `min_refresh_s` (attempts that fail count too); every other unknown kid in between is
    rejected without network I/O. A lock makes concurrent misses share one fetch. Rotation is
    still picked up within `min_refresh_s`.

    Revocation is handled by age: once the keys are older than `max_age_s`, the next lookup of a
    known kid starts a refresh in the background and is answered from the cache, so no request
    waits on the issuer. A revoked key therefore stops validating shortly after `max_age_s`. If
    the issuer is unreachable the cached keys keep working (availability over revocation), and
    attempts stay limited to one per `min_refresh_s`. An explicitly empty key set is an answer,
    not a failure: it clears the cache.

    An unknown kid arriving during a download may be the new key, so it waits up to `wait_s` for
    the result. KeysUnavailableError means the answer isn't known (still downloading, or no
    download has ever succeeded); only a settled "not in the set" returns None.
    """

    def __init__(
        self,
        url: str,
        *,
        min_refresh_s: float = 60.0,
        max_age_s: float = 300.0,
        timeout_s: float = 3.0,
        wait_s: float = 0.1,
        fetch: Callable[[str, float], Mapping[str, Any]] = _fetch_jwks,
        clock: Callable[[], float] = time.monotonic,
        spawn: Callable[[Callable[[], None]], None] = _in_background,
    ) -> None:
        self._url = url
        self._min_refresh_s = min_refresh_s
        self._max_age_s = max_age_s
        self._timeout_s = timeout_s
        self._wait_s = wait_s
        self._fetch = fetch
        self._clock = clock
        self._spawn = spawn
        self._keys: dict[str, jwt.PyJWK] = {}
        self._loaded_at: float | None = None
        self._last_attempt: float | None = None
        self._lock = threading.Lock()  # Guards the decision to start a download, never the I/O
        self._idle = threading.Event()  # Clear while a download is in flight
        self._idle.set()

    def needs_io(self, kid: str) -> bool:
        """Whether get(kid) could block. Stale known keys refresh in the background."""
        return kid not in self._keys and (not self._idle.is_set() or self._refresh_due())

    def get(self, kid: str) -> jwt.PyJWK | None:
        key = self._keys.get(kid)
        if key is not None:
            if self._stale() and self._begin_refresh():
                self._spawn(self._refresh)
            return key
        if self._begin_refresh():
            self._refresh()
        elif not self._idle.wait(self._wait_s if not self._idle.is_set() else 0):
            if kid in self._keys:
                return self._keys[kid]
            raise KeysUnavailableError
        key = self._keys.get(kid)
        if key is None and self._loaded_at is None:
            raise KeysUnavailableError  # Nothing has ever loaded, so this kid proves nothing.
        return key

    def _begin_refresh(self) -> bool:
        with self._lock:
            if not self._idle.is_set() or not self._refresh_due():
                return False
            self._last_attempt = self._clock()
            self._idle.clear()
            return True

    def _refresh(self) -> None:
        try:
            jwks = self._fetch(self._url, self._timeout_s)
            # PyJWT rejects an empty set, but an issuer that revoked every key must be believed.
            self._keys = {} if jwks.get("keys") == [] else StaticKeys(jwks).keys_by_id
            self._loaded_at = self._clock()
        except Exception as exc:  # noqa: BLE001 - keep serving cached keys
            logger.warning("JWKS refresh failed (%s)", type(exc).__name__)
        finally:
            self._idle.set()

    def _stale(self) -> bool:
        return self._loaded_at is None or self._clock() - self._loaded_at >= self._max_age_s

    def _refresh_due(self) -> bool:
        return (
            self._last_attempt is None or self._clock() - self._last_attempt >= self._min_refresh_s
        )


class TokenVerifier:
    def __init__(self, *, issuer: str, audience: str, keys: KeySource, leeway_s: int = 30) -> None:
        self._issuer = issuer
        self._audience = audience
        self._leeway_s = leeway_s
        self.keys = keys

    def verify(self, token: str) -> VerifiedToken:
        """Validate signature and standard claims; raise ProblemError(401) on any failure."""
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise unauthenticated("Malformed token") from exc

        alg, kid = header.get("alg"), header.get("kid")
        if alg not in ALLOWED_ALGORITHMS:
            raise unauthenticated("Unsupported token algorithm")
        try:
            key = self.keys.get(kid) if isinstance(kid, str) else None
        except KeysUnavailableError as exc:
            # Not the caller's fault, and not a rejection: merchants fail open and may retry.
            raise ProblemError(
                503,
                "SERVICE_UNAVAILABLE",
                "Service unavailable",
                "Signing keys are not available",
                retryable=True,
            ) from exc
        if key is None:
            raise unauthenticated("Unknown signing key")
        # Bind the algorithm to the key type to prevent algorithm-substitution attacks.
        if key.algorithm_name != alg:
            raise unauthenticated("Token algorithm does not match signing key")

        try:
            claims = jwt.decode(
                token,
                key.key,
                algorithms=[alg],
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway_s,
                options={"require": list(REQUIRED_CLAIMS)},
            )
        except jwt.ExpiredSignatureError as exc:
            raise unauthenticated("Token expired") from exc
        except jwt.PyJWTError as exc:
            raise unauthenticated("Invalid token") from exc

        client_id = claims.get("client_id") or claims.get("azp") or claims.get("sub")
        if not isinstance(client_id, str) or not client_id:
            raise unauthenticated("Token has no client identity")
        return VerifiedToken(client_id=client_id, scopes=_scopes(claims), claims=claims)

    async def averify(self, token: str) -> VerifiedToken:
        # Only a JWKS download can block; keep that off the event loop. Cached keys verify
        # inline, because a thread hop per request adds latency and queues behind slower work.
        kid = _unverified_kid(token)
        if kid is not None and self.keys.needs_io(kid):
            return await asyncio.to_thread(self.verify, token)
        return self.verify(token)


def _unverified_kid(token: str) -> str | None:
    try:
        kid = jwt.get_unverified_header(token).get("kid")
    except jwt.PyJWTError:
        return None  # verify() rejects it without any key lookup
    return kid if isinstance(kid, str) else None


def _scopes(claims: Mapping[str, Any]) -> frozenset[str]:
    scope = claims.get("scope")
    if isinstance(scope, str):
        return frozenset(scope.split())
    scp = claims.get("scp")
    if isinstance(scp, list) and all(isinstance(s, str) for s in scp):
        return frozenset(scp)
    return frozenset()


class TenantResolver(Protocol):
    async def resolve(self, client_id: str) -> str | None: ...


class FileTenantResolver:
    def __init__(self, mapping: Mapping[str, str]) -> None:
        self._mapping = dict(mapping)

    @classmethod
    def from_file(cls, path: Path) -> FileTenantResolver:
        return cls(json.loads(path.read_text()))

    async def resolve(self, client_id: str) -> str | None:
        return self._mapping.get(client_id)


def bearer_token(request: Request) -> str:
    header = request.headers.get("authorization")
    if not header:
        raise unauthenticated("Missing bearer token")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise unauthenticated("Authorization header must use the Bearer scheme")
    return token.strip()


def require_scope(scope: str) -> Callable[[Request], Awaitable[Principal]]:
    """FastAPI dependency: authenticate the merchant, resolve its tenant, require a scope."""

    async def dependency(request: Request) -> Principal:
        # Authentication is part of the request: the scoring deadline is measured from here.
        request.state.started = time.perf_counter()
        verifier: TokenVerifier = request.app.state.token_verifier
        token = await verifier.averify(bearer_token(request))
        resolver: TenantResolver = request.app.state.tenants
        try:
            # Bounded: identity resolution must not consume the whole request deadline.
            async with asyncio.timeout(TENANT_LOOKUP_TIMEOUT_S):
                tenant_id = await resolver.resolve(token.client_id)
        except Exception as exc:
            logger.error("Tenant mapping unavailable (%s)", type(exc).__name__)
            raise ProblemError(
                503,
                "SERVICE_UNAVAILABLE",
                "Service unavailable",
                "Caller identity could not be resolved",
                retryable=True,
            ) from exc
        if tenant_id is None:
            logger.warning("Authenticated client has no tenant mapping")
            metrics.AUTH_FAILURES.labels(reason="unmapped_client").inc()
            raise forbidden("Client is not mapped to a tenant")
        if scope not in token.scopes:
            metrics.AUTH_FAILURES.labels(reason="missing_scope").inc()
            raise forbidden(f"Token lacks required scope {scope}")
        request.state.tenant_id = tenant_id
        return Principal(tenant_id=tenant_id, client_id=token.client_id, scopes=token.scopes)

    return dependency
