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


class KeySource(Protocol):
    remote: bool

    def get(self, kid: str) -> jwt.PyJWK | None: ...


class StaticKeys:
    remote = False

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

    def get(self, kid: str) -> jwt.PyJWK | None:
        return self._keys.get(kid)


def _fetch_jwks(url: str, timeout_s: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310 - configured URL
        data: dict[str, Any] = json.loads(resp.read())
        return data


class RemoteKeys:
    """JWKS fetched from the issuer and cached in memory.

    Key lookup happens before the signature is checked, so the `kid` comes from an
    unauthenticated caller. An unknown kid therefore triggers at most one fetch per
    `min_refresh_s` (attempts that fail count too); every other unknown kid in between is
    rejected without network I/O. A lock makes concurrent misses share one fetch. Rotation is
    still picked up within `min_refresh_s`.
    """

    remote = True

    def __init__(
        self,
        url: str,
        *,
        min_refresh_s: float = 60.0,
        timeout_s: float = 3.0,
        fetch: Callable[[str, float], Mapping[str, Any]] = _fetch_jwks,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._url = url
        self._min_refresh_s = min_refresh_s
        self._timeout_s = timeout_s
        self._fetch = fetch
        self._clock = clock
        self._keys: dict[str, jwt.PyJWK] = {}
        self._last_attempt: float | None = None
        self._lock = threading.Lock()

    def get(self, kid: str) -> jwt.PyJWK | None:
        key = self._keys.get(kid)
        if key is not None or not self._refresh_due():
            return key
        with self._lock:
            # Another thread may have refreshed while this one waited for the lock.
            if kid in self._keys or not self._refresh_due():
                return self._keys.get(kid)
            self._last_attempt = self._clock()
            try:
                self._keys = StaticKeys(self._fetch(self._url, self._timeout_s)).keys_by_id
            except Exception as exc:  # noqa: BLE001 - keep serving cached keys
                logger.warning("JWKS refresh failed (%s)", type(exc).__name__)
            return self._keys.get(kid)

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
        key = self.keys.get(kid) if isinstance(kid, str) else None
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
        # Remote key sources may block on a JWKS fetch; keep that off the event loop.
        if self.keys.remote:
            return await asyncio.to_thread(self.verify, token)
        return self.verify(token)


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
        verifier: TokenVerifier = request.app.state.token_verifier
        token = await verifier.averify(bearer_token(request))
        resolver: TenantResolver = request.app.state.tenants
        try:
            # Bounded: identity resolution must not consume the merchant's checkout budget.
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
