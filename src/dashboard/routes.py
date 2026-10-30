"""Tenant-scoped analyst dashboard API and static UI (FR-10, T-401, T-402).

Users sign in through OIDC (authorization code + PKCE). Access tokens are validated for issuer,
audience, expiry and signature; the tenant comes from the token's `tenant_id` claim and the role
from `roles`. Every query runs under PostgreSQL RLS for that tenant. Each data access is written
to dashboard_audit.

RBAC (I-14 provisional):
  viewer   metrics and attempt list
  analyst  + attempt detail (assessment evidence, outcome events)
  admin    + configuration and audit trail

Label-based metrics count only attempts with a confirmed label and always show the unlabeled
count; an attempt with no label is never treated as legitimate.
"""

from __future__ import annotations

import base64
import binascii
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.auth import RemoteKeys, StaticKeys, TokenVerifier, bearer_token, forbidden
from src.api.problems import ProblemError
from src.api.schemas import ExternalId
from src.api.settings import Settings
from src.domain.models import format_utc
from src.storage.attempts import assessment_from_row
from src.storage.db import Conn, tenant_tx
from src.telemetry import metrics

logger = logging.getLogger(__name__)

ROLE_RANK = {"viewer": 0, "analyst": 1, "admin": 2}
STATIC_DIR = Path(__file__).parent / "static"

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@dataclass(frozen=True)
class User:
    subject: str
    tenant_id: str
    role: str
    name: str


def require_role(minimum: str) -> Callable[[Request], Awaitable[User]]:
    async def dependency(request: Request) -> User:
        verifier: TokenVerifier = request.app.state.dashboard_verifier
        token = await verifier.averify(bearer_token(request))
        claims = token.claims
        tenant = claims.get("tenant_id")
        roles = claims.get("roles") or claims.get("realm_access", {}).get("roles", [])
        held = [r for r in roles if r in ROLE_RANK] if isinstance(roles, list) else []
        if not isinstance(tenant, str) or not tenant or not held:
            metrics.AUTH_FAILURES.labels(reason="dashboard_claims").inc()
            raise forbidden("Token lacks a tenant or dashboard role")
        role = max(held, key=ROLE_RANK.__getitem__)
        if ROLE_RANK[role] < ROLE_RANK[minimum]:
            metrics.AUTH_FAILURES.labels(reason="dashboard_role").inc()
            raise forbidden(f"Requires role {minimum}")
        return User(str(claims["sub"]), tenant, role, str(claims.get("preferred_username", "")))

    return dependency


def _pool(request: Request) -> Any:
    pool = request.app.state.pool
    if pool is None:
        raise ProblemError(503, "SERVICE_UNAVAILABLE", "Service unavailable", retryable=True)
    return pool


async def _audit(conn: Conn, user: User, action: str, target: str | None, now: datetime) -> None:
    await conn.execute(
        "INSERT INTO dashboard_audit (tenant_id, subject, role, action, target, at) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (user.tenant_id, user.subject, user.role, action, target, now),
    )


def _cursor(received_at: datetime, attempt: str) -> str:
    return base64.urlsafe_b64encode(f"{received_at.isoformat()}|{attempt}".encode()).decode()


def _parse_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        ts, attempt = base64.urlsafe_b64decode(cursor.encode()).decode().split("|", 1)
        return datetime.fromisoformat(ts), attempt
    except (ValueError, binascii.Error, UnicodeDecodeError) as exc:
        raise ProblemError(422, "VALIDATION_ERROR", "Invalid cursor") from exc


def _row(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "payment_attempt_id": r["payment_attempt_id"],
        "transaction_id": r["transaction_id"],
        "amount_minor": r["amount_minor"],
        "currency": r["currency"].strip(),
        "received_at": format_utc(r["received_at"]),
        "status": r["status"],
        "risk_score": r["risk_score"],
        "recommendation": r["recommendation"],
        "scoring_mode": r["scoring_mode"],
        "label": r["label"] or "UNLABELED",
    }


@router.get("/me")
async def me(user: Annotated[User, Depends(require_role("viewer"))]) -> dict[str, str]:
    return {
        "subject": user.subject,
        "name": user.name,
        "tenant_id": user.tenant_id,
        "role": user.role,
    }


@router.get("/attempts")
async def list_attempts(
    request: Request,
    user: Annotated[User, Depends(require_role("viewer"))],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: Annotated[str | None, Query(max_length=200)] = None,
    recommendation: Annotated[
        str | None, Query(pattern="^(PROCEED|VERIFY|RECOMMEND_DECLINE)$")
    ] = None,
    scoring_mode: Annotated[str | None, Query(pattern="^(ML|FALLBACK)$")] = None,
    status: Annotated[str | None, Query(pattern="^(IN_PROGRESS|COMPLETED|UNAVAILABLE)$")] = None,
    label: Annotated[
        str | None, Query(pattern="^(CONFIRMED_FRAUD|CONFIRMED_LEGITIMATE|UNLABELED)$")
    ] = None,
) -> dict[str, Any]:
    clauses, params = [], []
    if cursor:
        ts, attempt = _parse_cursor(cursor)
        clauses.append("(a.received_at, a.payment_attempt_id) < (%s, %s)")
        params += [ts, attempt]
    for column, value in (
        ("s.recommendation", recommendation),
        ("s.scoring_mode", scoring_mode),
        ("a.status", status),
    ):
        if value:
            clauses.append(f"{column} = %s")
            params.append(value)
    if label == "UNLABELED":
        clauses.append("o.label IS NULL")
    elif label:
        clauses.append("o.label = %s")
        params.append(label)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    # `where` is built only from fixed column names; every value is a bound parameter.
    sql = (
        "SELECT a.payment_attempt_id, a.transaction_id, a.amount_minor, a.currency, "  # noqa: S608
        "a.received_at, a.status, s.risk_score, s.recommendation, s.scoring_mode, o.label "
        "FROM attempts a LEFT JOIN assessments s USING (tenant_id, payment_attempt_id) "
        "LEFT JOIN attempt_outcomes o USING (tenant_id, payment_attempt_id) "
        f"{where} ORDER BY a.received_at DESC, a.payment_attempt_id DESC LIMIT %s"
    )
    params.append(limit + 1)
    async with tenant_tx(_pool(request), user.tenant_id) as conn:
        rows = list(await (await conn.execute(sql, params)).fetchall())
        await _audit(conn, user, "list_attempts", None, request.app.state.clock())
    page = rows[:limit]
    next_cursor = (
        _cursor(page[-1]["received_at"], page[-1]["payment_attempt_id"])
        if len(rows) > limit
        else None
    )
    return {"items": [_row(r) for r in page], "next_cursor": next_cursor}


@router.get("/attempts/{payment_attempt_id}")
async def attempt_detail(
    request: Request,
    payment_attempt_id: ExternalId,
    user: Annotated[User, Depends(require_role("analyst"))],
) -> dict[str, Any]:
    async with tenant_tx(_pool(request), user.tenant_id) as conn:
        attempt = await (
            await conn.execute(
                "SELECT * FROM attempts WHERE payment_attempt_id = %s", (payment_attempt_id,)
            )
        ).fetchone()
        if attempt is None:
            raise ProblemError(404, "NOT_FOUND", "Not found")
        assessment = await (
            await conn.execute(
                "SELECT * FROM assessments WHERE payment_attempt_id = %s", (payment_attempt_id,)
            )
        ).fetchone()
        events = await (
            await conn.execute(
                "SELECT event_id, event_type, source, occurred_at, received_at, status "
                "FROM outcome_events WHERE payment_attempt_id = %s ORDER BY occurred_at",
                (payment_attempt_id,),
            )
        ).fetchall()
        projection = await (
            await conn.execute(
                "SELECT label, label_source, authorization_status, chargeback_reported_at "
                "FROM attempt_outcomes WHERE payment_attempt_id = %s",
                (payment_attempt_id,),
            )
        ).fetchone()
        await _audit(conn, user, "view_attempt", payment_attempt_id, request.app.state.clock())

    def present(ref: str | None) -> bool:
        return ref is not None

    return {
        "attempt": {
            "payment_attempt_id": attempt["payment_attempt_id"],
            "transaction_id": attempt["transaction_id"],
            "amount_minor": attempt["amount_minor"],
            "currency": attempt["currency"].strip(),
            "occurred_at": format_utc(attempt["occurred_at"]),
            "received_at": format_utc(attempt["received_at"]),
            "status": attempt["status"],
            # Pseudonymous references are shown only as present/absent.
            "signals_present": {
                k: present(attempt[f"{k}_ref"])
                for k in ("customer", "device", "payment_method", "ip", "shipping")
            },
        },
        "assessment": None
        if assessment is None
        else {
            **assessment_from_row(assessment).to_response(),
            "features": assessment["features"],
            "scoring_latency_ms": assessment["scoring_latency_ms"],
        },
        "outcomes": [
            {
                **e,
                "occurred_at": format_utc(e["occurred_at"]),
                "received_at": format_utc(e["received_at"]),
            }
            for e in events
        ],
        "label": (projection or {}).get("label") or "UNLABELED",
        "label_source": (projection or {}).get("label_source"),
        "authorization_status": (projection or {}).get("authorization_status"),
    }


@router.get("/metrics")
async def dashboard_metrics(
    request: Request,
    user: Annotated[User, Depends(require_role("viewer"))],
    hours: Annotated[int, Query(ge=1, le=720)] = 24,
) -> dict[str, Any]:
    since = request.app.state.clock() - timedelta(hours=hours)
    async with tenant_tx(_pool(request), user.tenant_id) as conn:
        throughput = await (
            await conn.execute(
                "SELECT date_trunc('hour', received_at) AS hour, status, count(*) AS n "
                "FROM attempts WHERE received_at >= %s GROUP BY 1, 2 ORDER BY 1",
                (since,),
            )
        ).fetchall()
        modes = await (
            await conn.execute(
                "SELECT scoring_mode, recommendation, count(*) AS n FROM assessments "
                "WHERE evaluated_at >= %s GROUP BY 1, 2",
                (since,),
            )
        ).fetchall()
        latency = await (
            await conn.execute(
                "SELECT percentile_cont(ARRAY[0.5, 0.95, 0.99]) WITHIN GROUP "
                "(ORDER BY scoring_latency_ms) AS p FROM assessments WHERE evaluated_at >= %s",
                (since,),
            )
        ).fetchone()
        labels = await (
            await conn.execute(
                "SELECT coalesce(o.label, 'UNLABELED') AS label, s.recommendation, count(*) AS n "
                "FROM assessments s LEFT JOIN attempt_outcomes o "
                "USING (tenant_id, payment_attempt_id) "
                "WHERE s.evaluated_at >= %s GROUP BY 1, 2",
                (since,),
            )
        ).fetchall()
        await _audit(conn, user, "view_metrics", f"{hours}h", request.app.state.clock())

    by_label: dict[str, dict[str, int]] = {}
    for r in labels:
        by_label.setdefault(r["label"], {})[r["recommendation"]] = r["n"]
    fraud = by_label.get("CONFIRMED_FRAUD", {})
    legit = by_label.get("CONFIRMED_LEGITIMATE", {})
    n_fraud, n_legit = sum(fraud.values()), sum(legit.values())
    p = (latency or {}).get("p") or [None, None, None]
    return {
        "window_hours": hours,
        "throughput": [
            {"hour": format_utc(r["hour"]), "status": r["status"], "count": r["n"]}
            for r in throughput
        ],
        "assessments": [
            {
                "scoring_mode": r["scoring_mode"],
                "recommendation": r["recommendation"],
                "count": r["n"],
            }
            for r in modes
        ],
        "server_scoring_latency_ms": {
            "p50": p[0],
            "p95": p[1],
            "p99": p[2],
            "note": "in-process scoring time; excludes network and DB",
        },
        "labels": {
            "confirmed_fraud": n_fraud,
            "confirmed_legitimate": n_legit,
            "unlabeled": sum(by_label.get("UNLABELED", {}).values()),
            "fraud_flagged_rate": (
                (fraud.get("VERIFY", 0) + fraud.get("RECOMMEND_DECLINE", 0)) / n_fraud
                if n_fraud
                else None
            ),
            "legit_decline_recommendation_rate": (
                legit.get("RECOMMEND_DECLINE", 0) / n_legit if n_legit else None
            ),
            "note": "Computed over confirmed labels only; unlabeled attempts are not assumed "
            "legitimate.",
        },
    }


@router.get("/admin/config")
async def admin_config(
    request: Request, user: Annotated[User, Depends(require_role("admin"))]
) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    engine = request.app.state.engine
    bundle = engine.bundle if engine else None
    return {
        "tenant_id": user.tenant_id,
        "scorer_mode": settings.scorer_mode,
        "model_version": bundle.primary.model_version if bundle else None,
        "policy_version": bundle.primary_policy.version if bundle else None,
        "fallback_model_version": (
            bundle.fallback.model_version if bundle and bundle.fallback else None
        ),
        "deadlines_ms": {
            "server": settings.server_deadline_s * 1000,
            "primary": settings.primary_budget_s * 1000,
            "fallback": settings.fallback_budget_s * 1000,
        },
        "clock_skew_s": {
            "max_future": settings.max_future_skew_s,
            "max_age": settings.max_attempt_age_s,
        },
    }


@router.get("/admin/audit")
async def admin_audit(
    request: Request,
    user: Annotated[User, Depends(require_role("admin"))],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    async with tenant_tx(_pool(request), user.tenant_id) as conn:
        rows = await (
            await conn.execute(
                "SELECT subject, role, action, target, at FROM dashboard_audit "
                "ORDER BY at DESC LIMIT %s",
                (limit,),
            )
        ).fetchall()
    return {"items": [{**r, "at": format_utc(r["at"])} for r in rows]}


def install(app: FastAPI, settings: Settings) -> None:
    if not settings.oidc_issuer or not settings.oidc_audience:
        raise ValueError("Dashboard requires FRAUD_OIDC_ISSUER and FRAUD_OIDC_AUDIENCE")
    keys = (
        StaticKeys.from_file(settings.oidc_jwks_path)
        if settings.oidc_jwks_path
        else RemoteKeys(settings.oidc_jwks_url or f"{settings.oidc_issuer.rstrip('/')}/jwks")
    )
    app.state.dashboard_verifier = TokenVerifier(
        issuer=settings.oidc_issuer,
        audience=settings.oidc_audience,
        keys=keys,
        leeway_s=settings.jwt_leeway_s,
    )
    app.include_router(router)

    @app.get("/dashboard/config.json", include_in_schema=False)
    async def dashboard_config() -> JSONResponse:
        return JSONResponse(
            {
                "issuer": settings.oidc_public_issuer or settings.oidc_issuer,
                "client_id": settings.oidc_client_id or settings.oidc_audience,
            }
        )

    app.mount("/dashboard", StaticFiles(directory=STATIC_DIR, html=True), name="dashboard")
