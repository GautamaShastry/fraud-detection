"""Runtime configuration, read from FRAUD_* environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRAUD_", frozen=True)

    # Merchant token validation (FR-01). JWKS from a trusted file (local issuer) or URL.
    jwt_issuer: str
    jwt_audience: str
    jwks_path: Path | None = None
    jwks_url: str | None = None
    jwt_leeway_s: int = 30

    # Trusted client -> tenant mapping: PostgreSQL api_clients (default) or a JSON file (tests).
    tenant_source: Literal["postgres", "file"] = "postgres"
    tenant_map_path: Path | None = None

    # Dependencies.
    postgres_dsn: str
    redis_url: str
    db_pool_min: int = 2
    db_pool_max: int = 20
    readiness_timeout_s: float = 1.0

    # Per-tenant HMAC key for identifier pseudonymization (I-12 provisional). >= 32 bytes.
    pseudonym_key: SecretStr

    # Scoring. "model" loads a promoted artifact; "stub" is the labeled deterministic test
    # stub; "none" answers every score request with an explicit unavailable response.
    scorer_mode: Literal["model", "stub", "none"] = "none"
    model_dir: Path | None = None
    model_manifest_sha256: str | None = None

    # Deadlines (PRD 4.1, provisional): 170 ms total server deadline.
    server_deadline_s: float = 0.170
    primary_budget_s: float = 0.110
    fallback_budget_s: float = 0.040
    commit_reserve_s: float = 0.020
    attempt_lease_s: float = 2.0

    # Request limits. Clock-skew bounds are provisional (I-10).
    max_body_bytes: int = 16_384
    max_future_skew_s: int = 300
    max_attempt_age_s: int = 86_400

    # Analyst dashboard (FR-10). OIDC access tokens carry roles and a tenant_id claim.
    dashboard_enabled: bool = False
    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    oidc_jwks_url: str | None = None
    oidc_jwks_path: Path | None = None
    oidc_client_id: str | None = None
    oidc_public_issuer: str | None = None  # Browser-facing issuer URL if it differs

    log_level: str = "INFO"
