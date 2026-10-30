"""Apply SQL migrations in order, once each. Runs as the schema owner, never as a runtime role.

python -m src.storage.migrate --dsn postgresql://owner@host/db [--seed-dev]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import psycopg
from psycopg import sql

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
RUNTIME_ROLES = {"fraud_app": "FRAUD_APP_DB_PASSWORD", "fraud_worker": "FRAUD_WORKER_DB_PASSWORD"}

# Test-only tenants and OAuth clients for the local stack. Never applied outside --seed-dev.
DEV_TENANTS = {"tenant-demo": "Demo Marketplace", "tenant-second": "Second Marketplace"}
DEV_CLIENTS = {"demo-merchant": "tenant-demo", "second-merchant": "tenant-second"}


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply pending migrations; return the names applied."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
    )
    conn.commit()
    applied = {r[0] for r in conn.execute("SELECT name FROM schema_migrations")}
    newly = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.name in applied:
            continue
        with conn.transaction():
            # Serialize concurrent migrators.
            conn.execute("SELECT pg_advisory_xact_lock(727274)")
            if conn.execute(
                "SELECT 1 FROM schema_migrations WHERE name = %s", (path.name,)
            ).fetchone():
                continue
            conn.execute(path.read_text())  # trusted migration files
            conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
        logger.info("Applied %s", path.name)
        newly.append(path.name)
    return newly


def set_role_passwords(conn: psycopg.Connection) -> None:
    for role, env in RUNTIME_ROLES.items():
        password = os.environ.get(env)
        if password:
            conn.execute(
                sql.SQL("ALTER ROLE {} PASSWORD {}").format(
                    sql.Identifier(role), sql.Literal(password)
                )
            )
    conn.commit()


def seed_dev(conn: psycopg.Connection) -> None:
    with conn.transaction():
        for tenant_id, name in DEV_TENANTS.items():
            conn.execute(
                "INSERT INTO tenants (tenant_id, display_name) VALUES (%s, %s) "
                "ON CONFLICT DO NOTHING",
                (tenant_id, name),
            )
        for client_id, tenant_id in DEV_CLIENTS.items():
            conn.execute(
                "INSERT INTO api_clients (client_id, tenant_id) VALUES (%s, %s) "
                "ON CONFLICT DO NOTHING",
                (client_id, tenant_id),
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default=os.environ.get("FRAUD_OWNER_DSN"))
    parser.add_argument("--seed-dev", action="store_true")
    args = parser.parse_args()
    if not args.dsn:
        parser.error("--dsn or FRAUD_OWNER_DSN is required")
    with psycopg.connect(args.dsn) as conn:
        applied = migrate(conn)
        set_role_passwords(conn)
        if args.seed_dev:
            seed_dev(conn)
    logger.info("Migrations complete (%d applied)", len(applied))


if __name__ == "__main__":
    main()
