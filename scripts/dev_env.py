"""Generate local-only credentials for the compose stack. Nothing it writes is committed.

    python scripts/dev_env.py

* infra/local/.env: random database, pseudonymization and Grafana passwords (existing keys,
  such as the promoted model pin, are kept).
* infra/local/secrets/dev-users.json: dashboard test users for the dev OIDC provider, with
  random passwords listed in infra/local/secrets/dev-users.txt.

Idempotent: values that already exist are never regenerated, so a running stack keeps working.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path

LOCAL = Path(__file__).resolve().parents[1] / "infra" / "local"
GENERATED_KEYS = (
    "POSTGRES_PASSWORD",
    "FRAUD_APP_DB_PASSWORD",
    "FRAUD_WORKER_DB_PASSWORD",
    "FRAUD_PSEUDONYM_KEY",
    "GRAFANA_ADMIN_PASSWORD",
)
DEV_USERS = (
    ("admin@demo", "tenant-demo", ["admin"]),
    ("analyst@demo", "tenant-demo", ["analyst"]),
    ("viewer@demo", "tenant-demo", ["viewer"]),
    ("analyst@second", "tenant-second", ["analyst"]),
)


def ensure_env(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    present = {ln.split("=", 1)[0].strip() for ln in lines if "=" in ln}
    added = [k for k in GENERATED_KEYS if k not in present]
    lines += [f"{k}={secrets.token_urlsafe(32)}" for k in added]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return added


def ensure_dev_users(secrets_dir: Path) -> bool:
    users_path, listing = secrets_dir / "dev-users.json", secrets_dir / "dev-users.txt"
    if users_path.exists():
        return False
    secrets_dir.mkdir(parents=True, exist_ok=True)
    users, rows = [], []
    for username, tenant, roles in DEV_USERS:
        password = secrets.token_urlsafe(12)
        users.append(
            {
                "username": username,
                "tenant_id": tenant,
                "roles": roles,
                "password_sha256": hashlib.sha256(password.encode()).hexdigest(),
            }
        )
        rows.append(f"{username}\t{password}\t{tenant}\t{','.join(roles)}")
    users_path.write_text(json.dumps({"users": users}, indent=2), encoding="utf-8")
    listing.write_text(
        "username\tpassword\ttenant\troles\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    return True


def main() -> None:
    added = ensure_env(LOCAL / ".env")
    created = ensure_dev_users(LOCAL / "secrets")
    print(f"infra/local/.env: {'added ' + ', '.join(added) if added else 'up to date'}")
    print(
        "dashboard test users: "
        + ("created, see infra/local/secrets/dev-users.txt" if created else "already exist")
    )


if __name__ == "__main__":
    main()
