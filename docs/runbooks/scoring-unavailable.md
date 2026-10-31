# Runbook: scoring unavailable (503/504 to merchants)

**Alert:** `ScoringUnavailableRate`. **Impact:** merchants fail open: checkout continues to ordinary
payment authorization *without* fraud screening. Auth is never bypassed.

1. Grafana "Scoring requests by outcome": is it `unavailable` (no scorer / both models failed /
   store down) or `deadline` (over 170 ms)?
2. `GET /health/ready`: which dependency reports `unavailable`? Only `postgres` fails readiness (503);
   `redis` being down reports `"status": "degraded"` with a 200 and the instance stays in service.
   - `postgres` → [Postgres outage](postgres-outage.md). Without the store, no assessment can be
     recorded, so none is returned (I-03).
   - `redis` → scoring degrades to FALLBACK rather than unavailable; see [Redis outage](redis-outage.md).
3. API logs (JSON): `Primary scoring failed (...)` / `Fallback scoring failed (...)` give the
   reason. `ArtifactError` at startup means the model failed verification and the container exited.
4. Deadline exceeded: check "Dependency latency p95". Database round trips are the usual cause.
5. Recovery check: unavailable attempts are marked `UNAVAILABLE` and are re-scored on retry with
   the same request. Run `python -m src.events.reconcile --dsn <owner or worker DSN>`. Its `ok` does
   not cover `IN_PROGRESS`, so also read `attempts_by_status`, and check the lease age directly:
   `SELECT count(*) FROM attempts WHERE status = 'IN_PROGRESS' AND lease_expires_at < now()`. Expired
   leases are harmless (a retry takes them over) and are deleted at retention.
