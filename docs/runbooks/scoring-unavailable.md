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
   the same request; run `python -m src.events.reconcile` to confirm nothing is stuck `IN_PROGRESS`.
