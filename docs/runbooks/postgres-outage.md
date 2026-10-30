# Runbook: PostgreSQL outage

**Symptoms:** `/health/ready` reports postgres unavailable; scoring answers
`503 SCORING_UNAVAILABLE` / `SERVICE_UNAVAILABLE`; outcome POSTs answer 503 (not acknowledged).

1. Merchants fail open; outcome senders must retry (events were *not* accepted, so nothing is lost
   on our side).
2. Restore the database. The API pool reconnects automatically; the worker resumes leases.
3. After recovery run reconciliation:
   `python -m src.events.reconcile --dsn <owner-dsn> --archive file:///.../archive`
   Expect `ok: true`. `IN_PROGRESS` attempts with expired leases are reclaimed by retries.
