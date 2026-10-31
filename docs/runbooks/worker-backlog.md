# Runbook: outcome worker backlog / dead letters

**Alerts:** `InboxBacklogGrowing`, `DeadLetteredEvents`, `OutboxNotDraining`.

1. Is the worker running? `docker compose ... logs worker`. It logs `Worker step <step> failed`; see the `WorkerStepFailing` alert.
2. Backlog with a healthy worker: scale workers (SKIP LOCKED makes this safe) or raise batch size.
3. Dead letters: `SELECT tenant_id, event_id, last_error FROM outcome_events WHERE status='DEAD'`.
   Fix the cause, then requeue one event in a transaction, and commit only if it reports `UPDATE 1`.
   An `event_id` is unique only within a tenant, so always filter on both:
   `BEGIN; UPDATE outcome_events SET status='PENDING', delivery_attempts=0, lease_until=NULL WHERE
   status='DEAD' AND tenant_id = ... AND event_id = ...; COMMIT;`. Reprocessing is idempotent.
4. Outbox not draining: archive store unreachable. Retention purge deliberately waits for archival,
   so operational data grows until export resumes; nothing is deleted unarchived.
