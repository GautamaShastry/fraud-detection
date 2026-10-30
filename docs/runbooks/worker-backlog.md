# Runbook: outcome worker backlog / dead letters

**Alerts:** `InboxBacklogGrowing`, `DeadLetteredEvents`, `OutboxNotDraining`.

1. Is the worker running? `docker compose ... logs worker`. It logs `Worker iteration failed`.
2. Backlog with a healthy worker: scale workers (SKIP LOCKED makes this safe) or raise batch size.
3. Dead letters: `SELECT tenant_id, event_id, last_error FROM outcome_events WHERE status='DEAD'`.
   Fix the cause, then requeue: `UPDATE outcome_events SET status='PENDING', delivery_attempts=0,
   lease_until=NULL WHERE status='DEAD' AND event_id = ...`. Reprocessing is idempotent.
4. Outbox not draining: archive store unreachable. Retention purge deliberately waits for archival,
   so operational data grows until export resumes; nothing is deleted unarchived.
