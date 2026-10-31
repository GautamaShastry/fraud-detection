# Runbook: Redis outage

**Symptoms:** `FallbackRateHigh`, `fraud_primary_failures_total{reason="ConnectionError"|"timeout"}`.

Expected behavior: the primary path needs F01 from Redis, so every request degrades to the
validated FALLBACK scorer (no velocity features, never recommends decline). Assessments are
still recorded; responses say `"scoring_mode": "FALLBACK"`. `GET /health/ready` returns 200 with
`"status": "degraded"`, so instances stay in service.

1. `docker compose -f infra/local/compose.yaml ps redis` / restart it.
2. No data repair is needed: F01 windows are 5 minutes and rebuild on their own. Attempts scored
   during the outage keep their FALLBACK assessments (immutable).
3. Note: retried attempts are deduplicated by attempt ID once Redis is back, so counters are not
   double-incremented. Dedupe keys lost with the outage mean a retry *of an attempt made during
   the outage* counts once more — bounded by the 5-minute window.
