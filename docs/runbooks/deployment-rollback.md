# Runbook: release and rollback

Releases are manual (`.github/workflows/deploy.yml`, protected environment). Each release is an
immutable image digest plus a pinned model manifest digest.

Rollback: re-run the deploy workflow with the previous image digest and previous
`FRAUD_MODEL_MANIFEST_SHA256`. Migrations are additive (expand/contract); a rollback never needs a
down-migration.

If a new release fails verification, `deploy.sh` restores the previous one, **but expect an
outage**: Compose replaces the running containers before the readiness check, so the previous
release does not keep serving meanwhile. Restoring can take the 60-second readiness loop (30 tries,
2 seconds apart) plus restart time. Merchants fail open during that window. Avoiding it needs a
start-first or blue-green deployment, which this single-host demo doesn't have.
