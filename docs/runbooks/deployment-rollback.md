# Runbook: release and rollback

Releases are manual (`.github/workflows/deploy.yml`, protected environment). Each release is an
immutable image digest plus a pinned model manifest digest.

Rollback: re-run the deploy workflow with the previous image digest and previous
`FRAUD_MODEL_MANIFEST_SHA256`. Migrations are additive (expand/contract); a rollback never needs a
down-migration. If the new model fails verification, the new container never becomes ready and
the previous one keeps serving.
