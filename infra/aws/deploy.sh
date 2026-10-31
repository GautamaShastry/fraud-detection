#!/usr/bin/env bash
# Host-side release with automatic rollback (T-502). Installed at /opt/fraud/deploy.sh and run by
# the deploy workflow via SSM:   deploy.sh <registry/repo@sha256:...> <model-manifest-sha256> <s3-uri>
# If the new release does not report ready, the previous one is restored. Compose replaces the
# running containers first, so expect an outage during a failed release (see the rollback runbook).
set -euo pipefail
IMAGE=$1
MODEL_SHA=$2
MODEL_S3=$3
cd /opt/fraud
[[ "$IMAGE" =~ ^[a-z0-9][a-z0-9./:_-]*@sha256:[0-9a-f]{64}$ ]] || { echo "refusing a non-digest image reference"; exit 1; }

previous=""
[ -f current-release.env ] && previous=current-release.env

aws ecr get-login-password | docker login --username AWS --password-stdin "${IMAGE%%/*}"
docker pull "$IMAGE"

rm -rf models/next && mkdir -p models/next
aws s3 cp --recursive "$MODEL_S3" models/next/
actual=$(sha256sum models/next/manifest.json | cut -d' ' -f1)
[ "$actual" = "$MODEL_SHA" ] || { echo "model manifest digest mismatch"; exit 1; }

printf 'FRAUD_IMAGE=%s\nFRAUD_MODEL_MANIFEST_SHA256=%s\n' "$IMAGE" "$MODEL_SHA" > next-release.env

# Migrations are additive (expand/contract), so applying them first is safe for the old release.
docker compose --env-file next-release.env -f compose.prod.yaml run --rm migrate

rm -rf models/current.prev
if [ -d models/current ]; then mv models/current models/current.prev; fi
mv models/next models/current
docker compose --env-file next-release.env -f compose.prod.yaml up -d api worker

for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8000/health/ready >/dev/null; then
    mv next-release.env current-release.env
    echo "release ready: $IMAGE (model $MODEL_SHA)"
    exit 0
  fi
  sleep 2
done

echo "new release did not become ready; rolling back" >&2
rm -rf models/current
if [ -d models/current.prev ]; then mv models/current.prev models/current; fi
if [ -n "$previous" ]; then
  docker compose --env-file "$previous" -f compose.prod.yaml up -d api worker
fi
exit 1
