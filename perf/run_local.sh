#!/usr/bin/env bash
# Local load run against the compose stack (free; results reflect THIS machine only).
#   perf/run_local.sh <rate> <duration> [run-id]
set -euo pipefail
RATE=${1:-50}; DURATION=${2:-60s}; RUN_ID=${3:-local$(date +%s)}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TOKEN=$("$ROOT/.venv/Scripts/python" "$ROOT/scripts/dev_jwt.py" mint --keys "$ROOT/infra/local/secrets" \
        --client-id demo-merchant --scope fraud:score --ttl 7200)
mkdir -p "$ROOT/evidence/perf"
MSYS_NO_PATHCONV=1 docker run --rm --network fraud-platform-local_default \
  -v "$ROOT/perf/k6:/scripts:ro" -v "$ROOT/evidence/perf:/evidence" \
  -e BASE_URL=http://api:8000 -e TOKEN="$TOKEN" -e RATE="$RATE" -e DURATION="$DURATION" \
  -e RUN_ID="$RUN_ID" grafana/k6:0.55.0 run --quiet \
  --summary-export="/evidence/$RUN_ID-summary.json" /scripts/score.js || true
"$ROOT/.venv/Scripts/python" "$ROOT/perf/reconcile_run.py" \
  --dsn "postgresql://fraud:${POSTGRES_PASSWORD:-REMOVED-CREDENTIAL}@127.0.0.1:55432/fraud" \
  --run-id "$RUN_ID" --summary "$ROOT/evidence/perf/$RUN_ID-summary.json" \
  | tee "$ROOT/evidence/perf/$RUN_ID-reconcile.json"
