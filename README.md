# Marketplace Fraud Detection Platform

A fraud-risk assessment API for e-commerce marketplaces. Before its payment-authorization
decision, a marketplace calls `POST /api/v1/fraud/score` and gets back a 0–100 risk score
(a rank, not a probability), a recommendation (`PROCEED` / `VERIFY` / `RECOMMEND_DECLINE`),
evidence signals, and model and policy versions. The marketplace makes the final decision. If
scoring is unavailable, checkout continues to ordinary authorization.

This is a learning/portfolio prototype with no customers. **[docs/REPORT.md](docs/REPORT.md)**
lists what is measured, what was missed, and what is untested. No cloud resources have been
used.

## What's here

| Area | Where |
| --- | --- |
| API contract | `contracts/openapi.yaml` |
| Scoring API: auth, tenancy, idempotency, deadlines, fallback | `src/api/`, `src/scoring/`, `src/features/` |
| PostgreSQL schema with row-level security, SQL functions | `src/storage/` (migrations 0001–0005) |
| Outcome inbox worker, projections, Parquet archive, retention, reconciliation | `src/events/` |
| Analyst dashboard (OIDC + RBAC) and checkout demo | `src/dashboard/`, `src/demo/` |
| Telemetry: Prometheus, OpenTelemetry, JSON logs | `src/telemetry/`, `infra/local/observability/` |
| ML lifecycle: IEEE-CIS audit and benchmark, simulator, training, promotion, SageMaker job | `ml/` |
| Load (k6), reconciliation, chaos | `perf/` |
| CI/CD and AWS (written, not deployed) | `.github/workflows/`, `infra/aws/` |
| Decisions, runbooks, report | `docs/` |

## Run locally (no AWS needed)

Requires Python 3.12 and Docker.

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; .venv/bin/activate elsewhere
pip install -c requirements-lock.txt -e ".[dev,ml]"

docker compose -f infra/local/compose.yaml up --build --wait
python scripts/smoke_local.py --keys infra/local/secrets
```

Without a promoted model the API uses a **labeled test stub** (`model_version:
stub-deterministic-0`). To serve the synthetic demo model:

```bash
python -m ml.train --seed 7                                  # writes models/<version>/
python -m ml.promote models/<version> --approver "<you>" --synthetic-demo-only
docker compose -f infra/local/compose.yaml up -d api         # picks up infra/local/.env
```

| URL | What |
| --- | --- |
| http://localhost:8000 | Scoring API (`/health/ready`, `/metrics`) |
| http://localhost:8000/dashboard/ | Dashboard. Test users: `analyst@demo` / `REMOVED-CREDENTIAL`, `admin@demo` / `REMOVED-CREDENTIAL`, `viewer@demo` / `REMOVED-CREDENTIAL`, `analyst@second` / `REMOVED-CREDENTIAL` |
| http://localhost:8090 | Checkout demo (synthetic; try "simulate outage") |
| `--profile observability` | Prometheus :9090, Grafana :3000, Jaeger :16686 (set `OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318`) |

Call the API yourself:

```bash
TOKEN=$(python scripts/dev_jwt.py mint --keys infra/local/secrets --client-id demo-merchant --scope fraud:score)
curl -s -X POST http://127.0.0.1:8000/api/v1/fraud/score \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"transaction_id":"o-1","payment_attempt_id":"pa-1","idempotency_key":"idem-0001",
       "amount_minor":4599,"currency":"USD","occurred_at":"<now, UTC, ...Z>",
       "device_id":"dev-1","payment_method_token":"tok-1"}'
```

## Tests and checks

```bash
ruff check . && ruff format --check . && mypy && openapi-spec-validator contracts/openapi.yaml
pytest                      # integration tests use the compose Postgres/Redis (55432/56379)
bash perf/run_local.sh 50 60s myrun          # k6 load run + reconciliation (local, free)
python perf/chaos_local.py --keys infra/local/secrets
```

On Windows with Anaconda, run pytest from the project venv (`.venv/Scripts/python -m pytest`):
a global `pytest-asyncio` from Anaconda breaks collection.

## ML data

The IEEE-CIS dataset is licensed and is never committed. Put the CSVs in `data/raw/ieee-cis/`
(git-ignored), then run `python -m ml.audit_ieee` and `python -m ml.ieee_benchmark`.

## Paid resources

SageMaker training (`ml/sagemaker_job.py`, `ml-train.yml`), image release, EC2 deployment and S3
archival are written but have **never been run**. Each requires explicit owner approval and a
cost estimate first; see `infra/aws/README.md`.
