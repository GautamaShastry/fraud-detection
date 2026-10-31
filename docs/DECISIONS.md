# Decision Index

Status of implementation decisions referenced by `PRD.md` and `OPEN_DECISIONS.md`.

- **Approved:** the owner explicitly approved it.
- **Provisional:** the documented proposed default (or a narrowly scoped engineering choice) that
  the code uses now. The owner can change it; the "where" column says what changes with it.
- **Open:** nothing depends on it yet.

Approved Decisions 001–019 live in the owner's decision log, which is not in this repository.

## Owner decisions (OPEN_DECISIONS.md)

| ID | Decision | Status | Where |
| --- | --- | --- | --- |
| I-01 | Python 3.12 + FastAPI + Pydantic API; Python worker | **Approved 2026-10-28** | `src/` |
| I-02 | PostgreSQL inbox + transactional outbox; `SKIP LOCKED` worker leases; idempotent at-least-once effects | Provisional | `src/events/worker.py`, migration 0003 |
| I-03 | Commit the immutable assessment before returning 200 | **Approved 2026-10-28** | migration 0005 `fraud_complete_attempt` |
| I-04 | Local test JWT issuer (merchants) and test-only OIDC provider (dashboard); managed issuer for cloud | Provisional (local half built) | `scripts/dev_jwt.py`, `src/devauth/oidc.py` |
| I-05 | API contract: 200 / 202 / 409 / 503 / 504, read-by-attempt endpoint | Provisional; **owner review requested** | `contracts/openapi.yaml` |
| I-06 | Interpretable L2 logistic baseline + CPU gradient-boosted trees (scikit-learn HistGradientBoosting), in-process inference | Provisional | `ml/train.py`, `src/scoring/compiled.py` |
| I-07 | Decline threshold at the approved ≤0.5% validation FPR; verify threshold at 2% validation FPR | 0.5% approved; **2% verify is provisional** | `ml/train.py` |
| I-08 | Fallback never recommends decline (enforced in policy, engine, artifact loader and a DB CHECK) | Provisional | several |
| I-09 | Label precedence MANUAL_REVIEW > ISSUER > PROCESSOR > MARKETPLACE; same source: latest wins; retraction needs equal or higher authority; approval is never legitimacy; 60-day maturity for training negatives | Provisional | `src/storage/outcomes.py`, `ml/train.py` |
| I-10 | `occurred_at` at most 300 s in the future and 24 h old; velocity uses server receive time; 2 s attempt lease | Provisional | `src/api/settings.py` |
| I-11 | F01 = one Lua script: per-attempt dedupe key (48 h) + per-device sorted set of token last-seen times; window exclusive at exactly 5 min | Provisional (capacity not yet benchmarked at scale) | `src/features/velocity.py` |
| I-12 | Per-tenant HMAC-SHA256 pseudonyms for customer, device, token, IP and shipping identifiers | Provisional (key rotation not implemented) | `src/domain/identity.py` |
| I-13 | Parquet (zstd) + JSON manifest (count, SHA-256, outbox id range) per batch; archive ledger gates retention | Provisional | `src/events/archive.py` |
| I-14 | Minimal React UI without a build step; RBAC viewer < analyst < admin | Provisional | `src/dashboard/` |
| I-15 | Single EC2 host, ECR, S3, budget alarm; SSM deploys | Provisional, **not deployed** | `infra/aws/` |
| I-16 | False-decline guardrails | Open (dashboard reports the confirmed-legit decline-recommendation rate) | — |
| I-17 | `risk_score` = percentile rank against validation outputs, not a probability | Provisional | `src/scoring/model.py` |
| I-18 | Unique-attempt reconciliation (k6 counts vs. DB states; inbox/outbox/archive accounting) | Provisional | `perf/reconcile_run.py`, `src/events/reconcile.py` |

## Engineering decisions made during implementation

| Decision | Why | Where |
| --- | --- | --- |
| Recommendations compare raw model output with thresholds (not integer scores) | With percentile scores, a 0.5% FPR cut falls inside the top percentile and cannot be expressed as an integer | `ScoringPolicy` |
| Compiled tree inference, verified against scikit-learn to 1e-9 at load | scikit-learn single-row `predict_proba`: p50 114 ms with OpenMP threads; compiled: 1.1 ms | `src/scoring/compiled.py` |
| Claim and commit as single-round-trip SQL functions (SECURITY INVOKER, so RLS still applies) | Halves database round trips on the hot path | migration 0005 |
| No psycopg pipeline mode | A pipelined query cancelled by the deadline left pooled connections broken | `src/storage/db.py` |
| Archive objects written before taking locks; short commit transaction afterwards | Holding locks during slow object writes stalled API commits | `src/events/archive.py` |
| Redis circuit breaker: opens for 2 s after 5 consecutive failures; 80 ms per call; a request's own deadline does not count | A dead Redis should cost a few timeouts, not one per request, and one slow call must not move all traffic to the fallback | `src/features/velocity.py` |
| Tenant lookup bounded to 100 ms | Identity resolution must not consume the whole request deadline | `src/api/auth.py` |
| One 170 ms deadline from the start of authentication; primary gets what is left after reserving 40 ms for the fallback and 20 ms for the commit | Separate timers threw away good fallback results in exactly the overloaded conditions the fallback exists for. A primary that would have finished in its last 60 ms now degrades to the fallback when the claim was slow | `src/scoring/engine.py`, `src/api/scoring.py` |
| JWKS keys refresh in the background after 5 minutes; at most one download per minute; a failed refresh keeps the cached keys | Bounds how long a revoked key is trusted without making requests wait on the issuer. If the issuer is down, availability wins over revocation | `RemoteKeys` |
| Synthetic-only artifacts need an explicit demo-only approval | The deployable model's features do not exist in IEEE-CIS | `ml/promote.py` |
| Test-only OIDC provider instead of Keycloak locally | Keycloak's JVM did not fit the development machine's free memory | `src/devauth/oidc.py` |
| Retention purges only archived rows; outcome events only after their attempt | A live attempt's projection is never rebuilt from partial history | `purge_expired` |
