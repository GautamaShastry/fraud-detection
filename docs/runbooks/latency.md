# Runbook: scoring latency

**Alert:** `ScoringLatencyP99High` (server-side; client latency comes from k6 results).

1. Grafana "Dependency latency p95": Postgres claim/commit vs Redis.
2. "Primary / fallback failures by reason" with `timeout` means the primary budget is hit (up to 110 ms, less if the claim was slow).
3. Model inference uses the compiled tree path (~1 ms). If an artifact falls back to
   `predict_proba`, latency rises sharply; check startup logs for the model type.
4. CPU saturation: native thread pools are pinned to 1 thread per process; scale processes.
