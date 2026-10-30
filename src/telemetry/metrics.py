"""Prometheus metrics (FR-12). No tenant, attempt or identifier labels: cardinality stays
bounded and no identifiers reach the metrics backend. Per-tenant views come from PostgreSQL."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.17, 0.2, 0.3, 0.5, 1.0)

SCORE_REQUESTS = Counter(
    "fraud_score_requests_total",
    # outcome: ml | fallback | replay | in_progress | unavailable | deadline | rejected
    "Scoring requests by outcome",
    ["outcome"],
)
SCORE_LATENCY = Histogram(
    "fraud_score_latency_seconds",
    "Server-side scoring request latency by outcome",
    ["outcome"],
    buckets=LATENCY_BUCKETS,
)
PRIMARY_FAILURES = Counter(
    "fraud_primary_failures_total", "Primary scoring failures by reason", ["reason"]
)
FALLBACK_FAILURES = Counter(
    "fraud_fallback_failures_total", "Fallback scoring failures by reason", ["reason"]
)
DEPENDENCY_LATENCY = Histogram(
    "fraud_dependency_latency_seconds",
    "Latency of dependency operations",
    ["dependency", "operation"],
    buckets=LATENCY_BUCKETS,
)
DEPENDENCY_ERRORS = Counter(
    "fraud_dependency_errors_total", "Dependency operation failures", ["dependency", "operation"]
)
OUTCOME_EVENTS = Counter(
    "fraud_outcome_events_total", "Outcome events received by result", ["result"]
)
WORKER_EVENTS = Counter(
    "fraud_worker_events_total", "Inbox events handled by the worker", ["result"]
)
INBOX_BACKLOG = Gauge(
    "fraud_inbox_backlog", "Pending inbox events at last poll", multiprocess_mode="max"
)
OUTBOX_BACKLOG = Gauge(
    "fraud_outbox_backlog", "Unpublished outbox rows at last poll", multiprocess_mode="max"
)
ARCHIVED_ROWS = Counter("fraud_archived_rows_total", "Rows exported to durable history")
AUTH_FAILURES = Counter(
    "fraud_auth_failures_total", "Rejected authentication or authorization attempts", ["reason"]
)
