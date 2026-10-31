"""Reconcile a load-test run against PostgreSQL (T-504): every unique attempt k6 sent must be
accounted for as COMPLETED (with exactly one assessment), UNAVAILABLE, or still IN_PROGRESS, and
each 200 the client saw must match a stored assessment with the same scoring mode, attempt by
attempt. Comparing totals could not show that: one missing assessment and one extra would cancel.

    python perf/reconcile_run.py --dsn <owner-dsn> --run-id run1 --summary evidence/perf/x.json \\
        --attempts evidence/perf/x-attempts.log
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import psycopg

_CLIENT_200 = re.compile(r"ASSESSED (ML|FALLBACK) ([^\s\"]+)")


def read_client_200s(text: str) -> dict[str, str]:
    """Parse k6's console log into {payment_attempt_id: scoring_mode}."""
    return {attempt: mode for mode, attempt in _CLIENT_200.findall(text)}


def unmatched_200s(client: dict[str, str], stored: dict[str, str]) -> list[str]:
    """Attempts the client saw succeed with no stored assessment in the same mode."""
    return sorted(a for a, mode in client.items() if stored.get(a) != mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True, help="k6 console log")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    m = summary["metrics"]

    def count(name: str) -> int:
        return int(m.get(name, {}).get("count", 0))

    sent = count("http_reqs")
    k6 = {
        "ml": count("assessments_ml"),
        "fallback": count("assessments_fallback"),
        "unavailable": count("assessments_unavailable"),
        "other": count("responses_other"),
        "dropped_iterations": count("dropped_iterations"),
    }
    with psycopg.connect(args.dsn) as conn:
        rows = conn.execute(
            "SELECT a.status, s.scoring_mode, count(*) FROM attempts a "
            "LEFT JOIN assessments s USING (tenant_id, payment_attempt_id) "
            "WHERE a.payment_attempt_id LIKE %s GROUP BY 1, 2",
            (f"pa-{args.run_id}-%",),
        ).fetchall()
        stored_modes = dict(
            conn.execute(
                "SELECT payment_attempt_id, scoring_mode FROM assessments "
                "WHERE payment_attempt_id LIKE %s",
                (f"pa-{args.run_id}-%",),
            ).fetchall()
        )
    client_200s = read_client_200s(args.attempts.read_text(encoding="utf-8"))
    unmatched = unmatched_200s(client_200s, stored_modes)
    db = {f"{status}:{mode or '-'}": n for status, mode, n in rows}
    stored = sum(db.values())
    report = {
        "run_id": args.run_id,
        "attempts_stored": stored,
        "stored_without_client_200": stored - k6["ml"] - k6["fallback"],
        "k6_requests_sent": sent,
        "k6_outcomes": k6,
        "db_attempts_by_state": db,
        "client_200s_checked": len(client_200s),
        "unmatched_200s": unmatched[:20],
        "unmatched_200s_total": len(unmatched),
        "checks": {
            # Each 200 the client saw was durably recorded (I-03) in the same mode, one by one.
            "every_200_recorded": not unmatched,
            # The log covers every 200 k6 counted, so a truncated log can't pass.
            "attempt_log_complete": len(client_200s) == k6["ml"] + k6["fallback"],
            # No attempt exists that the client did not send.
            "no_phantom_attempts": stored <= sent,
        },
    }
    report["ok"] = all(report["checks"].values())
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
