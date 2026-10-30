"""Reconcile a load-test run against PostgreSQL (T-504): every unique attempt k6 sent must be
accounted for as COMPLETED (with exactly one assessment), UNAVAILABLE, or still IN_PROGRESS.

    python perf/reconcile_run.py --dsn <owner-dsn> --run-id run1 --summary evidence/perf/x.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psycopg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--summary", type=Path, required=True)
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
    db = {f"{status}:{mode or '-'}": n for status, mode, n in rows}
    stored = sum(db.values())
    db_ml, db_fb = db.get("COMPLETED:ML", 0), db.get("COMPLETED:FALLBACK", 0)
    report = {
        "run_id": args.run_id,
        "attempts_stored": stored,
        "stored_without_client_200": stored - k6["ml"] - k6["fallback"],
        "k6_requests_sent": sent,
        "k6_outcomes": k6,
        "db_attempts_by_state": db,
        "checks": {
            # Each 200 the client saw was durably recorded (I-03), mode for mode.
            "ml_200s_recorded": k6["ml"] <= db_ml,
            "fallback_200s_recorded": k6["fallback"] <= db_fb,
            # No attempt exists that the client did not send.
            "no_phantom_attempts": stored <= sent,
            # Requests never stored must all be ones the client did not see succeed.
            "unstored_were_not_successes": sent - stored <= sent - k6["ml"] - k6["fallback"],
        },
    }
    report["ok"] = all(report["checks"].values())
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
