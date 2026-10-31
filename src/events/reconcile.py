"""Data accounting (PRD 4.1 "no unaccounted accepted events"; T-203, T-504).

Every accepted attempt must be COMPLETED (with exactly one assessment), UNAVAILABLE, or
explicitly IN_PROGRESS; every accepted outcome event must be PROCESSED, PENDING or DEAD; every
committed archive batch must match its object's checksum and row count; an archive object with
neither a committed batch nor an in-flight export is an orphan and fails reconciliation.

    python -m src.events.reconcile --dsn ... --archive file:///path [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import psycopg
from psycopg.rows import dict_row

from src.events.archive import ArchiveStore, store_from_uri, verify_object


def _batch_of(key: str) -> str:
    name = key.rsplit("/", 1)[-1]
    return name.removesuffix(".manifest.json").removesuffix(".parquet")


def reconcile(conn: psycopg.Connection[Any], store: ArchiveStore | None) -> dict[str, Any]:
    q = conn.execute
    report: dict[str, Any] = {}
    report["attempts_by_status"] = {
        r["status"]: r["n"]
        for r in q("SELECT status, count(*) AS n FROM attempts GROUP BY status").fetchall()
    }
    report["assessments"] = q("SELECT count(*) AS n FROM assessments").fetchone()["n"]  # type: ignore[index]
    report["completed_without_assessment"] = q(
        "SELECT count(*) AS n FROM attempts a WHERE status = 'COMPLETED' AND NOT EXISTS "
        "(SELECT 1 FROM assessments s WHERE s.tenant_id = a.tenant_id "
        "AND s.payment_attempt_id = a.payment_attempt_id)"
    ).fetchone()["n"]  # type: ignore[index]
    report["assessment_without_completed_attempt"] = q(
        "SELECT count(*) AS n FROM assessments s JOIN attempts a USING (tenant_id, "
        "payment_attempt_id) WHERE a.status <> 'COMPLETED'"
    ).fetchone()["n"]  # type: ignore[index]
    report["outcome_events_by_status"] = {
        r["status"]: r["n"]
        for r in q("SELECT status, count(*) AS n FROM outcome_events GROUP BY status").fetchall()
    }
    report["outbox_unpublished"] = q(
        "SELECT count(*) AS n FROM outbox WHERE published_at IS NULL"
    ).fetchone()["n"]  # type: ignore[index]
    report["assessments_not_yet_archived"] = q(
        "SELECT count(*) AS n FROM assessments s WHERE NOT EXISTS (SELECT 1 FROM archive_ledger l "
        "WHERE l.tenant_id = s.tenant_id AND l.payment_attempt_id = s.payment_attempt_id)"
    ).fetchone()["n"]  # type: ignore[index]

    batches = q("SELECT batch_id, object_key, manifest_key, row_count, sha256 FROM archive_batches")
    batch_rows = list(batches.fetchall())
    report["archive_batches"] = len(batch_rows)
    report["archive_rows"] = sum(b["row_count"] for b in batch_rows)
    if store is not None:
        bad = [
            v
            for b in batch_rows
            if not all(
                (v := verify_object(store, b["manifest_key"]))[k]
                for k in ("sha256_ok", "row_count_ok")
            )
        ]
        known = {b["object_key"] for b in batch_rows} | {b["manifest_key"] for b in batch_rows}
        report["archive_verification_failures"] = bad
        # An export between its object writes and its commit is not an orphan.
        in_flight = {
            r["batch_id"]
            for r in q(
                "SELECT DISTINCT batch_id FROM outbox "
                "WHERE published_at IS NULL AND batch_id IS NOT NULL"
            ).fetchall()
        }
        report["archive_orphan_objects"] = [
            k for k in store.list() if k not in known and _batch_of(k) not in in_flight
        ]

    problems = []
    if report["completed_without_assessment"] or report["assessment_without_completed_attempt"]:
        problems.append("attempt/assessment lifecycle mismatch")
    if report["outcome_events_by_status"].get("DEAD"):
        problems.append("dead-lettered outcome events need review")
    if report.get("archive_verification_failures"):
        problems.append("archive objects fail verification")
    if report.get("archive_orphan_objects"):
        problems.append("archive contains objects with no batch")
    report["problems"] = problems
    report["ok"] = not problems
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True, help="Owner or worker DSN (sees all tenants)")
    parser.add_argument("--archive", help="Archive URI to verify, e.g. file:///archive")
    args = parser.parse_args()
    store = store_from_uri(args.archive) if args.archive else None
    with psycopg.connect(args.dsn, row_factory=dict_row) as conn:
        report = reconcile(conn, store)
    print(json.dumps(report, indent=2, default=str))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
