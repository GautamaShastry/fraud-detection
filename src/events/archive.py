"""Durable history export (FR-09, T-204; I-13 provisional: Parquet + manifest).

Each export batch drains a set of outbox rows into one Parquet object per (tenant, topic) with
a JSON manifest holding the row count and SHA-256. Rows are assigned to a batch in the database
*before* anything is written, and the object key derives only from that batch. The batch, its
ledger rows and the outbox "published" marks commit in one transaction *after* the objects are
written, so:

* a crash before commit leaves the rows assigned but unpublished; the next run exports exactly
  those rows again and overwrites the same keys, however much later it runs and whatever new
  rows have arrived since;
* an object with neither a committed batch nor an in-flight one is an orphan: reconciliation
  fails on it, and it is never trusted;
* retention may delete operational rows only once they appear in the archive ledger.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from src.storage.db import Conn, Pool
from src.telemetry import metrics

ARCHIVE_SCHEMA_VERSION = "archive-1.0.0"


class ArchiveStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def list(self, prefix: str = "") -> list[str]: ...


class FileArchiveStore:
    """Local filesystem store with atomic writes (tests and the local stack)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("Key escapes archive root")
        return path

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def list(self, prefix: str = "") -> list[str]:
        base = self.root
        if not base.exists():
            return []
        return sorted(
            p.relative_to(base).as_posix()
            for p in base.rglob("*")
            if p.is_file()
            and not p.name.startswith(".tmp-")
            and p.relative_to(base).as_posix().startswith(prefix)
        )


class S3ArchiveStore:
    """S3 store. Real S3 writes are opt-in (cost and IAM are owner-approved, PRD FR-09)."""

    def __init__(self, bucket: str, prefix: str = "") -> None:
        import boto3

        self._s3 = boto3.client("s3")
        self._bucket = bucket
        self._prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    def put(self, key: str, data: bytes) -> None:
        import base64

        digest = base64.b64encode(hashlib.sha256(data).digest()).decode()
        self._s3.put_object(
            Bucket=self._bucket,
            Key=self._key(key),
            Body=data,
            ChecksumSHA256=digest,
            ServerSideEncryption="AES256",
        )

    def get(self, key: str) -> bytes:
        body = self._s3.get_object(Bucket=self._bucket, Key=self._key(key))["Body"]
        return bytes(body.read())

    def list(self, prefix: str = "") -> list[str]:
        keys: list[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._key(prefix)):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                keys.append(key[len(self._prefix) + 1 :] if self._prefix else key)
        return sorted(keys)


def store_from_uri(uri: str) -> ArchiveStore:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme == "file":
        return FileArchiveStore(
            Path(urllib.parse.unquote(parsed.path.lstrip("/") if os.name == "nt" else parsed.path))
        )
    if parsed.scheme == "s3":
        return S3ArchiveStore(parsed.netloc, parsed.path)
    raise ValueError(f"Unsupported archive URI scheme: {parsed.scheme}")


def _parquet(rows: list[dict[str, Any]]) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buf, compression="zstd")
    return buf.getvalue()


@dataclass(frozen=True)
class ExportResult:
    batch_ids: list[str]
    rows: int


_ASSESSMENT_ROWS = """
SELECT s.assessment_id::text, s.tenant_id, s.payment_attempt_id, s.transaction_id, s.risk_score,
       s.recommendation, s.scoring_mode, s.risk_signals::text AS risk_signals,
       s.features::text AS features, s.model_version, s.policy_version, s.evaluated_at,
       s.scoring_latency_ms, a.amount_minor, a.currency, a.occurred_at, a.received_at,
       a.customer_ref, a.device_ref, a.payment_method_ref, a.ip_ref, a.shipping_ref,
       a.account_created_at
FROM assessments s JOIN attempts a USING (tenant_id, payment_attempt_id)
WHERE s.tenant_id = %s AND s.assessment_id = ANY(%s::uuid[])
"""

_OUTCOME_ROWS = """
SELECT tenant_id, event_id, payment_attempt_id, transaction_id, event_type, source, occurred_at,
       received_at, reason_code, processed_at
FROM outcome_events
WHERE tenant_id = %s AND event_id = ANY(%s::text[])
"""


async def _assigned_rows(conn: Conn) -> list[dict[str, Any]]:
    cur = await conn.execute(
        "SELECT id, tenant_id, topic, payload, created_at, batch_id FROM outbox "
        "WHERE published_at IS NULL AND batch_id IS NOT NULL ORDER BY id"
    )
    return list(await cur.fetchall())


async def _assign_batches(conn: Conn, limit: int) -> None:
    """Fix which rows form each batch, so that a retry exports the same rows under the same key."""
    async with conn.transaction():
        cur = await conn.execute(
            "SELECT id, tenant_id, topic FROM outbox WHERE published_at IS NULL "
            "AND batch_id IS NULL ORDER BY id LIMIT %s FOR UPDATE SKIP LOCKED",
            (limit,),
        )
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for row in await cur.fetchall():
            groups[(row["tenant_id"], row["topic"])].append(row["id"])
        for (tenant_id, topic), ids in groups.items():
            kind = topic.split(".")[0]
            batch_id = f"{kind}-{tenant_id}-{min(ids):012d}-{max(ids):012d}"
            await conn.execute(
                "UPDATE outbox SET batch_id = %s WHERE id = ANY(%s)", (batch_id, ids)
            )


async def export_batch(
    pool: Pool, store: ArchiveStore, now: datetime, limit: int = 200
) -> ExportResult:
    """Export up to `limit` unpublished outbox rows. Returns what was committed.

    Batches left unfinished by an earlier run are finished first; new rows are assigned only
    when there are none. Objects are written *before* any lock is taken; a short transaction then
    re-checks that the rows are still unpublished and records the batch. Slow storage therefore
    never holds database locks or a long transaction open. If another exporter is working on the
    same batch, both write the same keys and only one commits.
    """
    async with pool.connection() as conn:
        outbox = await _assigned_rows(conn)
        if not outbox:
            await _assign_batches(conn, limit)
            outbox = await _assigned_rows(conn)
        if not outbox:
            return ExportResult([], 0)

        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in outbox:
            groups[row["batch_id"]].append(row)

        staged = []
        for batch_id, items in sorted(groups.items()):
            tenant_id, topic = items[0]["tenant_id"], items[0]["topic"]
            ids = [r["id"] for r in items]
            if topic == "assessment.recorded":
                key_column = "assessment_id"
                keys = [r["payload"]["assessment_id"] for r in items]
                cur = await conn.execute(_ASSESSMENT_ROWS, (tenant_id, keys))
            elif topic == "outcome.processed":
                key_column = "event_id"
                keys = [r["payload"]["event_id"] for r in items]
                cur = await conn.execute(_OUTCOME_ROWS, (tenant_id, keys))
            else:
                raise ValueError(f"Unknown outbox topic {topic}")
            rows = sorted(await cur.fetchall(), key=lambda r: r[key_column])
            if len(rows) != len(items):
                raise RuntimeError(
                    f"Outbox references {len(items)} rows but {len(rows)} exist ({topic})"
                )
            kind = topic.split(".")[0]
            # The date comes from the rows, not the clock, so a retry on a later day keeps the key.
            date = min(r["created_at"] for r in items).astimezone(UTC)
            object_key = f"{kind}s/tenant={tenant_id}/date={date:%Y-%m-%d}/{batch_id}.parquet"
            manifest_key = object_key.removesuffix(".parquet") + ".manifest.json"
            data = _parquet(rows)
            digest = hashlib.sha256(data).hexdigest()
            manifest = {
                "schema_version": ARCHIVE_SCHEMA_VERSION,
                "batch_id": batch_id,
                "tenant_id": tenant_id,
                "topic": topic,
                "object_key": object_key,
                "row_count": len(rows),
                "sha256": digest,
                "outbox_id_range": [min(ids), max(ids)],
                "created_at": now.isoformat(),
            }
            # Durable write first, outside any transaction.
            store.put(object_key, data)
            store.put(manifest_key, json.dumps(manifest, indent=2, sort_keys=True).encode())
            staged.append((tenant_id, kind, batch_id, object_key, manifest_key, ids, rows, digest))

        batch_ids: list[str] = []
        total = 0
        for tenant_id, kind, batch_id, object_key, manifest_key, ids, rows, digest in staged:
            async with conn.transaction():
                cur = await conn.execute(
                    "SELECT id FROM outbox WHERE id = ANY(%s) AND published_at IS NULL "
                    "AND batch_id = %s FOR UPDATE SKIP LOCKED",
                    (ids, batch_id),
                )
                if len(await cur.fetchall()) != len(ids):
                    continue  # another exporter owns or already published these rows
                await conn.execute(
                    "INSERT INTO archive_batches (batch_id, object_key, manifest_key, row_count, "
                    "sha256, created_at) VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (batch_id) DO UPDATE SET sha256 = EXCLUDED.sha256, "
                    "row_count = EXCLUDED.row_count",
                    (batch_id, object_key, manifest_key, len(rows), digest, now),
                )
                if kind == "assessment":
                    await conn.execute(
                        "INSERT INTO archive_ledger (tenant_id, payment_attempt_id, batch_id, "
                        "archived_at) SELECT %s, unnest(%s::text[]), %s, %s "
                        "ON CONFLICT DO NOTHING",
                        (tenant_id, [r["payment_attempt_id"] for r in rows], batch_id, now),
                    )
                await conn.execute(
                    "UPDATE outbox SET published_at = %s, batch_id = %s WHERE id = ANY(%s)",
                    (now, batch_id, ids),
                )
            batch_ids.append(batch_id)
            total += len(rows)

    metrics.ARCHIVED_ROWS.inc(total)
    return ExportResult(batch_ids, total)


def verify_object(store: ArchiveStore, manifest_key: str) -> dict[str, Any]:
    """Check an archived object against its manifest (count and checksum)."""
    manifest = json.loads(store.get(manifest_key))
    data = store.get(manifest["object_key"])
    actual = hashlib.sha256(data).hexdigest()
    rows = pq.read_table(io.BytesIO(data)).num_rows
    return {
        "batch_id": manifest["batch_id"],
        "sha256_ok": actual == manifest["sha256"],
        "row_count_ok": rows == manifest["row_count"],
    }


async def purge_expired(pool: Pool, now: datetime, retention: timedelta) -> dict[str, int]:
    """Delete operational rows past retention.

    Scored attempts go only once they are in durable history. Attempts that never got an
    assessment (UNAVAILABLE, or IN_PROGRESS and abandoned) have nothing to archive and are never
    exported, so they are deleted at retention directly. Their outcome events are archived
    separately, so the label history is kept.
    """
    cutoff = now - retention
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute(
            "DELETE FROM assessments s USING attempts a, archive_ledger l "
            "WHERE s.tenant_id = a.tenant_id AND s.payment_attempt_id = a.payment_attempt_id "
            "AND l.tenant_id = a.tenant_id AND l.payment_attempt_id = a.payment_attempt_id "
            "AND a.received_at < %s",
            (cutoff,),
        )
        assessments = cur.rowcount
        cur = await conn.execute(
            "DELETE FROM attempts a USING archive_ledger l WHERE l.tenant_id = a.tenant_id "
            "AND l.payment_attempt_id = a.payment_attempt_id AND a.received_at < %s "
            "AND NOT EXISTS (SELECT 1 FROM assessments s WHERE s.tenant_id = a.tenant_id "
            "AND s.payment_attempt_id = a.payment_attempt_id)",
            (cutoff,),
        )
        attempts = cur.rowcount
        cur = await conn.execute(
            "DELETE FROM attempts a WHERE a.status <> 'COMPLETED' AND a.received_at < %s "
            "AND NOT EXISTS (SELECT 1 FROM assessments s WHERE s.tenant_id = a.tenant_id "
            "AND s.payment_attempt_id = a.payment_attempt_id)",
            (cutoff,),
        )
        unscored = cur.rowcount
        # Outcome events go only once their attempt has left operational storage, so the
        # projection of a live attempt is never rebuilt from a partial event set. The archive
        # remains the authoritative label history for training.
        cur = await conn.execute(
            "DELETE FROM outcome_events e USING outbox o WHERE e.status = 'PROCESSED' "
            "AND e.received_at < %s AND o.topic = 'outcome.processed' "
            "AND o.tenant_id = e.tenant_id AND o.payload->>'event_id' = e.event_id "
            "AND o.published_at IS NOT NULL AND NOT EXISTS (SELECT 1 FROM attempts a "
            "WHERE a.tenant_id = e.tenant_id AND a.payment_attempt_id = e.payment_attempt_id)",
            (cutoff,),
        )
        events = cur.rowcount
        await conn.execute(
            "DELETE FROM attempt_outcomes p WHERE p.updated_at < %s AND NOT EXISTS "
            "(SELECT 1 FROM attempts a WHERE a.tenant_id = p.tenant_id "
            "AND a.payment_attempt_id = p.payment_attempt_id)",
            (cutoff,),
        )
        cur = await conn.execute(
            "DELETE FROM outbox WHERE published_at IS NOT NULL AND published_at < %s", (cutoff,)
        )
        return {
            "assessments": assessments,
            "attempts": attempts,
            "unscored_attempts": unscored,
            "outcome_events": events,
            "outbox": cur.rowcount,
        }
