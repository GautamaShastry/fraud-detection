"""Offline feature computation with online parity (T-303).

F01 is replayed with exactly the semantics of the Redis Lua script in src/features/velocity.py:
per (tenant, device) the last-seen time of each payment-method token, counting tokens seen
within the 5-minute window (exclusive at exactly 5 minutes) including the current attempt, with
retries of an attempt returning the value first recorded. Feature vectors then come from the
same `build_features` the API uses. The parity integration test replays a sequence through both
implementations and requires identical values.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from src.features.velocity import WINDOW_MS
from src.scoring.features import FeatureInputs, build_features


class OfflineF01:
    def __init__(self) -> None:
        self._last_seen: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
        self._recorded: dict[tuple[str, str], int] = {}

    def update(
        self, tenant: str, attempt: str, device: str | None, token: str | None, now_ms: int
    ) -> int | None:
        if device is None or token is None:
            return None
        key = (tenant, attempt)
        if key in self._recorded:
            return self._recorded[key]
        seen = self._last_seen[(tenant, device)]
        if seen.get(token, -1) < now_ms:
            seen[token] = now_ms
        for t in [t for t, ts in seen.items() if ts <= now_ms - WINDOW_MS]:
            del seen[t]
        count = sum(1 for ts in seen.values() if ts >= now_ms - WINDOW_MS + 1)
        self._recorded[key] = count
        return count


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def featurize(
    requests: Iterable[dict[str, Any]], tenant: str = "tenant-demo"
) -> list[dict[str, float | None]]:
    """Feature vectors for requests in arrival order (server receive time = occurred_at)."""
    f01 = OfflineF01()
    rows = []
    for req in requests:
        occurred = _ts(req["occurred_at"])
        value = f01.update(
            tenant,
            req["payment_attempt_id"],
            req.get("device_id"),
            req.get("payment_method_token"),
            int(occurred.timestamp() * 1000),
        )
        account = req.get("account_created_at")
        rows.append(
            build_features(
                FeatureInputs(
                    amount_minor=req["amount_minor"],
                    occurred_at=occurred,
                    account_created_at=_ts(account) if account else None,
                    f01=value,
                    has_customer="customer_id" in req,
                    has_device="device_id" in req,
                    has_payment_method="payment_method_token" in req,
                    has_ip="ip_address" in req,
                    has_shipping="shipping_destination_ref" in req,
                )
            )
        )
    return rows
