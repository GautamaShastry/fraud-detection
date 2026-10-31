"""Pseudonymization and request fingerprints (I-12 provisional: per-tenant HMAC-SHA256).

Raw customer, device, payment-token, IP and shipping identifiers never leave the request
handler: storage, Redis keys, logs and features see only these keyed pseudonyms.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any


class Pseudonymizer:
    def __init__(self, key: bytes) -> None:
        if len(key) < 32:
            raise ValueError("Pseudonymization key must be at least 32 bytes")
        self._key = key

    def ref(self, tenant_id: str, kind: str, value: str | None) -> str | None:
        """Stable per-tenant pseudonym. The same raw value in two tenants does not collide."""
        if value is None:
            return None
        msg = f"{tenant_id}\x1f{kind}\x1f{value}".encode()
        return hmac.new(self._key, msg, hashlib.sha256).hexdigest()[:32]

    def fingerprint(self, tenant_id: str, payload: dict[str, Any]) -> bytes:
        """Keyed digest of the canonical request, used to detect idempotency-key reuse."""
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hmac.new(self._key, f"{tenant_id}\x1f{canonical}".encode(), hashlib.sha256).digest()
