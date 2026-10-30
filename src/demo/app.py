"""Synthetic checkout demo (FR-11, T-403). NO REAL PAYMENTS, no card data, no processor.

Plays the marketplace: builds a scoring request, calls the real scoring API with the client
timeout (200 ms), and shows the recommendation next to the simulated payment decision. If no
assessment arrives (timeout, outage, 503/504) checkout continues to ordinary authorization
(fail-open); authentication errors are surfaced, never bypassed.

    uvicorn src.demo.app:create_app --factory --port 8090
"""

from __future__ import annotations

import asyncio
import os
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

STATIC = Path(__file__).parent / "static"
OUTAGE_URL = "http://10.255.255.1:9"

DECISIONS = {
    "PROCEED": "Proceed to payment authorization (simulated)",
    "VERIFY": "Ask the shopper to verify, then authorize (simulated)",
    "RECOMMEND_DECLINE": "Merchant policy: decline before authorization (simulated)",
}
FAIL_OPEN = "No assessment: continue to ordinary payment authorization (fail-open, simulated)"


class CheckoutRequest(BaseModel):
    amount_minor: int = Field(ge=0, le=10_000_000)
    device_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$")
    payment_method_token: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$")
    customer_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$")
    simulate_outage: bool = False


class TokenSource:
    """Mints merchant client-credential tokens with the local test issuer key."""

    def __init__(self, keys_dir: Path, client_id: str) -> None:
        self._keys_dir, self._client_id = keys_dir, client_id
        self._cached: tuple[str, float] | None = None

    def get(self) -> str:
        if self._cached and self._cached[1] > time.time() + 30:
            return self._cached[0]
        from scripts.dev_jwt import DEFAULT_AUDIENCE, DEFAULT_ISSUER, mint

        token = mint(
            self._keys_dir, self._client_id, "fraud:score", 600, DEFAULT_ISSUER, DEFAULT_AUDIENCE
        )
        self._cached = (token, time.time() + 600)
        return token


def create_app(
    api_url: str | None = None,
    tokens: Any = None,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout_s: float | None = None,
) -> FastAPI:
    api_url = api_url or os.environ.get("DEMO_FRAUD_API_URL") or "http://127.0.0.1:8000"
    timeout_s = timeout_s or float(os.environ.get("DEMO_TIMEOUT_S", "0.2"))
    tokens = tokens or TokenSource(
        Path(os.environ.get("DEMO_KEYS_DIR", "infra/local/secrets")),
        os.environ.get("DEMO_CLIENT_ID", "demo-merchant"),
    )
    clients: dict[str, httpx.AsyncClient] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Long-lived, pooled clients (as a real integration would keep): no per-checkout DNS
        # lookup or TCP handshake inside the 200 ms budget.
        clients["api"] = httpx.AsyncClient(base_url=api_url, timeout=timeout_s, transport=transport)
        clients["outage"] = httpx.AsyncClient(base_url=OUTAGE_URL, timeout=timeout_s)
        try:
            yield
        finally:
            for client in clients.values():
                await client.aclose()

    app = FastAPI(
        title="Checkout demo (synthetic)", openapi_url=None, docs_url=None, lifespan=lifespan
    )

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.post("/checkout")
    async def checkout(req: CheckoutRequest) -> dict[str, Any]:
        attempt = f"demo-{secrets.token_hex(6)}"
        body: dict[str, Any] = {
            "transaction_id": f"order-{attempt}",
            "payment_attempt_id": attempt,
            "idempotency_key": f"idem-{attempt}",
            "amount_minor": req.amount_minor,
            "currency": "USD",
            "occurred_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "device_id": req.device_id,
            "payment_method_token": req.payment_method_token,
        }
        if req.customer_id:
            body["customer_id"] = req.customer_id
        # A controlled outage: an unroutable endpoint, so the client timeout fires.
        client = clients["outage" if req.simulate_outage else "api"]
        headers = {"Authorization": f"Bearer {tokens.get()}"}
        started = time.perf_counter()
        status, response, error = None, None, None
        try:
            async with asyncio.timeout(timeout_s):  # total budget for the scoring call
                r = await client.post("/api/v1/fraud/score", json=body, headers=headers)
            status, response = r.status_code, r.json()
        except (httpx.HTTPError, TimeoutError) as exc:
            error = type(exc).__name__
        elapsed_ms = (time.perf_counter() - started) * 1000

        if status in (401, 403):
            decision = "Integration error: credentials rejected (fix configuration; not bypassed)"
        elif status == 200 and response and response.get("recommendation") in DECISIONS:
            decision = DECISIONS[response["recommendation"]]
        else:
            decision = FAIL_OPEN
        return {
            "request": body,
            "status": status,
            "response": response,
            "error": error,
            "client_latency_ms": round(elapsed_ms, 1),
            "payment_decision": decision,
            "disclaimer": "Synthetic demo: no real payment was attempted.",
        }

    return app
