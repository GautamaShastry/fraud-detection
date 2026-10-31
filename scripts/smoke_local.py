"""Smoke-test the running local stack. Runs every check; exits 1 if any fail.

python scripts/smoke_local.py --keys infra/local/secrets [--expect-mode ML]
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dev_jwt import DEFAULT_AUDIENCE, DEFAULT_ISSUER, mint  # noqa: E402


def call(
    url: str, method: str = "GET", body: dict[str, Any] | None = None, token: str | None = None
) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)  # noqa: S310 - local URL
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            raw = resp.read()
            return resp.status, json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode()
    except urllib.error.HTTPError as err:
        raw = err.read()
        try:
            return err.code, json.loads(raw)
        except ValueError:  # A proxy or framework error page
            return err.code, {"error": raw[:200].decode(errors="replace")}
    except (OSError, ValueError) as exc:  # Refused, reset, timed out, or an unreadable body
        return 0, {"error": type(exc).__name__}


def wait_for_stack(base_url: str, wait_s: float) -> None:
    """`compose up --wait` can return just before a server accepts connections."""
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline and call(f"{base_url}/health/live")[0] != 200:
        time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--demo-url", default="http://127.0.0.1:8090")
    parser.add_argument("--oidc-url", default="http://127.0.0.1:8081")
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--expect-mode", default=None, help="ML; omit to accept ML or FALLBACK")
    parser.add_argument("--wait-s", type=float, default=30.0, help="Time allowed for startup")
    args = parser.parse_args()
    wait_for_stack(args.base_url, args.wait_s)

    run = secrets.token_hex(4)
    body = {
        "transaction_id": f"smoke-order-{run}",
        "payment_attempt_id": f"smoke-pa-{run}",
        "idempotency_key": f"smoke-idem-{run}",
        "amount_minor": 1000,
        "currency": "USD",
        "occurred_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device_id": f"smoke-dev-{run}",
        "payment_method_token": f"smoke-tok-{run}",
    }
    score_url = f"{args.base_url}/api/v1/fraud/score"
    good = mint(
        args.keys,
        "demo-merchant",
        "fraud:score fraud:outcomes",
        300,
        DEFAULT_ISSUER,
        DEFAULT_AUDIENCE,
    )
    wrong_aud = mint(args.keys, "demo-merchant", "fraud:score", 300, DEFAULT_ISSUER, "other-api")
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        results.append((name, ok, str(detail)[:140]))

    s, p = call(f"{args.base_url}/health/live")
    check("live", s == 200, p)
    s, p = call(f"{args.base_url}/health/ready")
    check("ready", s == 200, p)
    s, p = call(score_url, "POST", body)
    check("no token -> 401", s == 401, p.get("code"))
    s, p = call(score_url, "POST", body, wrong_aud)
    check("wrong audience -> 401", s == 401, p.get("code"))
    s, p = call(score_url, "POST", {**body, "tenant_id": "x"}, good)
    check("tenant in payload -> 422", s == 422, p.get("code"))
    s, first = call(score_url, "POST", body, good)
    mode = first.get("scoring_mode") if s == 200 else None
    ok = mode == args.expect_mode if args.expect_mode else mode in ("ML", "FALLBACK")
    check(
        "score -> 200",
        s == 200 and ok,
        {k: first.get(k) for k in ("risk_score", "recommendation", "scoring_mode", "model_version")}
        if s == 200
        else first,
    )
    s, replay = call(score_url, "POST", body, good)
    check(
        "identical retry -> same assessment",
        s == 200 and replay == first,
        replay.get("assessment_id") if isinstance(replay, dict) else replay,
    )
    s, p = call(score_url, "POST", {**body, "amount_minor": 1}, good)
    check("key reuse, different body -> 409", s == 409, p.get("code"))
    s, p = call(
        f"{args.base_url}/api/v1/fraud/outcomes",
        "POST",
        {
            "event_id": f"smoke-evt-{run}",
            "payment_attempt_id": body["payment_attempt_id"],
            "event_type": "AUTHORIZATION_APPROVED",
            "source": "PROCESSOR",
            "occurred_at": body["occurred_at"],
        },
        good,
    )
    check("outcome -> 202", s == 202, p)
    s, p = call(f"{args.base_url}/metrics")
    check("metrics", s == 200 and "fraud_score_requests_total" in str(p), "prometheus text")
    s, p = call(f"{args.base_url}/dashboard/config.json")
    check("dashboard config", s == 200, p)
    s, p = call(f"{args.oidc_url}/.well-known/openid-configuration")
    check("dev OIDC discovery", s == 200, p.get("issuer") if isinstance(p, dict) else p)
    s, p = call(
        f"{args.demo_url}/checkout",
        "POST",
        {"amount_minor": 4599, "device_id": f"demo-{run}", "payment_method_token": f"tok-{run}"},
    )
    check(
        "demo checkout via real API", s == 200 and p.get("status") == 200, p.get("payment_decision")
    )
    s, p = call(
        f"{args.demo_url}/checkout",
        "POST",
        {
            "amount_minor": 4599,
            "device_id": "d",
            "payment_method_token": "t",
            "simulate_outage": True,
        },
    )
    check(
        "demo fails open on outage",
        s == 200 and "fail-open" in p.get("payment_decision", ""),
        p.get("payment_decision"),
    )

    for name, passed, detail in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")
    sys.exit(0 if all(passed for _, passed, _ in results) else 1)


if __name__ == "__main__":
    main()
