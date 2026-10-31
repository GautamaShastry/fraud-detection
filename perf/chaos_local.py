"""Local chaos scenarios against the compose stack (T-505). Free; uses docker CLI.

Each scenario injects a fault while sending scoring traffic, checks the documented behavior,
restores the service, and checks recovery. Results go to evidence/chaos/<timestamp>.json.

    python perf/chaos_local.py --keys infra/local/secrets
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dev_jwt import DEFAULT_AUDIENCE, DEFAULT_ISSUER, mint  # noqa: E402

COMPOSE = ["docker", "compose", "-f", "infra/local/compose.yaml"]
API = "http://127.0.0.1:8000"
DEMO = "http://127.0.0.1:8090"


def compose(*args: str) -> None:
    subprocess.run([*COMPOSE, *args], check=True, capture_output=True)  # noqa: S603


def body(run: str, i: int) -> dict[str, Any]:
    return {
        "transaction_id": f"chaos-o-{run}-{i}",
        "payment_attempt_id": f"chaos-pa-{run}-{i}",
        "idempotency_key": f"chaos-idem-{run}-{i}",
        "amount_minor": 2500,
        "currency": "USD",
        "occurred_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device_id": f"chaos-dev-{run}",
        "payment_method_token": f"chaos-tok-{run}-{i}",
    }


def burst(client: httpx.Client, token: str, run: str, n: int = 20) -> Counter[str]:
    seen: Counter[str] = Counter()
    for i in range(n):
        started = time.perf_counter()
        try:
            r = client.post(
                f"{API}/api/v1/fraud/score",
                json=body(run, i),
                headers={"Authorization": f"Bearer {token}"},
                timeout=2.0,
            )
            key = r.json().get("scoring_mode") if r.status_code == 200 else r.json().get("code")
            seen[f"{r.status_code}:{key}"] += 1
        except (httpx.HTTPError, ValueError) as exc:
            seen[f"error:{type(exc).__name__}"] += 1
        seen["slow>500ms"] += int(time.perf_counter() - started > 0.5)
    return seen


def ready(client: httpx.Client) -> dict[str, Any]:
    try:
        return client.get(f"{API}/health/ready", timeout=3).json()
    except (httpx.HTTPError, ValueError) as exc:
        return {"status": f"unreachable ({type(exc).__name__})"}


def wait_ready(client: httpx.Client, timeout_s: float = 60) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if ready(client).get("status") == "ok":
            return True
        time.sleep(1)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", type=Path, required=True)
    args = parser.parse_args()
    token = mint(
        args.keys,
        "demo-merchant",
        "fraud:score fraud:outcomes",
        3600,
        DEFAULT_ISSUER,
        DEFAULT_AUDIENCE,
    )
    results: list[dict[str, Any]] = []
    client = httpx.Client()

    def scenario(
        name: str,
        expect: str,
        inject: list[str],
        restore: list[str],
        check: Any,
        settle_s: float = 2.0,
    ) -> None:
        run = secrets.token_hex(3)
        try:
            compose(*inject)
            time.sleep(settle_s)
            during = burst(client, token, run)
            health = ready(client)
        finally:
            compose(*restore)  # Never leave the stack broken, whatever happened above
        recovered = wait_ready(client)
        after = burst(client, token, run + "r", 5)
        ok = bool(check(during)) and recovered and after.get("200:ML", 0) == 5
        results.append(
            {
                "scenario": name,
                "expected": expect,
                "during": dict(during),
                "readiness_during": health,
                "recovered": recovered,
                "after_recovery": dict(after),
                "pass": ok,
            }
        )
        print(f"{'PASS' if ok else 'FAIL'}  {name}: during={dict(during)} after={dict(after)}")

    assert wait_ready(client), "stack is not ready"
    scenario(
        "redis down",
        "every assessment FALLBACK, none unavailable, never decline",
        ["stop", "redis"],
        ["start", "redis"],
        lambda d: d.get("200:FALLBACK", 0) == 20,
    )
    scenario(
        "postgres paused",
        "explicit 503/504, no fabricated score, fast failure",
        ["pause", "postgres"],
        ["unpause", "postgres"],
        # Every response is the documented 503/504: a client timeout is not a pass.
        lambda d: all(k.startswith(("503:", "504:")) or k == "slow>500ms" for k in d)
        and d.get("slow>500ms", 0) == 0,
    )
    scenario(
        "worker killed",
        "scoring unaffected",
        ["kill", "worker"],
        ["start", "worker"],
        lambda d: d.get("200:ML", 0) == 20,
        settle_s=1,
    )

    # API down: the checkout demo must fail open within its 200 ms client timeout.
    try:
        compose("stop", "api")
        time.sleep(1)
        started = time.perf_counter()
        try:
            demo = client.post(
                f"{DEMO}/checkout",
                json={"amount_minor": 999, "device_id": "chaos", "payment_method_token": "chaos"},
                timeout=5,
            ).json()
        except httpx.HTTPError as exc:
            demo = {"payment_decision": f"demo unreachable ({type(exc).__name__})"}
        elapsed = time.perf_counter() - started
    finally:
        compose("start", "api")
    recovered = wait_ready(client, 90)
    ok = "fail-open" in demo.get("payment_decision", "") and elapsed < 1.5 and recovered
    results.append(
        {
            "scenario": "api down (merchant view)",
            "expected": "checkout fails open",
            "demo": demo,
            "elapsed_s": round(elapsed, 3),
            "recovered": recovered,
            "pass": ok,
        }
    )
    print(
        f"{'PASS' if ok else 'FAIL'}  api down: {demo.get('payment_decision')} "
        f"in {elapsed * 1000:.0f} ms"
    )

    out = Path("evidence/chaos")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"chaos-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    sys.exit(0 if all(r["pass"] for r in results) else 1)


if __name__ == "__main__":
    main()
