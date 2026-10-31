"""Seeded synthetic marketplace simulator (T-302).

Generates API scoring requests, later outcome events, and a separate ground-truth table.
Requests contain only fields the real API accepts; the scenario label is never encoded in a
request field, so scoring sees no ground truth. Same seed + config => identical output.

Benign behavior (with deliberately confusable edge cases):
  * regular purchases from a customer's own device and tokens
  * shared family devices (several customers, several tokens, one device)
  * customers adding a new card, retrying after a decline, traveling (new IP)
  * brand-new legitimate accounts making a first purchase

Fraud scenarios:
  * card_testing      rapid bursts of small attempts with many distinct tokens on one device
  * stolen_card       new account, new device, high amount, often at night
  * account_takeover  existing customer, new device and shipping reference, high amount

Outcomes: authorization result shortly after the attempt; for a configurable share of fraud a
CONFIRMED_FRAUD (sometimes preceded by a chargeback) arrives days later; a small share of benign
attempts get CONFIRMED_LEGITIMATE from manual review. Most benign attempts are never labeled,
as in production: an approved authorization is not proof of legitimacy.

All performance measured on this data is synthetic and must be labeled "Synthetic:".
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

DATA_SOURCE = "SYNTHETIC"


@dataclass(frozen=True)
class SimConfig:
    seed: int = 7
    days: int = 90
    start: str = "2026-01-01T00:00:00+00:00"
    customers: int = 6000
    benign_per_day: int = 1200
    card_testing_bursts_per_day: float = 2.0
    stolen_card_per_day: float = 6.0
    account_takeover_per_day: float = 3.0
    shared_device_share: float = 0.08
    fraud_report_rate: float = 0.8  # share of fraud eventually confirmed
    legit_review_rate: float = 0.02  # share of benign confirmed by manual review
    tenant_id: str = "tenant-demo"


@dataclass
class Customer:
    cid: str
    created_at: datetime
    devices: list[str]
    tokens: list[str]
    shipping: str
    ip: str


@dataclass
class Simulation:
    requests: list[dict[str, Any]] = field(default_factory=list)
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    truth: list[dict[str, Any]] = field(default_factory=list)


def _fmt(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class Simulator:
    def __init__(self, config: SimConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.start = datetime.fromisoformat(config.start)
        self._n = 0
        self.customers: list[Customer] = []
        self.sim = Simulation()

    # ---------------------------------------------------------------- identifiers

    def _id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}-{self._n:08d}"

    def _ip(self) -> str:
        a, b, c = (int(x) for x in self.rng.integers(1, 255, 3))
        return f"198.{a % 20 + 18}.{b}.{c}"

    # ---------------------------------------------------------------- population

    def _make_customers(self) -> None:
        for _ in range(self.cfg.customers):
            age_days = float(self.rng.exponential(400)) + 1
            created = self.start - timedelta(days=age_days)
            n_tokens = int(self.rng.choice([1, 1, 1, 2, 2, 3]))
            self.customers.append(
                Customer(
                    cid=self._id("cust"),
                    created_at=created,
                    devices=[self._id("dev")],
                    tokens=[self._id("tok") for _ in range(n_tokens)],
                    shipping=self._id("ship"),
                    ip=self._ip(),
                )
            )
        # Shared family devices: several customers on one device.
        n_shared = int(self.cfg.customers * self.cfg.shared_device_share)
        for i in range(0, n_shared - 2, 3):
            shared = self._id("dev")
            for c in self.customers[i : i + 3]:
                c.devices.append(shared)

    # ---------------------------------------------------------------- emission

    def _amount(self, mean_log: float, sigma: float) -> int:
        return max(100, int(np.exp(self.rng.normal(mean_log, sigma))))

    def _emit(
        self,
        at: datetime,
        scenario: str,
        *,
        customer: str | None,
        device: str | None,
        token: str | None,
        amount: int,
        account_created: datetime | None,
        shipping: str | None,
        ip: str | None,
    ) -> str:
        attempt = self._id("pa")
        order = self._id("order")
        req: dict[str, Any] = {
            "transaction_id": order,
            "payment_attempt_id": attempt,
            "idempotency_key": f"idem-{attempt}",
            "amount_minor": amount,
            "currency": "USD",
            "occurred_at": _fmt(at),
        }
        for key, value in (
            ("customer_id", customer),
            ("device_id", device),
            ("payment_method_token", token),
            ("ip_address", ip),
            ("shipping_destination_ref", shipping),
        ):
            if value is not None:
                req[key] = value
        if account_created is not None:
            req["account_created_at"] = _fmt(account_created)
        self.sim.requests.append(req)
        is_fraud = scenario not in ("benign", "benign_edge")
        self.sim.truth.append(
            {
                "payment_attempt_id": attempt,
                "scenario": scenario,
                "is_fraud": is_fraud,
                "occurred_at": _fmt(at),
            }
        )
        self._outcomes(attempt, order, at, is_fraud, scenario)
        return attempt

    def _outcomes(self, attempt: str, order: str, at: datetime, fraud: bool, scenario: str) -> None:
        def event(kind: str, source: str, when: datetime) -> None:
            self.sim.outcomes.append(
                {
                    "event_id": self._id("evt"),
                    "payment_attempt_id": attempt,
                    "transaction_id": order,
                    "event_type": kind,
                    "source": source,
                    "occurred_at": _fmt(when),
                }
            )

        auth_at = at + timedelta(seconds=float(self.rng.uniform(1, 5)))
        declined = (scenario == "card_testing" and self.rng.random() < 0.7) or (
            not fraud and self.rng.random() < 0.03
        )
        event(
            "AUTHORIZATION_DECLINED" if declined else "AUTHORIZATION_APPROVED", "PROCESSOR", auth_at
        )
        if fraud and not declined and self.rng.random() < self.cfg.fraud_report_rate:
            delay = timedelta(days=float(self.rng.uniform(2, 45)))
            if self.rng.random() < 0.6:
                event("CHARGEBACK_REPORTED", "ISSUER", at + delay)
                delay += timedelta(days=float(self.rng.uniform(1, 10)))
            event("CONFIRMED_FRAUD", "ISSUER", at + delay)
        elif not fraud and self.rng.random() < self.cfg.legit_review_rate:
            event(
                "CONFIRMED_LEGITIMATE",
                "MANUAL_REVIEW",
                at + timedelta(days=float(self.rng.uniform(1, 10))),
            )

    # ---------------------------------------------------------------- behaviors

    def _benign(self, at: datetime) -> None:
        c = self.customers[int(self.rng.integers(len(self.customers)))]
        r = self.rng.random()
        token = c.tokens[int(self.rng.integers(len(c.tokens)))]
        device = c.devices[int(self.rng.integers(len(c.devices)))]
        ip, scenario = c.ip, "benign"
        if r < 0.03:  # adds a new card
            token = self._id("tok")
            c.tokens.append(token)
            scenario = "benign_edge"
        elif r < 0.05:  # traveling
            ip, scenario = self._ip(), "benign_edge"
        amount = self._amount(8.3, 0.9)
        customer, created, shipping = c.cid, c.created_at, c.shipping
        if 0.05 <= r < 0.15:  # guest checkout: no account
            customer, created, scenario = None, None, "benign_edge"
        elif 0.15 <= r < 0.20:  # digital goods: no shipping reference
            shipping, scenario = None, "benign_edge"
        self._emit(
            at,
            scenario,
            customer=customer,
            device=device,
            token=token,
            amount=amount,
            account_created=created,
            shipping=shipping,
            ip=ip,
        )
        if r > 0.98:  # retries the same order after a decline: new attempt, same token
            self._emit(
                at + timedelta(seconds=float(self.rng.uniform(10, 90))),
                "benign_edge",
                customer=customer,
                device=device,
                token=token,
                amount=amount,
                account_created=created,
                shipping=shipping,
                ip=ip,
            )

    def _new_legit_account(self, at: datetime) -> None:
        c = Customer(
            self._id("cust"),
            at - timedelta(minutes=float(self.rng.uniform(5, 600))),
            [self._id("dev")],
            [self._id("tok")],
            self._id("ship"),
            self._ip(),
        )
        self.customers.append(c)
        self._emit(
            at,
            "benign_edge",
            customer=c.cid,
            device=c.devices[0],
            token=c.tokens[0],
            amount=self._amount(9.0, 1.0),
            account_created=c.created_at,
            shipping=c.shipping,
            ip=c.ip,
        )

    def _card_testing(self, at: datetime) -> None:
        device, ip = self._id("dev"), self._ip()
        account = self._id("cust") if self.rng.random() < 0.5 else None
        created = at - timedelta(minutes=float(self.rng.uniform(1, 30))) if account else None
        t = at
        for _ in range(int(self.rng.integers(4, 18))):
            t += timedelta(seconds=float(self.rng.uniform(5, 40)))
            self._emit(
                t,
                "card_testing",
                customer=account,
                device=device,
                token=self._id("tok"),
                amount=int(self.rng.integers(100, 600)),
                account_created=created,
                shipping=None,
                ip=ip,
            )

    def _stolen_card(self, at: datetime) -> None:
        if self.rng.random() < 0.6:  # fraud skews to night hours
            at = at.replace(hour=int(self.rng.integers(0, 6)))
        created = at - timedelta(minutes=float(self.rng.uniform(2, 1440)))
        device = self._id("dev")
        for i in range(int(self.rng.integers(1, 4))):
            self._emit(
                at + timedelta(minutes=3 * i),
                "stolen_card",
                customer=self._id("cust"),
                device=device,
                token=self._id("tok"),
                amount=self._amount(10.2, 0.7),
                account_created=created,
                shipping=self._id("ship"),
                ip=self._ip(),
            )

    def _account_takeover(self, at: datetime) -> None:
        c = self.customers[int(self.rng.integers(len(self.customers)))]
        self._emit(
            at,
            "account_takeover",
            customer=c.cid,
            device=self._id("dev"),
            token=c.tokens[0],
            amount=self._amount(10.0, 0.8),
            account_created=c.created_at,
            shipping=self._id("ship"),
            ip=self._ip(),
        )

    # ---------------------------------------------------------------- driver

    def run(self) -> Simulation:
        self._make_customers()
        cfg = self.cfg
        for day in range(cfg.days):
            base = self.start + timedelta(days=day)
            # Diurnal benign traffic: more purchases in the afternoon/evening.
            hours = self.rng.choice(24, size=cfg.benign_per_day, p=_diurnal()).astype(float)
            events: list[tuple[float, str]] = [
                (h * 3600 + float(self.rng.uniform(0, 3600)), "benign") for h in hours
            ]
            for kind, rate in (
                ("card_testing", cfg.card_testing_bursts_per_day),
                ("stolen_card", cfg.stolen_card_per_day),
                ("account_takeover", cfg.account_takeover_per_day),
                ("new_legit", cfg.benign_per_day * 0.01),
            ):
                for _ in range(int(self.rng.poisson(rate))):
                    events.append((float(self.rng.uniform(0, 86400)), kind))
            events.sort()
            handlers = {
                "benign": self._benign,
                "card_testing": self._card_testing,
                "stolen_card": self._stolen_card,
                "account_takeover": self._account_takeover,
                "new_legit": self._new_legit_account,
            }
            for offset, kind in events:
                handlers[kind](base + timedelta(seconds=offset))

        order = np.argsort([r["occurred_at"] for r in self.sim.requests], kind="mergesort")
        self.sim.requests = [self.sim.requests[i] for i in order]
        self.sim.truth = [self.sim.truth[i] for i in order]
        self.sim.outcomes.sort(key=lambda e: (e["occurred_at"], e["event_id"]))
        return self.sim


def _diurnal() -> np.ndarray:
    w = np.array(
        [2, 1, 1, 1, 1, 2, 3, 5, 6, 7, 8, 8, 9, 9, 9, 9, 10, 11, 12, 12, 11, 9, 6, 4], dtype=float
    )
    return w / w.sum()


def simulate(config: SimConfig) -> Simulation:
    return Simulator(config).run()


def write(sim: Simulation, config: SimConfig, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("requests", sim.requests),
        ("outcomes", sim.outcomes),
        ("truth", sim.truth),
    ):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "data_source": DATA_SOURCE,
        "config": asdict(config),
        "requests": len(sim.requests),
        "outcomes": len(sim.outcomes),
        "fraud_attempts": sum(t["is_fraud"] for t in sim.truth),
        "scenarios": {
            s: sum(t["scenario"] == s for t in sim.truth)
            for s in sorted({t["scenario"] for t in sim.truth})
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seeded synthetic marketplace data")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()
    config = SimConfig(seed=args.seed, days=args.days)
    sim = simulate(config)
    write(sim, config, args.out)
    print(json.dumps(json.loads((args.out / "summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
