"""Feature availability matrix: IEEE-CIS columns vs. what the scoring API receives (T-301).

PRD risk 11: IEEE-CIS records may lack production candidate features. This matrix states, for
every online feature in src/scoring/features.py, whether IEEE-CIS can supply it, so no model is
trained on columns the API cannot provide, and no API feature is fabricated from the dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.scoring.features import FEATURE_NAMES


@dataclass(frozen=True)
class Availability:
    online_feature: str
    api_source: str
    ieee_source: str
    status: str  # AVAILABLE | PARTIAL | UNAVAILABLE
    note: str


MATRIX: tuple[Availability, ...] = (
    Availability(
        "amount_log",
        "amount_minor",
        "TransactionAmt",
        "AVAILABLE",
        "IEEE amounts are in USD major units; currency is not recorded.",
    ),
    Availability(
        "hour_sin",
        "occurred_at",
        "TransactionDT",
        "PARTIAL",
        "TransactionDT is seconds from an undisclosed reference; hour-of-day is only "
        "recoverable up to an unknown offset.",
    ),
    Availability("hour_cos", "occurred_at", "TransactionDT", "PARTIAL", "As hour_sin."),
    Availability(
        "account_age_days",
        "account_created_at",
        "D1 (days since card first use?)",
        "UNAVAILABLE",
        "D-columns are undocumented timedeltas; none is defined as "
        "account age, so using one would invent semantics.",
    ),
    Availability(
        "device_distinct_payment_methods_5m",
        "device_id + payment_method_token",
        "none",
        "UNAVAILABLE",
        "IEEE has no device identifier (DeviceInfo is a model "
        "string), so F01 cannot be computed from IEEE-CIS.",
    ),
    Availability(
        "has_customer", "customer_id", "none", "UNAVAILABLE", "No customer identifier in IEEE-CIS."
    ),
    Availability(
        "has_device",
        "device_id",
        "DeviceType/DeviceInfo presence",
        "PARTIAL",
        "Presence of device metadata, not of a device identifier.",
    ),
    Availability(
        "has_payment_method",
        "payment_method_token",
        "card1 presence",
        "PARTIAL",
        "card1 is an anonymized card attribute, not a token.",
    ),
    Availability("has_ip", "ip_address", "none", "UNAVAILABLE", "No IP column."),
    Availability(
        "has_shipping",
        "shipping_destination_ref",
        "addr1/addr2 presence",
        "PARTIAL",
        "addr1/addr2 are anonymized billing-region codes, not shipping references.",
    ),
)

assert tuple(a.online_feature for a in MATRIX) == FEATURE_NAMES, "matrix out of date"


def as_rows() -> list[dict[str, str]]:
    return [asdict(a) for a in MATRIX]


def to_markdown() -> str:
    lines = [
        "| Online feature | API source | IEEE-CIS source | Status | Note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for a in MATRIX:
        lines.append(
            f"| {a.online_feature} | {a.api_source} | {a.ieee_source} | {a.status} | " f"{a.note} |"
        )
    return "\n".join(lines)
