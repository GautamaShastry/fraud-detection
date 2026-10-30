"""Online feature schema shared by inference and offline training (FR-06, T-303 parity).

Training code imports `build_features` from here, so offline and online feature values come
from one implementation. Changing a feature definition requires bumping FEATURE_SCHEMA; the
API refuses to load a model artifact built for a different schema.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

FEATURE_SCHEMA = "online-features-1.0.0"

# Order matters: it is the column order models are trained on.
FEATURE_NAMES: tuple[str, ...] = (
    "amount_log",
    "hour_sin",
    "hour_cos",
    "account_age_days",
    "device_distinct_payment_methods_5m",
    "has_customer",
    "has_device",
    "has_payment_method",
    "has_ip",
    "has_shipping",
)

# Features the fallback scorer may not rely on (they depend on Redis).
VELOCITY_FEATURES = frozenset({"device_distinct_payment_methods_5m"})
FALLBACK_FEATURE_NAMES: tuple[str, ...] = tuple(
    f for f in FEATURE_NAMES if f not in VELOCITY_FEATURES
)


@dataclass(frozen=True)
class FeatureInputs:
    amount_minor: int
    occurred_at: datetime
    account_created_at: datetime | None
    f01: int | None
    has_customer: bool
    has_device: bool
    has_payment_method: bool
    has_ip: bool
    has_shipping: bool


def build_features(x: FeatureInputs) -> dict[str, float | None]:
    """Feature values; None means missing (models must handle missingness explicitly)."""
    hour = x.occurred_at.hour + x.occurred_at.minute / 60.0
    angle = 2 * math.pi * hour / 24.0
    account_age = None
    if x.account_created_at is not None:
        account_age = max((x.occurred_at - x.account_created_at).total_seconds(), 0.0) / 86400.0
    return {
        "amount_log": math.log1p(x.amount_minor),
        "hour_sin": math.sin(angle),
        "hour_cos": math.cos(angle),
        "account_age_days": account_age,
        "device_distinct_payment_methods_5m": None if x.f01 is None else float(x.f01),
        "has_customer": float(x.has_customer),
        "has_device": float(x.has_device),
        "has_payment_method": float(x.has_payment_method),
        "has_ip": float(x.has_ip),
        "has_shipping": float(x.has_shipping),
    }


def to_row(features: dict[str, float | None], names: tuple[str, ...]) -> list[float]:
    """Model input row; missing values become NaN (native missing support in the models)."""
    return [math.nan if features[n] is None else float(features[n]) for n in names]  # type: ignore[arg-type]
