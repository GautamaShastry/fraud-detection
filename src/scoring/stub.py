"""Deterministic TEST STUB scorer (T-104). Not a model, and never presented as one.

Its version strings say "stub" so an assessment produced by it cannot be mistaken for ML
output. Enabled only with FRAUD_SCORER_MODE=stub.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.scoring.features import FALLBACK_FEATURE_NAMES, FEATURE_NAMES
from src.scoring.model import ScoringBundle, ScoringPolicy, SignalRule

STUB_MODEL_VERSION = "stub-deterministic-0"
STUB_FALLBACK_VERSION = "stub-fallback-0"


@dataclass
class StubScorer:
    model_version: str
    feature_names: tuple[str, ...]
    inline: bool = True

    def predict(self, features: dict[str, float | None]) -> float:
        # Arbitrary deterministic function of the inputs, for exercising the pipeline only.
        f01 = features.get("device_distinct_payment_methods_5m") or 0.0
        return min(1.0, 0.1 * float(f01) + 0.01 * (features["amount_log"] or 0.0))


def _linear_policy(version: str, decline_threshold: float | None) -> ScoringPolicy:
    return ScoringPolicy(
        version=version,
        cutpoints=tuple(i / 100 for i in range(101)),
        verify_threshold=0.5,
        decline_threshold=decline_threshold,
    )


def stub_bundle() -> ScoringBundle:
    return ScoringBundle(
        primary=StubScorer(STUB_MODEL_VERSION, FEATURE_NAMES),
        primary_policy=_linear_policy("stub-policy-0", decline_threshold=0.9),
        fallback=StubScorer(STUB_FALLBACK_VERSION, FALLBACK_FEATURE_NAMES),
        fallback_policy=_linear_policy("stub-fallback-policy-0", decline_threshold=None),
        signal_rules=(
            SignalRule(
                "device_distinct_payment_methods_5m",
                3.0,
                "DEVICE_DISTINCT_PAYMENT_METHODS_5M",
                "Several distinct payment methods used from this device in 5 minutes",
            ),
        ),
    )
