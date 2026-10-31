"""Scorers and scoring policy.

A scorer turns a feature dict into a model output in [0, 1]. A policy turns that output into a
0–100 risk score and a recommendation. The risk score is a monotonic rank against the
validation distribution (I-17 provisional), not a calibrated fraud probability.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.domain.models import Recommendation, RiskSignal
from src.scoring.features import to_row


class Scorer(Protocol):
    model_version: str
    feature_names: tuple[str, ...]

    def predict(self, features: dict[str, float | None]) -> float: ...


@dataclass(frozen=True)
class ScoringPolicy:
    """Maps a model output to a risk score and a recommendation.

    * risk_score: percentile rank of the output against the validation distribution
      (101 ascending cut points; score k when output >= cutpoints[k]).
    * recommendation: compares the raw output with thresholds selected on validation data at
      target false-positive rates. Using output thresholds (not integer scores) keeps the
      approved FPR guardrail exact even when it falls inside the top percentile.
    """

    version: str
    cutpoints: tuple[float, ...]
    verify_threshold: float
    decline_threshold: float | None  # None: this policy never recommends decline

    def __post_init__(self) -> None:
        if len(self.cutpoints) != 101 or list(self.cutpoints) != sorted(self.cutpoints):
            raise ValueError("cutpoints must be 101 ascending values")
        if self.decline_threshold is not None and self.decline_threshold < self.verify_threshold:
            raise ValueError("decline_threshold must be >= verify_threshold")

    def risk_score(self, output: float) -> int:
        if math.isnan(output):
            raise ValueError("model output is NaN")
        return max(0, min(100, bisect.bisect_right(self.cutpoints, output) - 1))

    def recommend(self, output: float) -> Recommendation:
        if self.decline_threshold is not None and output >= self.decline_threshold:
            return Recommendation.RECOMMEND_DECLINE
        if output >= self.verify_threshold:
            return Recommendation.VERIFY
        return Recommendation.PROCEED

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ScoringPolicy:
        decline = d.get("decline_threshold")
        return cls(
            version=d["version"],
            cutpoints=tuple(float(c) for c in d["cutpoints"]),
            verify_threshold=float(d["verify_threshold"]),
            decline_threshold=None if decline is None else float(decline),
        )


@dataclass(frozen=True)
class SignalRule:
    """Evidence flag: feature value at or above a threshold derived from training data."""

    feature: str
    threshold: float
    code: str
    description: str


@dataclass
class EstimatorScorer:
    """Wraps a fitted scikit-learn classifier.

    HistGradientBoosting models are served through the compiled single-row path (verified
    against predict_proba when loaded); other estimators fall back to predict_proba.
    """

    model_version: str
    feature_names: tuple[str, ...]
    estimator: Any
    compiled: Any = None

    @property
    def inline(self) -> bool:
        return self.compiled is not None

    def predict(self, features: dict[str, float | None]) -> float:
        row = to_row(features, self.feature_names)
        if self.compiled is not None:
            return float(self.compiled.predict_proba_one(row))
        return float(self.estimator.predict_proba([row])[0][1])


@dataclass(frozen=True)
class ScoringBundle:
    """Everything needed to produce assessments for one deployed model version."""

    primary: Scorer
    primary_policy: ScoringPolicy
    fallback: Scorer | None
    fallback_policy: ScoringPolicy | None
    signal_rules: tuple[SignalRule, ...] = field(default=())

    def signals(self, features: dict[str, float | None]) -> tuple[RiskSignal, ...]:
        out = []
        for rule in self.signal_rules:
            value = features.get(rule.feature)
            if value is not None and value >= rule.threshold:
                out.append(RiskSignal(rule.code, rule.description))
        return tuple(out)
