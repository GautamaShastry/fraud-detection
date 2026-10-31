"""Low-latency single-row inference for fitted HistGradientBoostingClassifier models.

scikit-learn's predict_proba is built for batches: per-call validation and OpenMP thread
fan-out cost 15-100+ ms for one row, which does not fit the 110 ms primary budget at load. This
module flattens the fitted trees into arrays once and walks them directly (~sub-millisecond).

It reproduces scikit-learn's numeric split rule exactly (NaN follows missing_go_to_left;
otherwise x <= threshold goes left). Models with categorical splits are rejected. `verify`
compares against predict_proba on sample rows and the artifact loader refuses any model whose
compiled output differs by more than 1e-9.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class _Tree:
    feature: tuple[int, ...]
    threshold: tuple[float, ...]
    missing_left: tuple[bool, ...]
    left: tuple[int, ...]
    right: tuple[int, ...]
    leaf: tuple[bool, ...]
    value: tuple[float, ...]


class CompiledHGB:
    def __init__(self, trees: list[_Tree], baseline: float, n_features: int) -> None:
        self._trees = trees
        self._baseline = baseline
        self.n_features = n_features

    @classmethod
    def from_estimator(cls, est: Any) -> CompiledHGB:
        if getattr(est, "n_trees_per_iteration_", 1) != 1:
            raise ValueError("Only binary classifiers are supported")
        trees = []
        for (predictor,) in est._predictors:
            nodes = predictor.nodes
            if nodes["is_categorical"].any():
                raise ValueError("Categorical splits are not supported by the compiled path")
            trees.append(
                _Tree(
                    feature=tuple(int(v) for v in nodes["feature_idx"]),
                    threshold=tuple(float(v) for v in nodes["num_threshold"]),
                    missing_left=tuple(bool(v) for v in nodes["missing_go_to_left"]),
                    left=tuple(int(v) for v in nodes["left"]),
                    right=tuple(int(v) for v in nodes["right"]),
                    leaf=tuple(bool(v) for v in nodes["is_leaf"]),
                    value=tuple(float(v) for v in nodes["value"]),
                )
            )
        baseline = float(np.asarray(est._baseline_prediction).ravel()[0])
        return cls(trees, baseline, int(est.n_features_in_))

    def raw(self, row: list[float]) -> float:
        total = self._baseline
        for t in self._trees:
            i = 0
            feature, threshold, left, right = t.feature, t.threshold, t.left, t.right
            leaf, missing_left = t.leaf, t.missing_left
            while not leaf[i]:
                x = row[feature[i]]
                if x != x:  # NaN
                    i = left[i] if missing_left[i] else right[i]
                elif x <= threshold[i]:
                    i = left[i]
                else:
                    i = right[i]
            total += t.value[i]
        return total

    def predict_proba_one(self, row: list[float]) -> float:
        raw = self.raw(row)
        # Numerically stable logistic, matching scipy.special.expit.
        if raw >= 0:
            return 1.0 / (1.0 + math.exp(-raw))
        e = math.exp(raw)
        return e / (1.0 + e)

    def verify(self, est: Any, rows: np.ndarray[Any, Any], tolerance: float = 1e-9) -> float:
        """Max absolute difference from predict_proba over `rows`; raises if above tolerance."""
        expected = est.predict_proba(rows)[:, 1]
        actual = np.array([self.predict_proba_one(list(r)) for r in rows])
        diff = float(np.max(np.abs(expected - actual))) if len(rows) else 0.0
        if diff > tolerance:
            raise ValueError(f"Compiled model diverges from scikit-learn by {diff}")
        return diff
