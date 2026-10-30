"""Minority-class evaluation (FR-13, T-304; PRD 4.2).

Thresholds are always selected on the validation split and then applied unchanged to test,
so reported test recall at a target FPR is an honest out-of-sample number. Each report states
its denominators and data source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

TARGET_FPRS = (0.001, 0.005, 0.01, 0.02)
APPROVED_FPR_GUARDRAIL = 0.005  # PRD 4.2: initial FPR <= 0.5% approved; tighter targets open


def threshold_for_fpr(y: np.ndarray, p: np.ndarray, target_fpr: float) -> tuple[float, float]:
    """Lowest threshold whose validation FPR does not exceed the target.

    Returns (threshold, achieved_fpr). If no positive-flagging threshold meets the target,
    returns (inf, 0.0) and the caller reports the minimum achievable FPR.
    """
    fpr, _, thr = roc_curve(y, p)
    ok = np.where(fpr <= target_fpr)[0]
    if len(ok) == 0:
        return float("inf"), 0.0
    i = ok[-1]
    return float(thr[i]), float(fpr[i])


def confusion(y: np.ndarray, flagged: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(((flagged == 1) & (y == 1)).sum()),
        "fp": int(((flagged == 1) & (y == 0)).sum()),
        "tn": int(((flagged == 0) & (y == 0)).sum()),
        "fn": int(((flagged == 0) & (y == 1)).sum()),
    }


def rates(c: dict[str, int]) -> dict[str, float | None]:
    pos, neg, flagged = c["tp"] + c["fn"], c["fp"] + c["tn"], c["tp"] + c["fp"]
    return {
        "recall": c["tp"] / pos if pos else None,
        "fpr": c["fp"] / neg if neg else None,
        "precision": c["tp"] / flagged if flagged else None,
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """ECE over equal-frequency bins (sample-weighted mean |predicted - observed|)."""
    order = np.argsort(p, kind="mergesort")
    bins = np.array_split(order, n_bins)
    rows, ece = [], 0.0
    for b in bins:
        if len(b) == 0:
            continue
        pred, obs = float(p[b].mean()), float(y[b].mean())
        ece += len(b) / len(p) * abs(pred - obs)
        rows.append({"n": len(b), "mean_predicted": pred, "observed_rate": obs})
    return {"ece": ece, "well_calibrated": ece <= 0.05, "bins": rows}


@dataclass
class OperatingPoint:
    target_fpr: float
    threshold: float
    val_fpr: float
    val_recall: float | None
    test: dict[str, Any]


def evaluate(
    y_val: np.ndarray,
    p_val: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    amount_test: np.ndarray | None = None,
    data_source: str = "UNSPECIFIED",
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "data_source": data_source,
        "denominators": {
            "val": {"n": int(len(y_val)), "positives": int(y_val.sum())},
            "test": {"n": int(len(y_test)), "positives": int(y_test.sum())},
        },
        "val": {
            "roc_auc": float(roc_auc_score(y_val, p_val)),
            "pr_auc": float(average_precision_score(y_val, p_val)),
        },
        "test": {
            "roc_auc": float(roc_auc_score(y_test, p_test)),
            "pr_auc": float(average_precision_score(y_test, p_test)),
        },
        "base_rate_test": float(y_test.mean()),
    }
    points = []
    for target in TARGET_FPRS:
        thr, val_fpr = threshold_for_fpr(y_val, p_val, target)
        val_c = confusion(y_val, (p_val >= thr).astype(int))
        test_c = confusion(y_test, (p_test >= thr).astype(int))
        points.append(
            OperatingPoint(
                target, thr, val_fpr, rates(val_c)["recall"], {**rates(test_c), "confusion": test_c}
            )
        )
    report["operating_points"] = [asdict(p) for p in points]
    report["calibration_test"] = expected_calibration_error(y_test, p_test)

    if amount_test is not None:
        guard = next(p for p in points if p.target_fpr == APPROVED_FPR_GUARDRAIL)
        edges = np.quantile(amount_test, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        strata = []
        for i in range(5):
            lo, hi = edges[i], edges[i + 1]
            m = (amount_test >= lo) & ((amount_test <= hi) if i == 4 else (amount_test < hi))
            c = confusion(y_test[m], (p_test[m] >= guard.threshold).astype(int))
            strata.append(
                {
                    "quintile": i + 1,
                    "amount_range": [float(lo), float(hi)],
                    "n": int(m.sum()),
                    "positives": int(y_test[m].sum()),
                    **rates(c),
                }
            )
        report["amount_quintiles_at_guardrail"] = strata
    return report


def score_cutpoints(p_val: np.ndarray) -> list[float]:
    """101 ascending cut points: risk score k when output >= cutpoint k (validation rank)."""
    qs = np.quantile(p_val, np.linspace(0, 1, 101))
    qs[0] = 0.0  # every output gets at least score 0
    return [float(x) for x in np.maximum.accumulate(qs)]


def score_at_threshold(cutpoints: list[float], threshold: float) -> int:
    """Smallest risk score whose cut point is >= the model-output threshold."""
    for k, c in enumerate(cutpoints):
        if c >= threshold:
            return k
    return 100
