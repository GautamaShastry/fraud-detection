"""Train the deployable scoring models on simulator data (T-303, T-304, T-307).

The online API's feature set does not exist in IEEE-CIS (see the availability matrix), so the
deployable demo model is trained on seeded synthetic data and labeled SYNTHETIC everywhere. Its
metrics establish nothing about real-world fraud detection.

Labels are what a marketplace would actually observe: CONFIRMED_FRAUD = 1; attempts whose
label maturity window has passed without fraud evidence = 0; immature attempts are excluded.
Ground-truth metrics are reported separately, prefixed "Synthetic:".

Outputs models/<model_version>/ with primary.joblib, fallback.joblib and manifest.json
(approval PENDING; see ml/promote.py).

    python -m ml.train --seed 7 --out models
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.evaluate import APPROVED_FPR_GUARDRAIL, evaluate, score_cutpoints, threshold_for_fpr
from ml.features_offline import featurize
from ml.simulator import DATA_SOURCE, SimConfig, simulate
from src.scoring.artifact import sha256_file
from src.scoring.features import FALLBACK_FEATURE_NAMES, FEATURE_NAMES, FEATURE_SCHEMA, to_row
from src.storage.outcomes import fold

LABEL_MATURITY = timedelta(days=60)  # provisional (I-09): no fraud evidence after 60 days => 0
VERIFY_FPR = 0.02  # provisional (I-07/I-16): verification friction budget
FALLBACK_MIN_TEST_AUC = 0.6  # provisional floor for shipping a fallback at all
SEARCH_SPACE = [
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "max_depth": 6,
        "min_samples_leaf": 50,
        "max_iter": 300,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.1,
        "max_leaf_nodes": 15,
        "max_depth": 4,
        "min_samples_leaf": 100,
        "max_iter": 200,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.03,
        "max_leaf_nodes": 31,
        "max_depth": 7,
        "min_samples_leaf": 30,
        "max_iter": 500,
        "l2_regularization": 1.0,
    },
]


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def observed_labels(
    requests: list[dict[str, Any]], outcomes: list[dict[str, Any]], as_of: datetime
) -> np.ndarray:
    """1 = confirmed fraud, 0 = matured without fraud evidence, -1 = immature (excluded)."""
    by_attempt: dict[str, list[dict[str, Any]]] = {}
    for e in outcomes:
        if _ts(e["occurred_at"]) <= as_of:  # only evidence known at training time
            by_attempt.setdefault(e["payment_attempt_id"], []).append(
                {**e, "occurred_at": _ts(e["occurred_at"]), "received_at": _ts(e["occurred_at"])}
            )
    labels = []
    for r in requests:
        p = fold(by_attempt.get(r["payment_attempt_id"], []))
        if p.label == "CONFIRMED_FRAUD":
            labels.append(1)
        elif p.label == "CONFIRMED_LEGITIMATE" or _ts(r["occurred_at"]) + LABEL_MATURITY <= as_of:
            labels.append(0)
        else:
            labels.append(-1)
    return np.array(labels)


def hgb(seed: int, params: dict[str, Any]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        class_weight="balanced", random_state=seed, early_stopping=False, **params
    )


def tune(
    X: dict[str, np.ndarray],  # noqa: N803
    y: dict[str, np.ndarray],
    seed: int,
) -> tuple[HistGradientBoostingClassifier, list[dict[str, Any]]]:
    trials, best = [], None
    for params in SEARCH_SPACE:
        model = hgb(seed, params).fit(X["train"], y["train"])
        auc = float(roc_auc_score(y["val"], model.predict_proba(X["val"])[:, 1]))
        trials.append({"params": params, "val_auc": auc})
        if best is None or auc > best[0]:
            best = (auc, model)
    assert best is not None
    return best[1], trials


def policy(
    version: str, p_val: np.ndarray, y_val: np.ndarray, allow_decline: bool
) -> dict[str, Any]:
    verify, verify_fpr = threshold_for_fpr(y_val, p_val, VERIFY_FPR)
    decline, decline_fpr = threshold_for_fpr(y_val, p_val, APPROVED_FPR_GUARDRAIL)
    return {
        "version": version,
        "cutpoints": score_cutpoints(p_val),
        "verify_threshold": verify,
        "decline_threshold": decline if allow_decline else None,
        "derivation": {
            "verify": {
                "target_val_fpr": VERIFY_FPR,
                "achieved_val_fpr": verify_fpr,
                "status": "provisional (I-07/I-16)",
            },
            "decline": (
                {
                    "target_val_fpr": APPROVED_FPR_GUARDRAIL,
                    "achieved_val_fpr": decline_fpr,
                    "status": "approved guardrail <=0.5% FPR",
                }
                if allow_decline
                else {"status": "fallback never declines (I-08)"}
            ),
        },
    }


def git_commit() -> str:
    try:
        return subprocess.run(  # noqa: S603 - fixed git command
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train(seed: int, out: Path, days: int = 90) -> Path:
    config = SimConfig(seed=seed, days=days)
    sim = simulate(config)
    features = featurize(sim.requests, config.tenant_id)
    as_of = _ts(sim.requests[-1]["occurred_at"]) + timedelta(days=30)
    observed = observed_labels(sim.requests, sim.outcomes, as_of)
    truth = np.array([int(t["is_fraud"]) for t in sim.truth])
    amounts = np.array([r["amount_minor"] for r in sim.requests], dtype=float)

    n = len(features)
    cut1, cut2 = int(n * 0.6), int(n * 0.8)
    idx = {"train": np.arange(0, cut1), "val": np.arange(cut1, cut2), "test": np.arange(cut2, n)}
    # Train/val use observed, matured labels only; test is scored against both observed
    # (where available) and ground truth.
    usable = {k: v[observed[v] >= 0] for k, v in idx.items()}

    def matrix(names: tuple[str, ...], rows: np.ndarray) -> np.ndarray:
        return np.array([to_row(features[i], names) for i in rows], dtype=float)

    results: dict[str, Any] = {
        "data_source": DATA_SOURCE,
        "label_policy": {
            "positive": "CONFIRMED_FRAUD",
            "negative": f"no fraud evidence after {LABEL_MATURITY.days} days",
            "as_of": as_of.isoformat(),
            "excluded_immature": {k: int((observed[v] < 0).sum()) for k, v in idx.items()},
        },
    }
    artifacts: dict[str, Any] = {}

    for name, names, allow_decline in (
        ("primary", FEATURE_NAMES, True),
        ("fallback", FALLBACK_FEATURE_NAMES, False),
    ):
        X = {k: matrix(names, usable[k]) for k in ("train", "val")}
        y = {k: observed[usable[k]] for k in ("train", "val")}
        model, trials = tune(X, y, seed)
        p_val = model.predict_proba(X["val"])[:, 1]
        X_test = matrix(names, idx["test"])
        p_test = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "synthetic_ground_truth": evaluate(
                y["val"],
                p_val,
                truth[idx["test"]],
                p_test,
                amounts[idx["test"]],
                data_source=DATA_SOURCE,
            ),
            "search_trials": trials,
            "n_trees": int(model.n_iter_),
        }
        artifacts[name] = (model, names, policy(f"{name}-policy", p_val, y["val"], allow_decline))

    # Interpretable baseline (L2 logistic regression) for comparison only.
    X = {k: matrix(FEATURE_NAMES, usable[k]) for k in ("train", "val")}
    base = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=1000, random_state=seed
                ),
            ),
        ]
    )
    base.fit(X["train"], observed[usable["train"]])
    results["baseline_logistic_regression"] = {
        "synthetic_ground_truth": evaluate(
            observed[usable["val"]],
            base.predict_proba(X["val"])[:, 1],
            truth[idx["test"]],
            base.predict_proba(matrix(FEATURE_NAMES, idx["test"]))[:, 1],
            amounts[idx["test"]],
            data_source=DATA_SOURCE,
        )
    }
    # Ablation: the fallback is the primary without F01.
    results["ablation_f01"] = {
        "with_f01_test_auc": results["primary"]["synthetic_ground_truth"]["test"]["roc_auc"],
        "without_f01_test_auc": results["fallback"]["synthetic_ground_truth"]["test"]["roc_auc"],
    }
    fallback_auc = results["fallback"]["synthetic_ground_truth"]["test"]["roc_auc"]
    ship_fallback = fallback_auc >= FALLBACK_MIN_TEST_AUC

    # Signal thresholds: 99th percentile of the feature among matured legitimate training rows.
    legit = usable["train"][observed[usable["train"]] == 0]
    signal_rules = []
    for feature, code, description, floor in (
        (
            "device_distinct_payment_methods_5m",
            "DEVICE_DISTINCT_PAYMENT_METHODS_5M",
            "Unusually many distinct payment methods on this device in 5 minutes",
            2.0,
        ),
        ("amount_log", "UNUSUAL_AMOUNT", "Amount is unusually high for this marketplace", 0.0),
    ):
        values = [features[i][feature] for i in legit if features[i][feature] is not None]
        threshold = max(float(np.quantile(values, 0.99)), floor) if values else floor
        signal_rules.append(
            {"feature": feature, "threshold": threshold, "code": code, "description": description}
        )

    stamp = datetime.now(UTC)
    model_version = f"synthetic-hgb-s{seed}-{stamp:%Y%m%d%H%M%S}"
    directory = out / model_version
    directory.mkdir(parents=True, exist_ok=False)
    entries: dict[str, Any] = {}
    for name, (model, names, _policy) in artifacts.items():
        if name == "fallback" and not ship_fallback:
            continue
        path = directory / f"{name}.joblib"
        joblib.dump(model, path, compress=3)
        entries[name] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "feature_names": list(names),
            "model_version": model_version if name == "primary" else f"{model_version}-fallback",
        }

    manifest = {
        "model_version": model_version,
        "feature_schema": FEATURE_SCHEMA,
        "data_source": DATA_SOURCE,
        "primary": entries["primary"],
        "fallback": entries.get("fallback"),
        "policy": {**artifacts["primary"][2], "version": f"{model_version}-policy"},
        "fallback_policy": (
            {**artifacts["fallback"][2], "version": f"{model_version}-fb-policy"}
            if ship_fallback
            else None
        ),
        "signal_rules": signal_rules,
        "training": {
            "timestamp": stamp.isoformat(),
            "git_commit": git_commit(),
            "seeds": {"master": seed, "simulator": config.seed, "model": seed},
            "simulator_config": config.__dict__,
            "split": {
                "method": "chronological",
                "train_pct": 0.6,
                "val_pct": 0.2,
                "test_pct": 0.2,
                "sizes": {k: int(len(v)) for k, v in idx.items()},
            },
            "algorithm": "sklearn.HistGradientBoostingClassifier (class_weight=balanced)",
            "fallback_shipped": ship_fallback,
        },
        "metrics": results,
        "library_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "approval": {"status": "PENDING"},
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float), encoding="utf-8"
    )
    (directory / "report.md").write_text(render(manifest), encoding="utf-8")
    return directory


def _pt(rep: dict[str, Any], fpr: float) -> dict[str, Any]:
    return next(p for p in rep["operating_points"] if p["target_fpr"] == fpr)


def render(m: dict[str, Any]) -> str:
    r = m["metrics"]
    rows = []
    for name in ("primary", "fallback", "baseline_logistic_regression"):
        rep = r[name]["synthetic_ground_truth"]
        g = _pt(rep, APPROVED_FPR_GUARDRAIL)["test"]
        rows.append(
            f"| {name} | {rep['test']['roc_auc']:.4f} | {rep['test']['pr_auc']:.4f} | "
            f"{g['recall']:.1%} | {g['fpr']:.3%} | {(g['precision'] or 0):.1%} |"
        )
    return (
        "\n".join(
            [
                f"# Model report: {m['model_version']}",
                "",
                "> **Performance on synthetic data does not represent real-world commercial "
                "results.**",
                "",
                f"Data source: {m['data_source']} (seeded simulator). "
                "Train/val labels: observed and "
                f"matured ({r['label_policy']['negative']}). Test: synthetic ground truth.",
                "",
                "| Synthetic: model | ROC-AUC | PR-AUC | Recall @0.5% FPR (val thr) | FPR "
                "| Precision |",
                "| --- | --- | --- | --- | --- | --- |",
                *rows,
                "",
                "Synthetic: F01 ablation test ROC-AUC with "
                f"{r['ablation_f01']['with_f01_test_auc']:.4f} / "
                f"without {r['ablation_f01']['without_f01_test_auc']:.4f}.",
                f"Fallback shipped: {m['training']['fallback_shipped']}.",
            ]
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()
    directory = train(args.seed, args.out, args.days)
    print(f"Wrote {directory}")
    print((directory / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
