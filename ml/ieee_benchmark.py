"""Offline IEEE-CIS benchmark (T-304 on real labeled data). NOT a deployable model.

Chronological 60/20/20 split on TransactionDT. Two models on genuine IEEE-CIS columns only:
an interpretable L2 logistic-regression baseline and CPU gradient-boosted trees (I-06 proposed
default). Thresholds are chosen on validation and applied unchanged to test.

The deployed API model cannot use these columns (see the availability matrix), so these numbers
benchmark the modeling approach; they are not the production model's performance.

    python -m ml.ieee_benchmark --data data/raw/ieee-cis --out evidence/ml --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler  # noqa: I001

from ml.data.split import ChronologicalSplitter, apply_split
from ml.evaluate import APPROVED_FPR_GUARDRAIL, evaluate

CATEGORICAL = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType"] + [
    f"M{i}" for i in range(1, 10)
]
# Integer-coded card/address attributes with thousands of values: used as numeric codes by the
# trees (HistGradientBoosting supports at most 255 categories), excluded from the linear model.
CODED = ["card1", "card2", "card3", "card5", "addr1", "addr2"]
BASE_NUMERIC = (
    ["TransactionAmt", "dist1"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
    + [f"id_{i:02d}" for i in range(1, 12)]
)
MAX_V_COLUMNS = 60  # memory bound for the local benchmark; the V block is highly redundant
LEGIT_TRAIN_SAMPLE = 0.25  # negative downsampling of the training split only


def select_v_columns(card_path: Path) -> list[str]:
    """V columns with the lowest missingness, from the published dataset card."""
    miss = json.loads(card_path.read_text(encoding="utf-8"))["missingness_rates"]
    vs = sorted((c for c in miss if c.startswith("V")), key=lambda c: (miss[c], int(c[1:])))
    return sorted(vs[:MAX_V_COLUMNS], key=lambda c: int(c[1:]))


def load(data: Path, numeric: list[str]) -> pd.DataFrame:
    idn_numeric = [c for c in numeric if c.startswith("id_")]
    tx_numeric = [c for c in numeric if not c.startswith("id_")]
    tx_cols = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        *tx_numeric,
        *CODED,
        *[c for c in CATEGORICAL if c != "DeviceType"],
    ]
    tx = pd.read_csv(
        data / "train_transaction.csv",
        usecols=tx_cols,
        dtype={c: "float32" for c in tx_numeric + CODED},
    )
    idn = pd.read_csv(
        data / "train_identity.csv",
        usecols=["TransactionID", "DeviceType", *idn_numeric],
        dtype={c: "float32" for c in idn_numeric},
    )
    return tx.merge(idn, on="TransactionID", how="left")


def hour_proxy(dt: pd.Series) -> pd.Series:
    # Hour within the (unknown-offset) day; documented as PARTIAL in the availability matrix.
    return ((dt // 3600) % 24).astype("float32")


def baseline(seed: int, numeric: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )
    lr = LogisticRegression(
        penalty="l2",
        C=0.1,
        class_weight="balanced",
        max_iter=300,
        solver="lbfgs",
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("lr", lr)])


def gbt(seed: int, params: dict[str, Any]) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                    min_frequency=20,
                    dtype=np.float32,
                ),
                CATEGORICAL,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingClassifier(
        categorical_features=list(range(len(CATEGORICAL))),
        class_weight="balanced",
        random_state=seed,
        early_stopping=False,
        **params,
    )
    return Pipeline([("pre", pre), ("gbt", model)])


SEARCH_SPACE = [  # random search would sample these; a fixed grid keeps runs reproducible
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "max_depth": None,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
        "max_iter": 400,
    },
    {
        "learning_rate": 0.1,
        "max_leaf_nodes": 31,
        "max_depth": 7,
        "min_samples_leaf": 50,
        "l2_regularization": 0.0,
        "max_iter": 300,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "max_depth": 6,
        "min_samples_leaf": 200,
        "l2_regularization": 1.0,
        "max_iter": 500,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/ieee-cis"))
    parser.add_argument("--out", type=Path, default=Path("evidence/ml"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    v_cols = select_v_columns(args.out / "ieee_dataset_card.json")
    numeric = BASE_NUMERIC + v_cols
    df = load(args.data, numeric)
    df["hour_proxy"] = hour_proxy(df["TransactionDT"])
    numeric = numeric + ["hour_proxy"]
    for c in CATEGORICAL:
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    split = ChronologicalSplitter().split(df, seed=args.seed)
    parts = apply_split(df, split)
    del df
    rng = np.random.default_rng(args.seed)
    train = parts["train"]
    keep = (train["isFraud"] == 1) | (rng.random(len(train)) < LEGIT_TRAIN_SAMPLE)
    parts["train"] = train[keep.to_numpy()].reset_index(drop=True)
    del train

    gbt_features = CATEGORICAL + CODED + numeric
    lr_features = numeric + CATEGORICAL
    y = {k: v["isFraud"].to_numpy() for k, v in parts.items()}
    print(
        f"loaded + split in {time.time() - t0:.0f}s; sizes "
        f"{ {k: len(v) for k, v in y.items()} }",
        flush=True,
    )

    results: dict[str, Any] = {
        "split": split.metadata(),
        "seed": args.seed,
        "features": {"categorical": CATEGORICAL, "coded": CODED, "numeric": numeric},
        "training_sample": {
            "legit_fraction_kept": LEGIT_TRAIN_SAMPLE,
            "train_rows_used": int(len(y["train"])),
            "note": "validation and test splits are complete",
        },
    }

    t = time.time()
    base = baseline(args.seed, numeric).fit(parts["train"][lr_features], y["train"])
    pv = base.predict_proba(parts["val"][lr_features])[:, 1]
    pt = base.predict_proba(parts["test"][lr_features])[:, 1]
    results["baseline_logistic_regression"] = evaluate(
        y["val"],
        pv,
        y["test"],
        pt,
        parts["test"]["TransactionAmt"].to_numpy(),
        data_source="IEEE-CIS train (chronological holdout)",
    )
    results["baseline_logistic_regression"]["train_seconds"] = time.time() - t
    print(
        f"baseline val AUC {results['baseline_logistic_regression']['val']['roc_auc']:.4f}",
        flush=True,
    )
    del base
    X = {k: v[gbt_features] for k, v in parts.items()}

    trials = []
    best = None
    for params in SEARCH_SPACE:
        t = time.time()
        model = gbt(args.seed, params).fit(X["train"], y["train"])
        val_auc = roc_auc_score(y["val"], model.predict_proba(X["val"])[:, 1])
        trials.append({"params": params, "val_auc": val_auc, "seconds": time.time() - t})
        print(f"gbt {params} val AUC {val_auc:.4f} ({time.time() - t:.0f}s)", flush=True)
        if best is None or val_auc > best[0]:
            best = (val_auc, params, model)
    assert best is not None
    _, best_params, model = best
    pv, pt = model.predict_proba(X["val"])[:, 1], model.predict_proba(X["test"])[:, 1]
    results["gbt"] = evaluate(
        y["val"],
        pv,
        y["test"],
        pt,
        parts["test"]["TransactionAmt"].to_numpy(),
        data_source="IEEE-CIS train (chronological holdout)",
    )
    results["gbt"]["best_params"] = best_params
    results["gbt"]["search_trials"] = trials
    results["gbt"]["n_trees"] = int(model.named_steps["gbt"].n_iter_)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ieee_benchmark.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8"
    )
    (args.out / "ieee_benchmark.md").write_text(render(results), encoding="utf-8")
    print(render(results))


def _point(report: dict[str, Any], fpr: float) -> dict[str, Any]:
    return next(p for p in report["operating_points"] if p["target_fpr"] == fpr)


def render(r: dict[str, Any]) -> str:
    rows = []
    for name in ("baseline_logistic_regression", "gbt"):
        rep = r[name]
        g = _point(rep, APPROVED_FPR_GUARDRAIL)
        rows.append(
            f"| {name} | {rep['test']['roc_auc']:.4f} | {rep['test']['pr_auc']:.4f} | "
            f"{g['test']['recall']:.1%} | {g['test']['fpr']:.3%} | "
            f"{(g['test']['precision'] or 0):.1%} | {rep['calibration_test']['ece']:.3f} |"
        )
    g = _point(r["gbt"], APPROVED_FPR_GUARDRAIL)["test"]["recall"]
    verdict = ("meets" if g >= 0.6 else "does NOT meet") + " the initial 60% recall target"
    return (
        "\n".join(
            [
                "# IEEE-CIS offline benchmark (chronological holdout)",
                "",
                "Data: IEEE-CIS `train_*` files only, split 60/20/20 by `TransactionDT`. "
                "Threshold for "
                "the 0.5% FPR guardrail chosen on validation, applied unchanged to test. **Offline "
                "benchmark of the modeling approach; the deployed API model cannot use these "
                "columns.**",
                "",
                "| Model | Test ROC-AUC | Test PR-AUC | Test recall @0.5% FPR (val threshold) | "
                "Test FPR achieved | Precision | ECE |",
                "| --- | --- | --- | --- | --- | --- | --- |",
                *rows,
                "",
                f"Gradient-boosted trees {verdict} at the 0.5% FPR guardrail (measured: {g:.1%}).",
                "",
                "Recall at other validation-selected FPR targets (test):",
                "",
                "| Model | 0.1% | 0.5% | 1% | 2% |",
                "| --- | --- | --- | --- | --- |",
                *[
                    f"| {n} | "
                    + " | ".join(
                        f"{_point(r[n], f)['test']['recall']:.1%}"
                        for f in (0.001, 0.005, 0.01, 0.02)
                    )
                    + " |"
                    for n in ("baseline_logistic_regression", "gbt")
                ],
                "",
                "Raw ECE is reported for class-weighted models; their outputs are rankings, not "
                "calibrated probabilities (I-17).",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
