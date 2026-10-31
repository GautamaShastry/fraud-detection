"""T-301: audit the IEEE-CIS training files and publish the dataset card + availability matrix.

    python -m ml.audit_ieee --data data/raw/ieee-cis --out evidence/ml

Only the labeled train_* files are audited; the Kaggle test files are unlabeled and support no
performance claim. The licensed CSVs are read in place and never copied into the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ml.data.availability import as_rows, to_markdown
from ml.data.chunked_audit import audit_files


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/ieee-cis"))
    parser.add_argument("--out", type=Path, default=Path("evidence/ml"))
    args = parser.parse_args()

    tx, idn = args.data / "train_transaction.csv", args.data / "train_identity.csv"
    card = audit_files(tx, idn)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = card.to_dict()
    payload["source_files"] = {
        p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in (tx, idn)
    }
    payload["feature_availability"] = as_rows()
    (args.out / "ieee_dataset_card.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    miss = sorted(card.missingness_rates.items(), key=lambda kv: -kv[1])
    span_days = (card.temporal_range[1] - card.temporal_range[0]) / 86400
    n_cat = sum(1 for t in card.feature_types.values() if t.value == "categorical")
    lines = [
        "# IEEE-CIS dataset card (train files)",
        "",
        f"- Transactions: {card.n_transactions:,}; identity rows: {card.n_identity_rows:,} "
        f"(identity coverage {card.identity_coverage:.1%})",
        f"- Join key: `{card.join_key}`; identity IDs are a subset of transaction IDs (verified)",
        f"- Fraud rate (`isFraud`): {card.fraud_rate:.3%}",
        f"- Temporal column: `{card.temporal_column}` range {card.temporal_range[0]:,.0f}–"
        f"{card.temporal_range[1]:,.0f} s ({span_days:.0f} days). "
        f"{card.temporal_semantics}",
        f"- Features: {len(card.feature_types)} ({n_cat} categorical, "
        f"{len(card.feature_types) - n_cat} numeric)",
        f"- Columns >90% missing: {sum(r > 0.9 for _, r in miss)}; >50% missing: "
        f"{sum(r > 0.5 for _, r in miss)}",
        f"- Leakage audit exclusions: {[e.name for e in card.excluded_features] or 'none'}",
        "",
        "## Availability of online API features in IEEE-CIS",
        "",
        to_markdown(),
        "",
        "**Consequence:** the deployed scoring model cannot be trained on IEEE-CIS, because most "
        "online features (including F01) do not exist in it. IEEE-CIS is used for an offline "
        "benchmark of the modeling approach only (see `ieee_benchmark.md`).",
    ]
    (args.out / "ieee_dataset_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:9]))


if __name__ == "__main__":
    main()
