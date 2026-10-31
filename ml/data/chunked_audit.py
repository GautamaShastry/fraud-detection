"""Memory-bounded IEEE-CIS audit (T-301): streams the CSVs in chunks.

Produces the same DatasetCard as DatasetAuditor without loading the 590k x 394 transaction
table at once.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.data.audit import (
    IEEE_CATEGORICAL_COLUMNS,
    JOIN_KEY,
    LABEL_COLUMN,
    NON_FEATURE_COLUMNS,
    TEMPORAL_COLUMN,
    TEMPORAL_SEMANTICS,
    ColumnSchema,
    DatasetCard,
    FeatureType,
    audit_for_leakage,
    classify_feature_type,
)
from ml.exceptions import DataValidationError

CARDINALITY_CAP = 100_000  # stop tracking distinct values beyond this (reported as >= cap)


class _Accumulator:
    def __init__(self) -> None:
        self.rows = 0
        self.missing: dict[str, int] = defaultdict(int)
        self.dtypes: dict[str, str] = {}
        self.types: dict[str, FeatureType] = {}
        self.distinct: dict[str, set[Any]] = defaultdict(set)
        self.lo: dict[str, float] = {}
        self.hi: dict[str, float] = {}
        self.columns: list[str] = []

    def update(self, chunk: pd.DataFrame) -> None:
        if not self.columns:
            self.columns = list(chunk.columns)
        self.rows += len(chunk)
        for col, n in chunk.isna().sum().items():
            self.missing[col] += int(n)
        for col in chunk.columns:
            series = chunk[col]
            ftype = classify_feature_type(series, IEEE_CATEGORICAL_COLUMNS)
            # A column is categorical if any chunk says so (e.g. all-NaN chunks look numeric).
            if self.types.get(col) != FeatureType.CATEGORICAL:
                self.types[col] = ftype
                self.dtypes[col] = str(series.dtype)
            if ftype == FeatureType.CATEGORICAL or col in IEEE_CATEGORICAL_COLUMNS:
                seen = self.distinct[col]
                if len(seen) < CARDINALITY_CAP:
                    seen.update(series.dropna().unique().tolist())
            if pd.api.types.is_numeric_dtype(series):
                values = series.to_numpy(dtype=np.float64, na_value=np.nan)
                if np.isfinite(values).any():
                    self.lo[col] = min(self.lo.get(col, np.inf), float(np.nanmin(values)))
                    self.hi[col] = max(self.hi.get(col, -np.inf), float(np.nanmax(values)))

    def schema(self) -> list[ColumnSchema]:
        out = []
        for col in self.columns:
            if self.types[col] == FeatureType.CATEGORICAL:
                out.append(ColumnSchema(col, self.dtypes[col], cardinality=len(self.distinct[col])))
            else:
                rng = (self.lo[col], self.hi[col]) if col in self.lo else None
                out.append(ColumnSchema(col, self.dtypes[col], numeric_range=rng))
        return out


def audit_files(
    transaction_path: Path, identity_path: Path, chunksize: int = 50_000
) -> DatasetCard:
    tx = _Accumulator()
    tx_ids: set[int] = set()
    fraud = 0
    t_min, t_max = np.inf, -np.inf
    for chunk in pd.read_csv(transaction_path, chunksize=chunksize, low_memory=False):
        missing = {JOIN_KEY, LABEL_COLUMN, TEMPORAL_COLUMN} - set(chunk.columns)
        if missing:
            raise DataValidationError(f"Missing required columns: {sorted(missing)}")
        tx.update(chunk)
        tx_ids.update(chunk[JOIN_KEY].tolist())
        fraud += int(chunk[LABEL_COLUMN].sum())
        t_min = min(t_min, float(chunk[TEMPORAL_COLUMN].min()))
        t_max = max(t_max, float(chunk[TEMPORAL_COLUMN].max()))

    idn = _Accumulator()
    orphans = 0
    covered = 0
    for chunk in pd.read_csv(identity_path, chunksize=chunksize, low_memory=False):
        idn.update(chunk)
        present = chunk[JOIN_KEY].isin(tx_ids)
        orphans += int((~present).sum())
        covered += int(present.sum())
    if orphans:
        raise DataValidationError(f"{orphans} identity TransactionIDs not found in transactions")

    missingness = {c: tx.missing[c] / tx.rows for c in tx.columns}
    missingness.update({c: idn.missing[c] / idn.rows for c in idn.columns if c != JOIN_KEY})
    feature_types = {
        c: t for acc in (tx, idn) for c, t in acc.types.items() if c not in NON_FEATURE_COLUMNS
    }
    _, excluded = audit_for_leakage(feature_types.keys())
    return DatasetCard(
        transaction_columns=tx.schema(),
        identity_columns=idn.schema(),
        join_key=JOIN_KEY,
        fraud_rate=fraud / tx.rows,
        temporal_column=TEMPORAL_COLUMN,
        temporal_semantics=TEMPORAL_SEMANTICS,
        temporal_range=(t_min, t_max),
        missingness_rates=missingness,
        feature_types=feature_types,
        n_transactions=tx.rows,
        n_identity_rows=idn.rows,
        identity_coverage=covered / tx.rows,
        excluded_features=excluded,
    )
