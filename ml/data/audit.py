"""IEEE-CIS dataset audit: schema, statistics, temporal info, and leakage screening.

Implements Requirements 1.1-1.7 and 2.1-2.5.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd

from ml.exceptions import DataValidationError

logger = logging.getLogger(__name__)

JOIN_KEY = "TransactionID"
LABEL_COLUMN = "isFraud"
TEMPORAL_COLUMN = "TransactionDT"
TEMPORAL_SEMANTICS = (
    "Seconds elapsed from an undisclosed reference datetime (not a real timestamp). "
    "Monotonic in event time; used for chronological ordering and splitting."
)

# Columns the IEEE-CIS data description documents as categorical even though some are
# stored as integers or floats (e.g. card1 is an integer card identifier).
IEEE_CATEGORICAL_COLUMNS = frozenset(
    ["ProductCD", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]
    + [f"card{i}" for i in range(1, 7)]
    + ["addr1", "addr2"]
    + [f"M{i}" for i in range(1, 10)]
    + [f"id_{i}" for i in range(12, 39)]
)

# Columns that are never model features: the join key and the label.
NON_FEATURE_COLUMNS = frozenset([JOIN_KEY, LABEL_COLUMN])

LEAKAGE_KEYWORDS = ("future", "daysuntil", "next", "after")


class FeatureType(str, Enum):
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"


@dataclass
class ColumnSchema:
    name: str
    dtype: str
    cardinality: int | None = None  # Set for categoricals
    numeric_range: tuple[float, float] | None = None  # Set for numerics


@dataclass
class ExcludedFeature:
    name: str
    justification: str


@dataclass
class DatasetCard:
    transaction_columns: list[ColumnSchema]
    identity_columns: list[ColumnSchema]
    join_key: str
    fraud_rate: float
    temporal_column: str
    temporal_semantics: str
    temporal_range: tuple[float, float]
    missingness_rates: dict[str, float]
    feature_types: dict[str, FeatureType]
    n_transactions: int
    n_identity_rows: int
    identity_coverage: float  # Fraction of transactions with an identity row
    excluded_features: list[ExcludedFeature] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["feature_types"] = {k: v.value for k, v in self.feature_types.items()}
        return d

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))


def compute_missingness(df: pd.DataFrame) -> dict[str, float]:
    """Fraction of missing values per column (0.0 for an empty frame)."""
    if len(df) == 0:
        return {col: 0.0 for col in df.columns}
    return {col: float(rate) for col, rate in df.isna().mean().items()}


def classify_feature_type(
    series: pd.Series, known_categorical: Iterable[str] = IEEE_CATEGORICAL_COLUMNS
) -> FeatureType:
    """Classify a column as CATEGORICAL or NUMERIC.

    Object/string/bool/category dtypes and documented IEEE-CIS categorical columns are
    CATEGORICAL; all other int/float columns are NUMERIC.
    """
    if series.name in set(known_categorical):
        return FeatureType.CATEGORICAL
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return FeatureType.CATEGORICAL
    return FeatureType.NUMERIC


def audit_for_leakage(
    features: Iterable[str],
    extra_exclusions: dict[str, str] | None = None,
) -> tuple[list[str], list[ExcludedFeature]]:
    """Screen feature names for temporal leakage risk.

    Flags names containing future-looking keywords (case-insensitive), plus any
    explicitly listed exclusions with their own justification.

    Returns:
        (clean_features, excluded_features)
    """
    extra_exclusions = extra_exclusions or {}
    clean: list[str] = []
    excluded: list[ExcludedFeature] = []

    for feature in features:
        if feature in extra_exclusions:
            excluded.append(ExcludedFeature(feature, extra_exclusions[feature]))
            continue
        hits = [kw for kw in LEAKAGE_KEYWORDS if kw in feature.lower()]
        if hits:
            justification = (
                f"Name contains leakage keyword(s) {hits}; may encode information "
                "unavailable at inference time."
            )
            excluded.append(ExcludedFeature(feature, justification))
            logger.warning("Excluding feature due to leakage risk: %s", feature)
        else:
            clean.append(feature)

    return clean, excluded


class DatasetAuditor:
    """Audit the IEEE-CIS dataset and generate a Dataset Card."""

    def __init__(self, known_categorical: Iterable[str] = IEEE_CATEGORICAL_COLUMNS):
        self.known_categorical = frozenset(known_categorical)

    def audit(
        self,
        transaction_path: str | Path | None = None,
        identity_path: str | Path | None = None,
        transactions: pd.DataFrame | None = None,
        identity: pd.DataFrame | None = None,
        leakage_exclusions: dict[str, str] | None = None,
    ) -> DatasetCard:
        """Generate dataset documentation from CSV paths or already-loaded frames."""
        if transactions is None:
            if transaction_path is None:
                raise ValueError("Provide transaction_path or transactions")
            transactions = pd.read_csv(transaction_path)
        if identity is None:
            if identity_path is None:
                raise ValueError("Provide identity_path or identity")
            identity = pd.read_csv(identity_path)

        self._validate_columns(transactions, [JOIN_KEY, LABEL_COLUMN, TEMPORAL_COLUMN])
        self._validate_columns(identity, [JOIN_KEY])
        self.verify_join_integrity(transactions, identity)

        missingness = compute_missingness(transactions)
        missingness.update(compute_missingness(identity.drop(columns=[JOIN_KEY])))

        feature_types: dict[str, FeatureType] = {}
        for df in (transactions, identity):
            for col in df.columns:
                if col not in NON_FEATURE_COLUMNS:
                    feature_types[col] = classify_feature_type(df[col], self.known_categorical)

        _, excluded = audit_for_leakage(feature_types.keys(), leakage_exclusions)
        n_tx = len(transactions)

        return DatasetCard(
            transaction_columns=self._get_schema(transactions),
            identity_columns=self._get_schema(identity),
            join_key=JOIN_KEY,
            fraud_rate=float(transactions[LABEL_COLUMN].mean()),
            temporal_column=TEMPORAL_COLUMN,
            temporal_semantics=TEMPORAL_SEMANTICS,
            temporal_range=(
                float(transactions[TEMPORAL_COLUMN].min()),
                float(transactions[TEMPORAL_COLUMN].max()),
            ),
            missingness_rates=missingness,
            feature_types=feature_types,
            n_transactions=n_tx,
            n_identity_rows=len(identity),
            identity_coverage=(
                float(transactions[JOIN_KEY].isin(identity[JOIN_KEY]).mean()) if n_tx else 0.0
            ),
            excluded_features=excluded,
        )

    @staticmethod
    def verify_join_integrity(transactions: pd.DataFrame, identity: pd.DataFrame) -> None:
        """Raise if any identity TransactionID is absent from the transaction table."""
        orphan_mask = ~identity[JOIN_KEY].isin(transactions[JOIN_KEY])
        if orphan_mask.any():
            orphans = identity.loc[orphan_mask, JOIN_KEY]
            raise DataValidationError(
                f"{len(orphans)} identity TransactionIDs not found in transactions "
                f"(e.g. {orphans.head(5).tolist()})"
            )

    @staticmethod
    def _validate_columns(df: pd.DataFrame, expected: list[str]) -> None:
        missing = set(expected) - set(df.columns)
        if missing:
            raise DataValidationError(f"Missing required columns: {sorted(missing)}")

    def _get_schema(self, df: pd.DataFrame) -> list[ColumnSchema]:
        schema = []
        for col in df.columns:
            series = df[col]
            if classify_feature_type(series, self.known_categorical) == FeatureType.CATEGORICAL:
                schema.append(
                    ColumnSchema(col, str(series.dtype), cardinality=int(series.nunique()))
                )
            else:
                non_null = series.dropna()
                rng = (float(non_null.min()), float(non_null.max())) if len(non_null) else None
                schema.append(ColumnSchema(col, str(series.dtype), numeric_range=rng))
        return schema
