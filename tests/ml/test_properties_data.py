import numpy as np
import pandas as pd
from hypothesis import assume, given
from hypothesis import strategies as st

from ml.data.audit import FeatureType, classify_feature_type, compute_missingness
from ml.data.split import ChronologicalSplitter, apply_split


# Feature: ml-model-pipeline, Property 1: Feature Missingness Rate Accuracy
@st.composite
def nullable_columns(draw):
    n_rows = draw(st.integers(1, 200))
    cell = st.one_of(st.none(), st.floats(allow_nan=False))
    return draw(st.lists(st.lists(cell, min_size=n_rows, max_size=n_rows), min_size=1, max_size=5))


@given(nullable_columns())
def test_missingness_rate_accuracy(columns):
    df = pd.DataFrame({f"c{i}": pd.Series(col, dtype=float) for i, col in enumerate(columns)})
    rates = compute_missingness(df)
    for i, col in enumerate(columns):
        true_rate = sum(v is None for v in col) / len(col)
        assert abs(rates[f"c{i}"] - true_rate) <= 1e-6


# Feature: ml-model-pipeline, Property 2: Feature Type Classification Correctness
@given(
    kind=st.sampled_from(["int", "float", "str"]),
    values=st.lists(st.integers(-1000, 1000), min_size=1, max_size=100),
)
def test_feature_type_classification(kind, values):
    if kind == "int":
        series, expected = pd.Series(values, dtype="int64"), FeatureType.NUMERIC
    elif kind == "float":
        series, expected = pd.Series(values, dtype="float64"), FeatureType.NUMERIC
    else:
        series, expected = pd.Series([f"v{v % 49}" for v in values]), FeatureType.CATEGORICAL
    series.name = "generic_col"
    assert classify_feature_type(series) == expected


timestamps = st.lists(st.integers(0, 10_000_000), min_size=10, max_size=500)
pcts = st.tuples(st.floats(0.1, 0.8), st.floats(0.05, 0.45)).filter(lambda t: t[0] + t[1] < 0.95)


def _frame(ts):
    return pd.DataFrame({"TransactionID": np.arange(len(ts)), "TransactionDT": ts})


# Feature: ml-model-pipeline, Property 3: Dataset Temporal Sorting
@given(timestamps)
def test_temporal_sorting(ts):
    data = _frame(ts)
    split = ChronologicalSplitter().split(data)
    frames = apply_split(data, split)
    combined = pd.concat([frames["train"], frames["val"], frames["test"]])
    assert combined["TransactionDT"].is_monotonic_increasing


# Feature: ml-model-pipeline, Property 4: Split Size Conformance
@given(timestamps, pcts)
def test_split_size_conformance(ts, p):
    train_pct, val_pct = p
    test_pct = 1.0 - train_pct - val_pct
    n = len(ts)
    assume(min(train_pct, val_pct, test_pct) * n >= 1.5)
    split = ChronologicalSplitter().split(_frame(ts), train_pct, val_pct, test_pct)
    for ids, pct in [
        (split.train_ids, train_pct),
        (split.val_ids, val_pct),
        (split.test_ids, test_pct),
    ]:
        assert abs(len(ids) - pct * n) <= max(0.01 * n, 1)


# Feature: ml-model-pipeline, Property 5: Chronological Ordering Across Splits
@given(timestamps)
def test_chronological_ordering_across_splits(ts):
    data = _frame(ts)
    frames = apply_split(data, ChronologicalSplitter().split(data))
    assert frames["train"]["TransactionDT"].max() <= frames["val"]["TransactionDT"].min()
    assert frames["val"]["TransactionDT"].max() <= frames["test"]["TransactionDT"].min()
