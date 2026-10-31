import pandas as pd
import pytest

from ml.data.split import ChronologicalSplit, ChronologicalSplitter, apply_split
from ml.exceptions import DataValidationError


@pytest.fixture
def data():
    # Deliberately unsorted timestamps
    return pd.DataFrame(
        {"TransactionID": range(100, 110), "TransactionDT": [9, 3, 7, 1, 5, 8, 2, 6, 0, 4]}
    )


def test_exact_sizes_and_boundaries(data):
    split = ChronologicalSplitter().split(data)
    assert (len(split.train_ids), len(split.val_ids), len(split.test_ids)) == (6, 2, 2)
    assert split.temporal_boundaries == {
        "train_range": (0.0, 5.0),
        "val_range": (6.0, 7.0),
        "test_range": (8.0, 9.0),
    }
    assert split.train_ids == [108, 103, 106, 101, 109, 104]


def test_reproducible(data):
    a = ChronologicalSplitter().split(data, seed=1)
    b = ChronologicalSplitter().split(data.sample(frac=1, random_state=0), seed=1)
    assert (a.train_ids, a.val_ids, a.test_ids) == (b.train_ids, b.val_ids, b.test_ids)


def test_ties_broken_by_id():
    data = pd.DataFrame({"TransactionID": [5, 4, 3, 2, 1], "TransactionDT": [0] * 5})
    split = ChronologicalSplitter().split(data)
    assert split.train_ids + split.val_ids + split.test_ids == [1, 2, 3, 4, 5]


def test_minimum_dataset_size():
    tiny = pd.DataFrame({"TransactionID": [1, 2], "TransactionDT": [0, 1]})
    with pytest.raises(DataValidationError):
        ChronologicalSplitter().split(tiny)
    three = pd.DataFrame({"TransactionID": [1, 2, 3], "TransactionDT": [0, 1, 2]})
    split = ChronologicalSplitter().split(three, 0.34, 0.33, 0.33)
    assert (split.train_ids, split.val_ids, split.test_ids) == ([1], [2], [3])


def test_extreme_ratios():
    data = pd.DataFrame({"TransactionID": range(1000), "TransactionDT": range(1000)})
    split = ChronologicalSplitter().split(data, 0.98, 0.01, 0.01)
    assert (len(split.train_ids), len(split.val_ids), len(split.test_ids)) == (980, 10, 10)


def test_invalid_percentages(data):
    with pytest.raises(ValueError):
        ChronologicalSplitter().split(data, 0.5, 0.2, 0.2)


def test_save_load_round_trip(tmp_path, data):
    split = ChronologicalSplitter().split(data, seed=7)
    split.save(tmp_path)
    assert (tmp_path / "train_ids.txt").exists()
    loaded = ChronologicalSplit.load(tmp_path)
    assert loaded == split


def test_verify_detects_temporal_overlap(data):
    split = ChronologicalSplitter().split(data)
    split.temporal_boundaries["train_range"] = (0.0, 6.5)
    with pytest.raises(DataValidationError, match="validation"):
        ChronologicalSplitter.verify(split, len(data))


def test_apply_split(data):
    split = ChronologicalSplitter().split(data)
    frames = apply_split(data, split)
    assert frames["test"]["TransactionDT"].tolist() == [8, 9]
    assert frames["train"]["TransactionDT"].is_monotonic_increasing
