"""Chronological train/validation/test splitting (Requirements 3.1-3.6)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml.exceptions import DataValidationError

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class ChronologicalSplit:
    train_ids: list[int]
    val_ids: list[int]
    test_ids: list[int]
    train_pct: float
    val_pct: float
    test_pct: float
    seed: int
    temporal_column: str
    temporal_boundaries: dict[str, tuple[float, float]]

    def ids(self, name: str) -> list[int]:
        return {"train": self.train_ids, "val": self.val_ids, "test": self.test_ids}[name]

    def metadata(self) -> dict:
        return {
            "train_pct": self.train_pct,
            "val_pct": self.val_pct,
            "test_pct": self.test_pct,
            "seed": self.seed,
            "temporal_column": self.temporal_column,
            "temporal_boundaries": {k: list(v) for k, v in self.temporal_boundaries.items()},
            "sizes": {name: len(self.ids(name)) for name in SPLIT_NAMES},
        }

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in SPLIT_NAMES:
            (directory / f"{name}_ids.txt").write_text(
                "\n".join(str(i) for i in self.ids(name)) + "\n"
            )
        (directory / "split_metadata.json").write_text(json.dumps(self.metadata(), indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> ChronologicalSplit:
        directory = Path(directory)
        meta = json.loads((directory / "split_metadata.json").read_text())
        ids = {
            name: [int(x) for x in (directory / f"{name}_ids.txt").read_text().split()]
            for name in SPLIT_NAMES
        }
        return cls(
            train_ids=ids["train"],
            val_ids=ids["val"],
            test_ids=ids["test"],
            train_pct=meta["train_pct"],
            val_pct=meta["val_pct"],
            test_pct=meta["test_pct"],
            seed=meta["seed"],
            temporal_column=meta["temporal_column"],
            temporal_boundaries={k: tuple(v) for k, v in meta["temporal_boundaries"].items()},
        )


class ChronologicalSplitter:
    """Split data chronologically so evaluation reflects predicting the future."""

    def split(
        self,
        data: pd.DataFrame,
        train_pct: float = 0.6,
        val_pct: float = 0.2,
        test_pct: float = 0.2,
        temporal_column: str = "TransactionDT",
        id_column: str = "TransactionID",
        seed: int = 42,
    ) -> ChronologicalSplit:
        """Partition data sorted by time into train/val/test transaction IDs.

        Ties on the temporal column are broken by id_column, so the split is fully
        deterministic. The seed is recorded for provenance; no randomness is used.
        """
        if abs(train_pct + val_pct + test_pct - 1.0) > 1e-6:
            raise ValueError("Split percentages must sum to 1.0")
        if min(train_pct, val_pct, test_pct) <= 0:
            raise ValueError("Split percentages must all be positive")

        n = len(data)
        train_end = int(round(n * train_pct))
        val_end = train_end + int(round(n * val_pct))
        if not (0 < train_end < val_end < n):
            raise DataValidationError(f"Dataset of {n} rows too small for a 3-way split")

        sorted_data = data.sort_values([temporal_column, id_column], kind="mergesort")
        times = sorted_data[temporal_column].to_numpy()
        ids = sorted_data[id_column].to_numpy()

        split = ChronologicalSplit(
            train_ids=ids[:train_end].tolist(),
            val_ids=ids[train_end:val_end].tolist(),
            test_ids=ids[val_end:].tolist(),
            train_pct=train_pct,
            val_pct=val_pct,
            test_pct=test_pct,
            seed=seed,
            temporal_column=temporal_column,
            temporal_boundaries={
                "train_range": (float(times[0]), float(times[train_end - 1])),
                "val_range": (float(times[train_end]), float(times[val_end - 1])),
                "test_range": (float(times[val_end]), float(times[-1])),
            },
        )
        self.verify(split, n)
        return split

    @staticmethod
    def verify(split: ChronologicalSplit, n_total: int, size_tolerance: float = 0.01) -> None:
        """Check split sizes and temporal ordering before training (Req 3.6)."""
        b = split.temporal_boundaries
        if b["train_range"][1] > b["val_range"][0]:
            raise DataValidationError("Training data extends past start of validation")
        if b["val_range"][1] > b["test_range"][0]:
            raise DataValidationError("Validation data extends past start of test")

        sizes = {name: len(split.ids(name)) for name in SPLIT_NAMES}
        if sum(sizes.values()) != n_total:
            raise DataValidationError(f"Split sizes {sizes} do not sum to {n_total}")
        all_ids = split.train_ids + split.val_ids + split.test_ids
        if len(set(all_ids)) != len(all_ids):
            raise DataValidationError("Transaction IDs overlap between splits")

        targets = {"train": split.train_pct, "val": split.val_pct, "test": split.test_pct}
        for name, pct in targets.items():
            # Allow ±1% of the dataset, or one row for tiny datasets
            if abs(sizes[name] - pct * n_total) > max(size_tolerance * n_total, 1):
                raise DataValidationError(
                    f"{name} split has {sizes[name]} rows, expected ~{pct * n_total:.0f}"
                )


def apply_split(
    data: pd.DataFrame, split: ChronologicalSplit, id_column: str = "TransactionID"
) -> dict[str, pd.DataFrame]:
    """Materialize split frames, each sorted chronologically."""
    indexed = data.set_index(id_column, drop=False)
    return {name: indexed.loc[split.ids(name)].reset_index(drop=True) for name in SPLIT_NAMES}
