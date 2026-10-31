import json

import numpy as np
import pandas as pd
import pytest

from ml.data.audit import (
    DatasetAuditor,
    FeatureType,
    audit_for_leakage,
    classify_feature_type,
)
from ml.exceptions import DataValidationError


@pytest.fixture
def transactions():
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "isFraud": [0, 1, 0, 0],
            "TransactionDT": [86400, 86500, 90000, 100000],
            "TransactionAmt": [10.0, np.nan, 30.0, 40.0],
            "ProductCD": ["W", "C", "W", None],
            "card1": [1000, 2000, 1000, 3000],
        }
    )


@pytest.fixture
def identity():
    return pd.DataFrame({"TransactionID": [2, 3], "DeviceType": ["mobile", np.nan]})


def test_audit_core_statistics(transactions, identity):
    card = DatasetAuditor().audit(transactions=transactions, identity=identity)

    assert card.join_key == "TransactionID"
    assert card.fraud_rate == pytest.approx(0.25)
    assert card.temporal_column == "TransactionDT"
    assert card.temporal_range == (86400.0, 100000.0)
    assert card.missingness_rates["TransactionAmt"] == pytest.approx(0.25)
    assert card.missingness_rates["DeviceType"] == pytest.approx(0.5)
    assert card.n_transactions == 4
    assert card.identity_coverage == pytest.approx(0.5)


def test_audit_documents_all_columns(transactions, identity):
    card = DatasetAuditor().audit(transactions=transactions, identity=identity)
    assert [c.name for c in card.transaction_columns] == list(transactions.columns)
    assert [c.name for c in card.identity_columns] == list(identity.columns)


def test_feature_types_exclude_label_and_key(transactions, identity):
    card = DatasetAuditor().audit(transactions=transactions, identity=identity)
    assert "isFraud" not in card.feature_types
    assert "TransactionID" not in card.feature_types
    assert card.feature_types["TransactionAmt"] == FeatureType.NUMERIC
    assert card.feature_types["ProductCD"] == FeatureType.CATEGORICAL
    # card1 is an integer column but documented as categorical in IEEE-CIS
    assert card.feature_types["card1"] == FeatureType.CATEGORICAL
    assert card.feature_types["DeviceType"] == FeatureType.CATEGORICAL


def test_join_integrity_violation_raises(transactions):
    bad_identity = pd.DataFrame({"TransactionID": [2, 99]})
    with pytest.raises(DataValidationError, match="not found in transactions"):
        DatasetAuditor().audit(transactions=transactions, identity=bad_identity)


def test_missing_required_column_raises(transactions, identity):
    with pytest.raises(DataValidationError, match="isFraud"):
        DatasetAuditor().audit(transactions=transactions.drop(columns="isFraud"), identity=identity)


def test_schema_records_cardinality_and_range(transactions, identity):
    card = DatasetAuditor().audit(transactions=transactions, identity=identity)
    by_name = {c.name: c for c in card.transaction_columns}
    assert by_name["ProductCD"].cardinality == 2
    assert by_name["TransactionAmt"].numeric_range == (10.0, 40.0)


def test_card_round_trips_to_json(tmp_path, transactions, identity):
    card = DatasetAuditor().audit(transactions=transactions, identity=identity)
    out = tmp_path / "card.json"
    card.save(out)
    loaded = json.loads(out.read_text())
    assert loaded["fraud_rate"] == pytest.approx(0.25)
    assert loaded["feature_types"]["ProductCD"] == "categorical"


def test_audit_from_csv_paths(tmp_path, transactions, identity):
    tx_path, id_path = tmp_path / "tx.csv", tmp_path / "id.csv"
    transactions.to_csv(tx_path, index=False)
    identity.to_csv(id_path, index=False)
    card = DatasetAuditor().audit(transaction_path=tx_path, identity_path=id_path)
    assert card.n_transactions == 4


@pytest.mark.parametrize(
    "series,expected",
    [
        (pd.Series([1.0, 2.0], name="x"), FeatureType.NUMERIC),
        (pd.Series([1, 2], name="x"), FeatureType.NUMERIC),
        (pd.Series(["a", "b"], name="x"), FeatureType.CATEGORICAL),
        (pd.Series([True, False], name="x"), FeatureType.CATEGORICAL),
        (pd.Series([], dtype=float, name="x"), FeatureType.NUMERIC),
        (pd.Series([], dtype=object, name="x"), FeatureType.CATEGORICAL),
        (pd.Series([1, "a"], name="x"), FeatureType.CATEGORICAL),
    ],
)
def test_classify_feature_type_edge_cases(series, expected):
    assert classify_feature_type(series) == expected


class TestLeakageAudit:
    def test_flags_keywords_case_insensitive(self):
        clean, excluded = audit_for_leakage(["amt", "DaysUntilChargeback", "NEXT_txn", "after_x"])
        assert clean == ["amt"]
        assert [e.name for e in excluded] == ["DaysUntilChargeback", "NEXT_txn", "after_x"]
        assert all(e.justification for e in excluded)

    def test_clean_features_pass_through(self):
        clean, excluded = audit_for_leakage(["TransactionAmt", "card1", "D1"])
        assert clean == ["TransactionAmt", "card1", "D1"]
        assert excluded == []

    def test_explicit_exclusions_carry_justification(self):
        clean, excluded = audit_for_leakage(["D1", "D15"], {"D15": "manual review"})
        assert clean == ["D1"]
        assert excluded[0].justification == "manual review"

    def test_excluded_features_recorded_in_card(self, transactions, identity):
        transactions = transactions.assign(future_amt=1.0)
        card = DatasetAuditor().audit(transactions=transactions, identity=identity)
        assert [e.name for e in card.excluded_features] == ["future_amt"]
