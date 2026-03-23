"""
Comprehensive pytest unit tests for fraud_risk_scoring.py module.

Tests cover all public functions with multiple test cases including:
- Normal/expected inputs
- Edge cases and boundary conditions
- Score caps and risk level boundaries
- Empty DataFrames where applicable
"""

import os
import tempfile

import pandas as pd
import pytest

# Ensure the repo root is importable
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fraud_risk_scoring import (
    assign_risk_level,
    compute_amount_risk,
    compute_balance_anomaly_risk,
    compute_cashout_pattern_risk,
    compute_destination_risk,
    compute_repeat_account_risk,
    compute_type_risk,
    generate_risk_report,
    load_dataset,
)


# ---------------------------------------------------------------------------
# Helper to build minimal DataFrames
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with all required columns, filling missing ones with defaults."""
    columns = [
        "transaction_id", "step", "type", "amount", "nameOrig",
        "oldbalanceOrg", "newbalanceOrig", "nameDest",
        "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
    ]
    defaults = {
        "transaction_id": 0,
        "step": 1,
        "type": "PAYMENT",
        "amount": 100.0,
        "nameOrig": "C1000",
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 9900.0,
        "nameDest": "C2000",
        "oldbalanceDest": 5000.0,
        "newbalanceDest": 5100.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }
    if not rows:
        return pd.DataFrame(columns=columns)
    full_rows = []
    for i, row in enumerate(rows):
        r = {**defaults, **row}
        r.setdefault("transaction_id", i)
        full_rows.append(r)
    df = pd.DataFrame(full_rows, columns=columns)
    return df


# ===========================================================================
# Tests for load_dataset
# ===========================================================================


class TestLoadDataset:
    """Tests for the load_dataset function."""

    def test_load_creates_transaction_id(self, tmp_path):
        """load_dataset should add a sequential transaction_id column."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,"
            "nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n"
            "1,PAYMENT,100.0,C1,1000,900,M1,0,0,0,0\n"
            "2,TRANSFER,5000.0,C2,5000,0,C3,1000,6000,0,0\n"
        )
        df = load_dataset(str(csv_path))
        assert "transaction_id" in df.columns
        assert list(df["transaction_id"]) == [0, 1]

    def test_load_preserves_columns(self, tmp_path):
        """All original columns should be preserved."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,"
            "nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n"
            "1,PAYMENT,100.0,C1,1000,900,M1,0,0,0,0\n"
        )
        df = load_dataset(str(csv_path))
        expected_cols = {
            "transaction_id", "step", "type", "amount", "nameOrig",
            "oldbalanceOrg", "newbalanceOrig", "nameDest",
            "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_load_empty_csv(self, tmp_path):
        """Loading a CSV with headers but no rows should return an empty DataFrame."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text(
            "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,"
            "nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n"
        )
        df = load_dataset(str(csv_path))
        assert len(df) == 0
        assert "transaction_id" in df.columns

    def test_load_nonexistent_file_raises(self):
        """Loading a file that does not exist should raise."""
        with pytest.raises(Exception):
            load_dataset("/nonexistent/path/file.csv")


# ===========================================================================
# Tests for compute_amount_risk
# ===========================================================================


class TestComputeAmountRisk:
    """Tests for the compute_amount_risk function."""

    @pytest.mark.parametrize(
        "amount, expected_score",
        [
            (500001, 25.0),
            (1000000, 25.0),
            (200001, 20.0),
            (500000, 20.0),
            (100001, 15.0),
            (200000, 15.0),
            (10001, 10.0),
            (100000, 10.0),
            (5001, 5.0),
            (10000, 5.0),
            (5000, 0.0),
            (100, 0.0),
            (0, 0.0),
        ],
    )
    def test_amount_thresholds(self, amount, expected_score):
        """Score should correspond to the correct threshold bracket."""
        score, reason = compute_amount_risk(amount)
        assert score == expected_score

    def test_very_high_amount_reason(self):
        """Reason string should be populated for high amounts."""
        score, reason = compute_amount_risk(600000)
        assert score == 25.0
        assert reason != ""

    def test_zero_amount(self):
        """Zero amount should return score 0 and empty reason."""
        score, reason = compute_amount_risk(0)
        assert score == 0.0
        assert reason == ""

    def test_negative_amount(self):
        """Negative amount should return score 0."""
        score, reason = compute_amount_risk(-100)
        assert score == 0.0

    def test_boundary_exactly_500000(self):
        """500000 is NOT > 500000, so it should fall to the 200k bracket."""
        score, _ = compute_amount_risk(500000)
        assert score == 20.0

    def test_boundary_exactly_200000(self):
        """200000 is NOT > 200000, so it should fall to 100k bracket."""
        score, _ = compute_amount_risk(200000)
        assert score == 15.0

    def test_boundary_exactly_100000(self):
        """100000 is NOT > 100000, so it should fall to 10k bracket."""
        score, _ = compute_amount_risk(100000)
        assert score == 10.0

    def test_boundary_exactly_10000(self):
        """10000 is NOT > 10000, so it should fall to 5k bracket."""
        score, _ = compute_amount_risk(10000)
        assert score == 5.0

    def test_boundary_exactly_5000(self):
        """5000 is NOT > 5000, so it should return 0."""
        score, _ = compute_amount_risk(5000)
        assert score == 0.0


# ===========================================================================
# Tests for compute_type_risk
# ===========================================================================


class TestComputeTypeRisk:
    """Tests for the compute_type_risk function."""

    @pytest.mark.parametrize(
        "txn_type, expected_score",
        [
            ("CASH_OUT", 20.0),
            ("TRANSFER", 15.0),
            ("DEBIT", 5.0),
            ("PAYMENT", 0.0),
            ("CASH_IN", 0.0),
        ],
    )
    def test_known_types(self, txn_type, expected_score):
        """Each known transaction type should return the correct score."""
        score, reason = compute_type_risk(txn_type)
        assert score == expected_score

    def test_unknown_type_returns_zero(self):
        """Unknown transaction types should return 0 score and empty reason."""
        score, reason = compute_type_risk("UNKNOWN_TYPE")
        assert score == 0.0
        assert reason == ""

    def test_cash_out_has_reason(self):
        """CASH_OUT should have a non-empty reason string."""
        _, reason = compute_type_risk("CASH_OUT")
        assert "CASH_OUT" in reason

    def test_payment_has_empty_reason(self):
        """PAYMENT should have an empty reason string."""
        _, reason = compute_type_risk("PAYMENT")
        assert reason == ""


# ===========================================================================
# Tests for compute_balance_anomaly_risk
# ===========================================================================


class TestComputeBalanceAnomalyRisk:
    """Tests for the compute_balance_anomaly_risk function."""

    def test_no_anomaly(self):
        """Normal transaction with consistent balances should score 0."""
        row = pd.Series({
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 9000.0,
            "amount": 1000.0,
            "oldbalanceDest": 5000.0,
            "newbalanceDest": 6000.0,
            "nameDest": "C2000",
        })
        score, reason = compute_balance_anomaly_risk(row)
        assert score == 0.0
        assert reason == ""

    def test_account_drained(self):
        """Account drained to zero should add 10 points."""
        row = pd.Series({
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 0.0,
            "amount": 10000.0,
            "oldbalanceDest": 5000.0,
            "newbalanceDest": 15000.0,
            "nameDest": "C2000",
        })
        score, reason = compute_balance_anomaly_risk(row)
        assert score >= 10.0
        assert "drained" in reason.lower()

    def test_balance_discrepancy(self):
        """Balance mismatch at origin should add 5 points."""
        row = pd.Series({
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 8000.0,  # expected: 10000 - 1000 = 9000
            "amount": 1000.0,
            "oldbalanceDest": 5000.0,
            "newbalanceDest": 6000.0,
            "nameDest": "C2000",
        })
        score, reason = compute_balance_anomaly_risk(row)
        assert score == 5.0
        assert "discrepancy" in reason.lower()

    def test_zero_initial_balance(self):
        """Transaction from zero-balance account should add 5 points."""
        row = pd.Series({
            "oldbalanceOrg": 0.0,
            "newbalanceOrig": 0.0,
            "amount": 5000.0,
            "oldbalanceDest": 1000.0,
            "newbalanceDest": 6000.0,
            "nameDest": "C2000",
        })
        score, reason = compute_balance_anomaly_risk(row)
        assert score == 5.0
        assert "zero initial balance" in reason.lower()

    def test_dest_balance_dropped(self):
        """Destination balance dropping to zero should add 5 points."""
        row = pd.Series({
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 9000.0,
            "amount": 1000.0,
            "oldbalanceDest": 5000.0,
            "newbalanceDest": 0.0,
            "nameDest": "C2000",
        })
        score, reason = compute_balance_anomaly_risk(row)
        assert score == 5.0
        assert "destination" in reason.lower()

    def test_merchant_dest_excluded(self):
        """Merchant destinations (M prefix) should not trigger dest anomaly."""
        row = pd.Series({
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 9000.0,
            "amount": 1000.0,
            "oldbalanceDest": 5000.0,
            "newbalanceDest": 0.0,
            "nameDest": "M1000",  # Merchant
        })
        score, reason = compute_balance_anomaly_risk(row)
        assert score == 0.0

    def test_multiple_anomalies_capped_at_20(self):
        """Combined anomalies should be capped at 20."""
        # Triggers: account drained (10) + balance discrepancy (5) + dest dropped (5) = 20
        row = pd.Series({
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 0.0,  # drained (10) + discrepancy: 10000-5000 != 0 (5)
            "amount": 5000.0,
            "oldbalanceDest": 5000.0,
            "newbalanceDest": 0.0,  # dest dropped (5)
            "nameDest": "C2000",
        })
        score, reason = compute_balance_anomaly_risk(row)
        assert score == 20.0

    def test_all_anomalies_still_capped(self):
        """Even if all signals fire, score should not exceed 20."""
        # drained (10) + discrepancy (5) + zero initial would not apply since old > 0
        # but dest dropped (5) => total 20, capped at 20
        row = pd.Series({
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 0.0,
            "amount": 3000.0,  # discrepancy: 10000 - 3000 = 7000 != 0 (5)
            "oldbalanceDest": 2000.0,
            "newbalanceDest": 0.0,  # dest dropped (5)
            "nameDest": "C2000",
        })
        score, reason = compute_balance_anomaly_risk(row)
        assert score <= 20.0

    def test_dest_zero_but_was_already_zero(self):
        """If dest old balance is 0 and new is 0, should NOT trigger dest anomaly."""
        row = pd.Series({
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 9000.0,
            "amount": 1000.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "nameDest": "C2000",
        })
        score, reason = compute_balance_anomaly_risk(row)
        # old_dest > 0 check prevents this from firing
        assert score == 0.0

    def test_small_discrepancy_within_tolerance(self):
        """Discrepancy within 0.01 tolerance should not trigger."""
        row = pd.Series({
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 9000.005,  # expected 9000.0, diff = 0.005 < 0.01
            "amount": 1000.0,
            "oldbalanceDest": 5000.0,
            "newbalanceDest": 6000.0,
            "nameDest": "C2000",
        })
        score, reason = compute_balance_anomaly_risk(row)
        assert score == 0.0


# ===========================================================================
# Tests for compute_repeat_account_risk
# ===========================================================================


class TestComputeRepeatAccountRisk:
    """Tests for the compute_repeat_account_risk function."""

    def test_no_repeats(self):
        """Unique accounts should all get score 0."""
        df = _make_df([
            {"nameOrig": "C1001", "transaction_id": 0},
            {"nameOrig": "C1002", "transaction_id": 1},
            {"nameOrig": "C1003", "transaction_id": 2},
        ])
        risk = compute_repeat_account_risk(df)
        for idx in df.index:
            assert risk[idx][0] == 0.0

    def test_two_repeats(self):
        """Account with 2 transactions should get score = min(2*5, 15) = 10."""
        df = _make_df([
            {"nameOrig": "C1001", "transaction_id": 0},
            {"nameOrig": "C1001", "transaction_id": 1},
            {"nameOrig": "C1002", "transaction_id": 2},
        ])
        risk = compute_repeat_account_risk(df)
        assert risk[0][0] == 10.0
        assert risk[1][0] == 10.0
        assert risk[2][0] == 0.0

    def test_three_repeats(self):
        """Account with 3 transactions should get score = min(3*5, 15) = 15."""
        df = _make_df([
            {"nameOrig": "C1001", "transaction_id": 0},
            {"nameOrig": "C1001", "transaction_id": 1},
            {"nameOrig": "C1001", "transaction_id": 2},
        ])
        risk = compute_repeat_account_risk(df)
        assert risk[0][0] == 15.0

    def test_capped_at_15(self):
        """Account with >3 transactions should still be capped at 15."""
        df = _make_df([
            {"nameOrig": "C1001", "transaction_id": i} for i in range(5)
        ])
        risk = compute_repeat_account_risk(df)
        for idx in df.index:
            assert risk[idx][0] == 15.0

    def test_reason_contains_account_name(self):
        """Reason should mention the repeated account name."""
        df = _make_df([
            {"nameOrig": "C9999", "transaction_id": 0},
            {"nameOrig": "C9999", "transaction_id": 1},
        ])
        risk = compute_repeat_account_risk(df)
        assert "C9999" in risk[0][1]

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty dict."""
        df = _make_df([])
        risk = compute_repeat_account_risk(df)
        assert risk == {}


# ===========================================================================
# Tests for compute_destination_risk
# ===========================================================================


class TestComputeDestinationRisk:
    """Tests for the compute_destination_risk function."""

    def test_no_high_traffic(self):
        """Destinations with <=2 transactions should get score 0."""
        df = _make_df([
            {"nameDest": "C2001", "transaction_id": 0},
            {"nameDest": "C2002", "transaction_id": 1},
            {"nameDest": "C2003", "transaction_id": 2},
        ])
        risk = compute_destination_risk(df)
        for idx in df.index:
            assert risk[idx][0] == 0.0

    def test_high_traffic_destination(self):
        """Destination with 3 transactions should get score = min(3*3, 10) = 9."""
        df = _make_df([
            {"nameDest": "C2001", "transaction_id": 0},
            {"nameDest": "C2001", "transaction_id": 1},
            {"nameDest": "C2001", "transaction_id": 2},
        ])
        risk = compute_destination_risk(df)
        assert risk[0][0] == 9.0

    def test_capped_at_10(self):
        """Destination with many transactions should cap at 10."""
        df = _make_df([
            {"nameDest": "C2001", "transaction_id": i} for i in range(6)
        ])
        risk = compute_destination_risk(df)
        for idx in df.index:
            assert risk[idx][0] == 10.0

    def test_merchant_excluded(self):
        """Merchant destinations (M prefix) should always score 0."""
        df = _make_df([
            {"nameDest": "M1000", "transaction_id": 0},
            {"nameDest": "M1000", "transaction_id": 1},
            {"nameDest": "M1000", "transaction_id": 2},
            {"nameDest": "M1000", "transaction_id": 3},
        ])
        risk = compute_destination_risk(df)
        for idx in df.index:
            assert risk[idx][0] == 0.0

    def test_exactly_two_transactions(self):
        """Destination with exactly 2 transactions should NOT trigger (>2 required)."""
        df = _make_df([
            {"nameDest": "C2001", "transaction_id": 0},
            {"nameDest": "C2001", "transaction_id": 1},
        ])
        risk = compute_destination_risk(df)
        assert risk[0][0] == 0.0

    def test_reason_contains_dest_name(self):
        """Reason should mention the high-traffic destination."""
        df = _make_df([
            {"nameDest": "C5555", "transaction_id": i} for i in range(4)
        ])
        risk = compute_destination_risk(df)
        assert "C5555" in risk[0][1]

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty dict."""
        df = _make_df([])
        risk = compute_destination_risk(df)
        assert risk == {}


# ===========================================================================
# Tests for compute_cashout_pattern_risk
# ===========================================================================


class TestComputeCashoutPatternRisk:
    """Tests for the compute_cashout_pattern_risk function."""

    def test_no_pattern(self):
        """Normal payments should not trigger layering detection."""
        df = _make_df([
            {"type": "PAYMENT", "amount": 100.0, "nameOrig": "C1", "nameDest": "M1", "transaction_id": 0},
        ])
        risk = compute_cashout_pattern_risk(df)
        assert risk[0][0] == 0.0

    def test_transfer_to_cashout_account(self):
        """TRANSFER to an account that also performs CASH_OUT should score 10."""
        df = _make_df([
            {
                "type": "TRANSFER", "amount": 20000.0,
                "nameOrig": "C1", "nameDest": "C2", "transaction_id": 0,
            },
            {
                "type": "CASH_OUT", "amount": 20000.0,
                "nameOrig": "C2", "nameDest": "C3", "transaction_id": 1,
            },
        ])
        risk = compute_cashout_pattern_risk(df)
        # The TRANSFER to C2 should be flagged because C2 is in cashout_dests
        assert risk[0][0] == 10.0
        assert "layering" in risk[0][1].lower()

    def test_large_cashout_over_50k(self):
        """CASH_OUT > 50k should score 10."""
        df = _make_df([
            {
                "type": "CASH_OUT", "amount": 60000.0,
                "nameOrig": "C1", "nameDest": "C2", "transaction_id": 0,
            },
        ])
        risk = compute_cashout_pattern_risk(df)
        assert risk[0][0] == 10.0
        assert "cash-out" in risk[0][1].lower()

    def test_cashout_exactly_50k(self):
        """CASH_OUT of exactly 50000 should NOT trigger (>50000 required)."""
        df = _make_df([
            {
                "type": "CASH_OUT", "amount": 50000.0,
                "nameOrig": "C1", "nameDest": "C2", "transaction_id": 0,
            },
        ])
        risk = compute_cashout_pattern_risk(df)
        assert risk[0][0] == 0.0

    def test_small_transfer_not_flagged(self):
        """TRANSFER with amount <= 10000 should not trigger even if dest does CASH_OUT."""
        df = _make_df([
            {
                "type": "TRANSFER", "amount": 10000.0,
                "nameOrig": "C1", "nameDest": "C2", "transaction_id": 0,
            },
            {
                "type": "CASH_OUT", "amount": 5000.0,
                "nameOrig": "C2", "nameDest": "C3", "transaction_id": 1,
            },
        ])
        risk = compute_cashout_pattern_risk(df)
        # amount 10000 is NOT > 10000
        assert risk[0][0] == 0.0

    def test_transfer_to_non_cashout_account(self):
        """TRANSFER to an account that does NOT do CASH_OUT should score 0."""
        df = _make_df([
            {
                "type": "TRANSFER", "amount": 20000.0,
                "nameOrig": "C1", "nameDest": "C5", "transaction_id": 0,
            },
            {
                "type": "PAYMENT", "amount": 100.0,
                "nameOrig": "C5", "nameDest": "M1", "transaction_id": 1,
            },
        ])
        risk = compute_cashout_pattern_risk(df)
        assert risk[0][0] == 0.0

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty dict."""
        df = _make_df([])
        risk = compute_cashout_pattern_risk(df)
        assert risk == {}


# ===========================================================================
# Tests for assign_risk_level
# ===========================================================================


class TestAssignRiskLevel:
    """Tests for the assign_risk_level function."""

    @pytest.mark.parametrize(
        "score, expected_level",
        [
            (0, "LOW"),
            (10, "LOW"),
            (39, "LOW"),
            (39.99, "LOW"),
            (40, "MEDIUM"),
            (50, "MEDIUM"),
            (70, "MEDIUM"),
            (70.01, "HIGH"),
            (71, "HIGH"),
            (85, "HIGH"),
            (100, "HIGH"),
        ],
    )
    def test_risk_level_boundaries(self, score, expected_level):
        """Risk level should follow threshold rules: >70=HIGH, >=40=MEDIUM, else LOW."""
        assert assign_risk_level(score) == expected_level

    def test_zero_score(self):
        """Zero score should return LOW."""
        assert assign_risk_level(0) == "LOW"

    def test_exactly_70(self):
        """Score of exactly 70 is >=40 but NOT >70, so MEDIUM."""
        assert assign_risk_level(70) == "MEDIUM"

    def test_exactly_40(self):
        """Score of exactly 40 is >=40, so MEDIUM."""
        assert assign_risk_level(40) == "MEDIUM"


# ===========================================================================
# Tests for generate_risk_report
# ===========================================================================


class TestGenerateRiskReport:
    """Tests for the generate_risk_report function."""

    def test_basic_report_structure(self):
        """Report should contain the expected columns."""
        df = _make_df([
            {"transaction_id": 0, "type": "PAYMENT", "amount": 100.0},
        ])
        report = generate_risk_report(df)
        assert set(report.columns) == {
            "transaction_id", "risk_score", "risk_level", "explanation",
        }

    def test_low_risk_transaction(self):
        """A benign PAYMENT with small amount should be LOW risk."""
        df = _make_df([{
            "transaction_id": 0,
            "type": "PAYMENT",
            "amount": 100.0,
            "nameOrig": "C1",
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 9900.0,
            "nameDest": "M1",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
        }])
        report = generate_risk_report(df)
        assert report.iloc[0]["risk_level"] == "LOW"
        assert report.iloc[0]["risk_score"] == 0.0

    def test_high_risk_cash_out(self):
        """Large CASH_OUT with fraud flags should produce HIGH risk."""
        df = _make_df([{
            "transaction_id": 0,
            "type": "CASH_OUT",
            "amount": 600000.0,
            "nameOrig": "C1",
            "oldbalanceOrg": 600000.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C2",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 600000.0,
            "isFraud": 1,
            "isFlaggedFraud": 1,
        }])
        report = generate_risk_report(df)
        row = report.iloc[0]
        assert row["risk_level"] == "HIGH"
        # amount(25) + type(20) + drained(10) + discrepancy(5) + fraud(15) + flagged(10) = 85
        assert row["risk_score"] > 70

    def test_fraud_boost_is_fraud(self):
        """isFraud=1 should add 15 points."""
        df_fraud = _make_df([{
            "transaction_id": 0, "type": "PAYMENT", "amount": 100.0,
            "isFraud": 1, "isFlaggedFraud": 0,
            "nameOrig": "C1", "oldbalanceOrg": 10000.0, "newbalanceOrig": 9900.0,
            "nameDest": "M1", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        }])
        df_no_fraud = _make_df([{
            "transaction_id": 0, "type": "PAYMENT", "amount": 100.0,
            "isFraud": 0, "isFlaggedFraud": 0,
            "nameOrig": "C1", "oldbalanceOrg": 10000.0, "newbalanceOrig": 9900.0,
            "nameDest": "M1", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        }])
        report_fraud = generate_risk_report(df_fraud)
        report_no = generate_risk_report(df_no_fraud)
        assert report_fraud.iloc[0]["risk_score"] - report_no.iloc[0]["risk_score"] == 15.0

    def test_fraud_boost_is_flagged_fraud(self):
        """isFlaggedFraud=1 should add 10 points."""
        df_flagged = _make_df([{
            "transaction_id": 0, "type": "PAYMENT", "amount": 100.0,
            "isFraud": 0, "isFlaggedFraud": 1,
            "nameOrig": "C1", "oldbalanceOrg": 10000.0, "newbalanceOrig": 9900.0,
            "nameDest": "M1", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        }])
        df_no = _make_df([{
            "transaction_id": 0, "type": "PAYMENT", "amount": 100.0,
            "isFraud": 0, "isFlaggedFraud": 0,
            "nameOrig": "C1", "oldbalanceOrg": 10000.0, "newbalanceOrig": 9900.0,
            "nameDest": "M1", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        }])
        report_flagged = generate_risk_report(df_flagged)
        report_no = generate_risk_report(df_no)
        assert report_flagged.iloc[0]["risk_score"] - report_no.iloc[0]["risk_score"] == 10.0

    def test_score_capped_at_100(self):
        """Even with all signals maxed, score should not exceed 100."""
        # Build a transaction designed to hit every risk signal heavily
        df = _make_df([
            {
                "transaction_id": 0,
                "type": "CASH_OUT",
                "amount": 600000.0,
                "nameOrig": "C1",
                "oldbalanceOrg": 600000.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C2",
                "oldbalanceDest": 5000.0,
                "newbalanceDest": 0.0,
                "isFraud": 1,
                "isFlaggedFraud": 1,
            },
            # Duplicate to trigger repeat account risk
            {
                "transaction_id": 1,
                "type": "CASH_OUT",
                "amount": 600000.0,
                "nameOrig": "C1",
                "oldbalanceOrg": 600000.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C2",
                "oldbalanceDest": 5000.0,
                "newbalanceDest": 0.0,
                "isFraud": 1,
                "isFlaggedFraud": 1,
            },
            {
                "transaction_id": 2,
                "type": "CASH_OUT",
                "amount": 600000.0,
                "nameOrig": "C1",
                "oldbalanceOrg": 600000.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C2",
                "oldbalanceDest": 5000.0,
                "newbalanceDest": 0.0,
                "isFraud": 1,
                "isFlaggedFraud": 1,
            },
        ])
        report = generate_risk_report(df)
        for _, row in report.iterrows():
            assert row["risk_score"] <= 100.0

    def test_no_risk_signals_explanation(self):
        """Transaction with no risk signals should explain as such."""
        df = _make_df([{
            "transaction_id": 0,
            "type": "PAYMENT",
            "amount": 100.0,
            "nameOrig": "C1",
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 9900.0,
            "nameDest": "M1",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 100.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
        }])
        report = generate_risk_report(df)
        assert "no significant risk" in report.iloc[0]["explanation"].lower()

    def test_multiple_transactions(self):
        """Report should have one row per transaction."""
        df = _make_df([
            {"transaction_id": i, "nameOrig": f"C{i}"} for i in range(10)
        ])
        report = generate_risk_report(df)
        assert len(report) == 10

    def test_medium_risk_range(self):
        """A TRANSFER with moderate amount should be MEDIUM risk."""
        df = _make_df([{
            "transaction_id": 0,
            "type": "TRANSFER",
            "amount": 15000.0,
            "nameOrig": "C1",
            "oldbalanceOrg": 50000.0,
            "newbalanceOrig": 35000.0,
            "nameDest": "C2",
            "oldbalanceDest": 10000.0,
            "newbalanceDest": 25000.0,
            "isFraud": 1,
            "isFlaggedFraud": 0,
        }])
        report = generate_risk_report(df)
        row = report.iloc[0]
        # type(15) + amount(10) + fraud(15) = 40 => MEDIUM
        assert row["risk_level"] == "MEDIUM"
        assert row["risk_score"] >= 40

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty report."""
        df = _make_df([])
        report = generate_risk_report(df)
        assert len(report) == 0
