"""
Comprehensive pytest tests for sequence_anomaly_detection.py

Covers all 11 public functions with multiple test cases including
normal inputs, edge cases, boundary conditions, and empty DataFrames.
"""

import os
import tempfile

import pandas as pd
import pytest

# Import the module under test
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from fraudRiskScoring import sequenceAnomalyDetection as sad


# ---------------------------------------------------------------------------
# Helper: build a minimal DataFrame with all required columns
# ---------------------------------------------------------------------------
COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud",
]


def _make_df(rows):
    """Create a DataFrame from a list of dicts, filling missing columns with defaults."""
    defaults = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 100.0,
        "nameOrig": "C1000",
        "oldbalanceOrg": 50000.0,
        "newbalanceOrig": 49900.0,
        "nameDest": "C2000",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 100.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }
    full_rows = []
    for r in rows:
        row = {**defaults, **r}
        full_rows.append(row)
    df = pd.DataFrame(full_rows, columns=COLUMNS)
    return df


def _empty_df():
    """Return an empty DataFrame with the expected columns."""
    return pd.DataFrame(columns=COLUMNS)


# ===================================================================
# 1. load_and_prepare
# ===================================================================
class TestLoadAndPrepare:
    def test_basic_load(self, tmp_path):
        """Load a small CSV and verify transaction_id assignment and sorting."""
        csv_path = tmp_path / "test.csv"
        df_src = pd.DataFrame([
            {"step": 2, "type": "TRANSFER", "amount": 500, "nameOrig": "C200",
             "oldbalanceOrg": 1000, "newbalanceOrig": 500, "nameDest": "C300",
             "oldbalanceDest": 0, "newbalanceDest": 500, "isFraud": 0, "isFlaggedFraud": 0},
            {"step": 1, "type": "PAYMENT", "amount": 100, "nameOrig": "C100",
             "oldbalanceOrg": 5000, "newbalanceOrig": 4900, "nameDest": "M100",
             "oldbalanceDest": 0, "newbalanceDest": 100, "isFraud": 0, "isFlaggedFraud": 0},
            {"step": 1, "type": "CASH_OUT", "amount": 200, "nameOrig": "C200",
             "oldbalanceOrg": 1200, "newbalanceOrig": 1000, "nameDest": "C400",
             "oldbalanceDest": 0, "newbalanceDest": 200, "isFraud": 0, "isFlaggedFraud": 0},
        ])
        df_src.to_csv(csv_path, index=False)

        result = sad.load_and_prepare(str(csv_path))

        # transaction_id should be 1-indexed based on original row order
        assert "transaction_id" in result.columns
        assert list(result["transaction_id"].sort_values()) == [1, 2, 3]

        # Sorted by nameOrig then step
        assert list(result["nameOrig"]) == ["C100", "C200", "C200"]
        # C200 rows sorted by step: step=1 first, step=2 second
        c200_steps = result[result["nameOrig"] == "C200"]["step"].tolist()
        assert c200_steps == [1, 2]

    def test_single_row(self, tmp_path):
        """Load a CSV with a single row."""
        csv_path = tmp_path / "single.csv"
        df_src = pd.DataFrame([{
            "step": 1, "type": "PAYMENT", "amount": 50, "nameOrig": "C1",
            "oldbalanceOrg": 100, "newbalanceOrig": 50, "nameDest": "M1",
            "oldbalanceDest": 0, "newbalanceDest": 50, "isFraud": 0, "isFlaggedFraud": 0,
        }])
        df_src.to_csv(csv_path, index=False)
        result = sad.load_and_prepare(str(csv_path))
        assert len(result) == 1
        assert result["transaction_id"].iloc[0] == 1

    def test_transaction_id_is_one_indexed(self, tmp_path):
        """Verify IDs start at 1, not 0."""
        csv_path = tmp_path / "ids.csv"
        rows = []
        for i in range(5):
            rows.append({
                "step": i, "type": "PAYMENT", "amount": 10 * (i + 1),
                "nameOrig": "C1", "oldbalanceOrg": 1000, "newbalanceOrig": 900,
                "nameDest": "M1", "oldbalanceDest": 0, "newbalanceDest": 10,
                "isFraud": 0, "isFlaggedFraud": 0,
            })
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        result = sad.load_and_prepare(str(csv_path))
        assert result["transaction_id"].min() == 1
        assert result["transaction_id"].max() == 5


# ===================================================================
# 2. build_dest_index
# ===================================================================
class TestBuildDestIndex:
    def test_basic_grouping(self):
        df = _make_df([
            {"nameDest": "C2000"},
            {"nameDest": "C2000"},
            {"nameDest": "C3000"},
        ])
        idx = sad.build_dest_index(df)
        assert len(idx["C2000"]) == 2
        assert len(idx["C3000"]) == 1

    def test_empty_df(self):
        idx = sad.build_dest_index(_empty_df())
        assert idx == {}

    def test_single_dest(self):
        df = _make_df([{"nameDest": "M100"}])
        idx = sad.build_dest_index(df)
        assert "M100" in idx
        assert len(idx["M100"]) == 1

    def test_row_dict_content(self):
        """Verify the list items are dicts with expected keys."""
        df = _make_df([{"nameDest": "C5000", "amount": 999.99}])
        idx = sad.build_dest_index(df)
        entry = idx["C5000"][0]
        assert isinstance(entry, dict)
        assert entry["amount"] == 999.99
        assert entry["nameDest"] == "C5000"


# ===================================================================
# 3. build_transfer_amounts
# ===================================================================
class TestBuildTransferAmounts:
    def test_only_transfers_included(self):
        df = _make_df([
            {"type": "TRANSFER", "step": 1, "amount": 5000.0},
            {"type": "CASH_OUT", "step": 1, "amount": 5000.0},
            {"type": "PAYMENT", "step": 1, "amount": 5000.0},
        ])
        ta = sad.build_transfer_amounts(df)
        # Only TRANSFER rows should be indexed
        assert (1, 5000.0) in ta
        assert len(ta[(1, 5000.0)]) == 1
        assert ta[(1, 5000.0)][0]["type"] == "TRANSFER"

    def test_multiple_transfers_same_key(self):
        df = _make_df([
            {"type": "TRANSFER", "step": 2, "amount": 10000.0, "nameOrig": "C100"},
            {"type": "TRANSFER", "step": 2, "amount": 10000.0, "nameOrig": "C200"},
        ])
        ta = sad.build_transfer_amounts(df)
        assert len(ta[(2, 10000.0)]) == 2

    def test_empty_df(self):
        ta = sad.build_transfer_amounts(_empty_df())
        assert ta == {}

    def test_amount_rounding(self):
        """Amounts should be rounded to 2 decimal places for key."""
        df = _make_df([
            {"type": "TRANSFER", "step": 1, "amount": 1234.5678},
        ])
        ta = sad.build_transfer_amounts(df)
        assert (1, 1234.57) in ta

    def test_no_transfers(self):
        df = _make_df([
            {"type": "CASH_OUT", "amount": 5000},
            {"type": "PAYMENT", "amount": 3000},
        ])
        ta = sad.build_transfer_amounts(df)
        assert ta == {}


# ===================================================================
# 4. detect_repeated_high_value_to_dest
# ===================================================================
class TestDetectRepeatedHighValueToDest:
    def test_two_high_value_to_same_dest(self):
        """Two high-value txns to same non-merchant dest -> findings."""
        df = _make_df([
            {"nameOrig": "C100", "nameDest": "C999", "amount": 20000, "type": "TRANSFER"},
            {"nameOrig": "C200", "nameDest": "C999", "amount": 30000, "type": "TRANSFER"},
        ])
        dest_index = sad.build_dest_index(df)
        findings = sad.detect_repeated_high_value_to_dest(df, dest_index)
        assert len(findings) == 2  # one per sender
        # Score = 30 + 10*(2-1) = 40
        assert all(f["score"] == 40 for f in findings)

    def test_three_high_value_to_same_dest(self):
        """Three high-value txns -> score = min(30+10*2, 60) = 50."""
        df = _make_df([
            {"nameOrig": "C100", "nameDest": "C999", "amount": 15000, "type": "TRANSFER"},
            {"nameOrig": "C200", "nameDest": "C999", "amount": 25000, "type": "TRANSFER"},
            {"nameOrig": "C300", "nameDest": "C999", "amount": 35000, "type": "TRANSFER"},
        ])
        dest_index = sad.build_dest_index(df)
        findings = sad.detect_repeated_high_value_to_dest(df, dest_index)
        assert len(findings) == 3
        assert all(f["score"] == 50 for f in findings)

    def test_score_capped_at_60(self):
        """Five high-value txns -> score = min(30+10*4, 60) = 60."""
        rows = [
            {"nameOrig": f"C{i}", "nameDest": "C999", "amount": 50000, "type": "TRANSFER"}
            for i in range(5)
        ]
        df = _make_df(rows)
        dest_index = sad.build_dest_index(df)
        findings = sad.detect_repeated_high_value_to_dest(df, dest_index)
        assert all(f["score"] == 60 for f in findings)

    def test_merchant_excluded(self):
        """Destinations starting with 'M' should be skipped."""
        df = _make_df([
            {"nameOrig": "C100", "nameDest": "M999", "amount": 20000, "type": "TRANSFER"},
            {"nameOrig": "C200", "nameDest": "M999", "amount": 30000, "type": "TRANSFER"},
        ])
        dest_index = sad.build_dest_index(df)
        findings = sad.detect_repeated_high_value_to_dest(df, dest_index)
        assert findings == []

    def test_below_threshold_ignored(self):
        """Transactions <= 10000 are not counted as high-value."""
        df = _make_df([
            {"nameOrig": "C100", "nameDest": "C999", "amount": 10000, "type": "TRANSFER"},
            {"nameOrig": "C200", "nameDest": "C999", "amount": 9999, "type": "TRANSFER"},
        ])
        dest_index = sad.build_dest_index(df)
        findings = sad.detect_repeated_high_value_to_dest(df, dest_index)
        assert findings == []

    def test_only_one_high_value(self):
        """Need >= 2 high-value txns to trigger; one is not enough."""
        df = _make_df([
            {"nameOrig": "C100", "nameDest": "C999", "amount": 50000, "type": "TRANSFER"},
            {"nameOrig": "C200", "nameDest": "C999", "amount": 5000, "type": "PAYMENT"},
        ])
        dest_index = sad.build_dest_index(df)
        findings = sad.detect_repeated_high_value_to_dest(df, dest_index)
        assert findings == []

    def test_empty_df(self):
        df = _empty_df()
        dest_index = sad.build_dest_index(df)
        findings = sad.detect_repeated_high_value_to_dest(df, dest_index)
        assert findings == []

    def test_finding_fields(self):
        """Check that findings have the expected keys."""
        df = _make_df([
            {"nameOrig": "C100", "nameDest": "C999", "amount": 20000, "type": "TRANSFER"},
            {"nameOrig": "C200", "nameDest": "C999", "amount": 30000, "type": "CASH_OUT"},
        ])
        dest_index = sad.build_dest_index(df)
        findings = sad.detect_repeated_high_value_to_dest(df, dest_index)
        for f in findings:
            assert "customer_id" in f
            assert "pattern" in f
            assert "score" in f
            assert "explanation" in f
            assert "REPEATED_HIGH_VALUE_TO_C999" in f["pattern"]


# ===================================================================
# 5. detect_transfer_cashout_pairs
# ===================================================================
class TestDetectTransferCashoutPairs:
    def test_matching_pair(self):
        """A TRANSFER and CASH_OUT at the same step and amount -> 2 findings, score 45 each."""
        df = _make_df([
            {"type": "TRANSFER", "step": 3, "amount": 50000.0,
             "nameOrig": "C100", "nameDest": "C200"},
            {"type": "CASH_OUT", "step": 3, "amount": 50000.0,
             "nameOrig": "C300", "nameDest": "C400"},
        ])
        ta = sad.build_transfer_amounts(df)
        findings = sad.detect_transfer_cashout_pairs(df, ta)
        assert len(findings) == 2
        assert findings[0]["score"] == 45
        assert findings[1]["score"] == 45
        # One for the transfer originator, one for the cashout originator
        customers = {f["customer_id"] for f in findings}
        assert customers == {"C100", "C300"}

    def test_no_match_different_step(self):
        """Same amount but different step -> no match."""
        df = _make_df([
            {"type": "TRANSFER", "step": 1, "amount": 50000.0, "nameOrig": "C100"},
            {"type": "CASH_OUT", "step": 2, "amount": 50000.0, "nameOrig": "C200"},
        ])
        ta = sad.build_transfer_amounts(df)
        findings = sad.detect_transfer_cashout_pairs(df, ta)
        assert findings == []

    def test_no_match_different_amount(self):
        """Same step but different amount -> no match."""
        df = _make_df([
            {"type": "TRANSFER", "step": 1, "amount": 50000.0, "nameOrig": "C100"},
            {"type": "CASH_OUT", "step": 1, "amount": 50001.0, "nameOrig": "C200"},
        ])
        ta = sad.build_transfer_amounts(df)
        findings = sad.detect_transfer_cashout_pairs(df, ta)
        assert findings == []

    def test_duplicate_pair_deduplicated(self):
        """Same (transfer_orig, cashout_orig) pair seen twice -> only one pair produced."""
        df = _make_df([
            {"type": "TRANSFER", "step": 1, "amount": 25000.0,
             "nameOrig": "C100", "nameDest": "C500"},
            {"type": "TRANSFER", "step": 1, "amount": 25000.0,
             "nameOrig": "C100", "nameDest": "C600"},
            {"type": "CASH_OUT", "step": 1, "amount": 25000.0,
             "nameOrig": "C200", "nameDest": "C700"},
        ])
        ta = sad.build_transfer_amounts(df)
        findings = sad.detect_transfer_cashout_pairs(df, ta)
        # The pair_key (C100, C200) is seen twice, but only one pair kept
        pair_customers = [(f["customer_id"]) for f in findings]
        assert pair_customers.count("C100") == 1
        assert pair_customers.count("C200") == 1

    def test_empty_df(self):
        ta = sad.build_transfer_amounts(_empty_df())
        findings = sad.detect_transfer_cashout_pairs(_empty_df(), ta)
        assert findings == []

    def test_no_cashouts(self):
        df = _make_df([
            {"type": "TRANSFER", "step": 1, "amount": 50000.0, "nameOrig": "C100"},
        ])
        ta = sad.build_transfer_amounts(df)
        findings = sad.detect_transfer_cashout_pairs(df, ta)
        assert findings == []

    def test_no_transfers(self):
        df = _make_df([
            {"type": "CASH_OUT", "step": 1, "amount": 50000.0, "nameOrig": "C100"},
        ])
        ta = sad.build_transfer_amounts(df)
        findings = sad.detect_transfer_cashout_pairs(df, ta)
        assert findings == []

    def test_explanation_contains_layering(self):
        """Explanation should mention 'layering'."""
        df = _make_df([
            {"type": "TRANSFER", "step": 5, "amount": 10000.0,
             "nameOrig": "C100", "nameDest": "C200"},
            {"type": "CASH_OUT", "step": 5, "amount": 10000.0,
             "nameOrig": "C300", "nameDest": "C400"},
        ])
        ta = sad.build_transfer_amounts(df)
        findings = sad.detect_transfer_cashout_pairs(df, ta)
        for f in findings:
            assert "layering" in f["explanation"].lower()


# ===================================================================
# 6. detect_sudden_amount_spike
# ===================================================================
class TestDetectSuddenAmountSpike:
    def test_spike_detected(self):
        """Amount/balance >= 0.95 AND amount > 10000 -> finding."""
        df = _make_df([{
            "nameOrig": "C100", "type": "TRANSFER",
            "amount": 19000, "oldbalanceOrg": 20000,
            "newbalanceOrig": 1000,
        }])
        findings = sad.detect_sudden_amount_spike(df)
        assert len(findings) == 1
        assert findings[0]["customer_id"] == "C100"
        # ratio = 19000/20000 = 0.95, score = min(25 + int(0.95*10), 50) = min(25+9, 50) = 34
        assert findings[0]["score"] == 34

    def test_ratio_exactly_095(self):
        """Boundary: ratio == 0.95 should trigger."""
        df = _make_df([{
            "amount": 19000, "oldbalanceOrg": 20000,
            "newbalanceOrig": 1000,
        }])
        findings = sad.detect_sudden_amount_spike(df)
        assert len(findings) == 1

    def test_ratio_below_095(self):
        """Ratio < 0.95 -> no finding."""
        df = _make_df([{
            "amount": 18000, "oldbalanceOrg": 20000,
            "newbalanceOrig": 2000,
        }])
        findings = sad.detect_sudden_amount_spike(df)
        assert findings == []

    def test_amount_at_threshold(self):
        """Amount == 10000 (not > 10000) -> no finding."""
        df = _make_df([{
            "amount": 10000, "oldbalanceOrg": 10000,
            "newbalanceOrig": 0,
        }])
        findings = sad.detect_sudden_amount_spike(df)
        assert findings == []

    def test_amount_just_above_threshold(self):
        """Amount = 10001, ratio = 10001/10001 = 1.0 -> finding."""
        df = _make_df([{
            "amount": 10001, "oldbalanceOrg": 10001,
            "newbalanceOrig": 0,
        }])
        findings = sad.detect_sudden_amount_spike(df)
        assert len(findings) == 1

    def test_zero_balance_skipped(self):
        """balance <= 0 is skipped to avoid division by zero."""
        df = _make_df([{
            "amount": 50000, "oldbalanceOrg": 0,
            "newbalanceOrig": 0,
        }])
        findings = sad.detect_sudden_amount_spike(df)
        assert findings == []

    def test_negative_balance_skipped(self):
        df = _make_df([{
            "amount": 50000, "oldbalanceOrg": -100,
            "newbalanceOrig": 0,
        }])
        findings = sad.detect_sudden_amount_spike(df)
        assert findings == []

    def test_score_capped_at_50(self):
        """Very high ratio should cap score at 50."""
        # ratio = 100000/100001 ~= 1.0, score = min(25 + 10, 50) = 35
        # Need ratio high enough: e.g., amount=500000, balance=100000 => ratio=5.0
        # score = min(25 + int(5.0*10), 50) = min(75, 50) = 50
        df = _make_df([{
            "amount": 500000, "oldbalanceOrg": 100000,
            "newbalanceOrig": 0,
        }])
        findings = sad.detect_sudden_amount_spike(df)
        assert len(findings) == 1
        assert findings[0]["score"] == 50

    def test_empty_df(self):
        findings = sad.detect_sudden_amount_spike(_empty_df())
        assert findings == []

    def test_high_amount_low_ratio(self):
        """Large amount but ratio < 0.95 -> no finding."""
        df = _make_df([{
            "amount": 50000, "oldbalanceOrg": 1000000,
            "newbalanceOrig": 950000,
        }])
        findings = sad.detect_sudden_amount_spike(df)
        assert findings == []


# ===================================================================
# 7. detect_high_value_cashout
# ===================================================================
class TestDetectHighValueCashout:
    def test_cashout_above_200k(self):
        """CASH_OUT > 200k -> score 50."""
        df = _make_df([{"type": "CASH_OUT", "amount": 250000, "nameOrig": "C100"}])
        findings = sad.detect_high_value_cashout(df)
        assert len(findings) == 1
        assert findings[0]["score"] == 50

    def test_cashout_above_100k(self):
        """CASH_OUT > 100k but <= 200k -> score 40."""
        df = _make_df([{"type": "CASH_OUT", "amount": 150000, "nameOrig": "C100"}])
        findings = sad.detect_high_value_cashout(df)
        assert len(findings) == 1
        assert findings[0]["score"] == 40

    def test_cashout_above_threshold(self):
        """CASH_OUT > 10000 but <= 100k -> score 30."""
        df = _make_df([{"type": "CASH_OUT", "amount": 50000, "nameOrig": "C100"}])
        findings = sad.detect_high_value_cashout(df)
        assert len(findings) == 1
        assert findings[0]["score"] == 30

    def test_cashout_at_threshold(self):
        """CASH_OUT == 10000 (not > 10000) -> no finding."""
        df = _make_df([{"type": "CASH_OUT", "amount": 10000, "nameOrig": "C100"}])
        findings = sad.detect_high_value_cashout(df)
        assert findings == []

    def test_cashout_below_threshold(self):
        df = _make_df([{"type": "CASH_OUT", "amount": 5000, "nameOrig": "C100"}])
        findings = sad.detect_high_value_cashout(df)
        assert findings == []

    def test_non_cashout_excluded(self):
        """Only CASH_OUT type should produce findings."""
        df = _make_df([
            {"type": "TRANSFER", "amount": 250000, "nameOrig": "C100"},
            {"type": "PAYMENT", "amount": 250000, "nameOrig": "C200"},
        ])
        findings = sad.detect_high_value_cashout(df)
        assert findings == []

    def test_boundary_100k(self):
        """Amount == 100000 (not > 100k) -> score 30."""
        df = _make_df([{"type": "CASH_OUT", "amount": 100000, "nameOrig": "C100"}])
        findings = sad.detect_high_value_cashout(df)
        assert len(findings) == 1
        assert findings[0]["score"] == 30

    def test_boundary_200k(self):
        """Amount == 200000 (not > 200k) -> score 40."""
        df = _make_df([{"type": "CASH_OUT", "amount": 200000, "nameOrig": "C100"}])
        findings = sad.detect_high_value_cashout(df)
        assert len(findings) == 1
        assert findings[0]["score"] == 40

    def test_boundary_100001(self):
        """Amount == 100001 (> 100k) -> score 40."""
        df = _make_df([{"type": "CASH_OUT", "amount": 100001, "nameOrig": "C100"}])
        findings = sad.detect_high_value_cashout(df)
        assert findings[0]["score"] == 40

    def test_boundary_200001(self):
        """Amount == 200001 (> 200k) -> score 50."""
        df = _make_df([{"type": "CASH_OUT", "amount": 200001, "nameOrig": "C100"}])
        findings = sad.detect_high_value_cashout(df)
        assert findings[0]["score"] == 50

    def test_empty_df(self):
        findings = sad.detect_high_value_cashout(_empty_df())
        assert findings == []

    def test_multiple_cashouts(self):
        df = _make_df([
            {"type": "CASH_OUT", "amount": 50000, "nameOrig": "C100"},
            {"type": "CASH_OUT", "amount": 150000, "nameOrig": "C200"},
            {"type": "CASH_OUT", "amount": 300000, "nameOrig": "C300"},
        ])
        findings = sad.detect_high_value_cashout(df)
        assert len(findings) == 3
        scores = {f["customer_id"]: f["score"] for f in findings}
        assert scores["C100"] == 30
        assert scores["C200"] == 40
        assert scores["C300"] == 50


# ===================================================================
# 8. detect_high_value_transfer
# ===================================================================
class TestDetectHighValueTransfer:
    def test_transfer_above_500k(self):
        """TRANSFER > 500k -> score 40."""
        df = _make_df([{"type": "TRANSFER", "amount": 600000, "nameOrig": "C100", "nameDest": "C200"}])
        findings = sad.detect_high_value_transfer(df)
        assert len(findings) == 1
        assert findings[0]["score"] == 40

    def test_transfer_above_100k(self):
        """TRANSFER > 100k but <= 500k -> score 30."""
        df = _make_df([{"type": "TRANSFER", "amount": 200000, "nameOrig": "C100", "nameDest": "C200"}])
        findings = sad.detect_high_value_transfer(df)
        assert len(findings) == 1
        assert findings[0]["score"] == 30

    def test_transfer_above_threshold(self):
        """TRANSFER > 10000 but <= 100k -> score 20."""
        df = _make_df([{"type": "TRANSFER", "amount": 50000, "nameOrig": "C100", "nameDest": "C200"}])
        findings = sad.detect_high_value_transfer(df)
        assert len(findings) == 1
        assert findings[0]["score"] == 20

    def test_transfer_at_threshold(self):
        """TRANSFER == 10000 -> no finding."""
        df = _make_df([{"type": "TRANSFER", "amount": 10000, "nameOrig": "C100", "nameDest": "C200"}])
        findings = sad.detect_high_value_transfer(df)
        assert findings == []

    def test_non_transfer_excluded(self):
        df = _make_df([
            {"type": "CASH_OUT", "amount": 600000, "nameOrig": "C100"},
            {"type": "PAYMENT", "amount": 600000, "nameOrig": "C200"},
        ])
        findings = sad.detect_high_value_transfer(df)
        assert findings == []

    def test_boundary_100k(self):
        """Amount == 100000 -> score 20 (not > 100k)."""
        df = _make_df([{"type": "TRANSFER", "amount": 100000, "nameOrig": "C100", "nameDest": "C200"}])
        findings = sad.detect_high_value_transfer(df)
        assert findings[0]["score"] == 20

    def test_boundary_500k(self):
        """Amount == 500000 -> score 30 (not > 500k)."""
        df = _make_df([{"type": "TRANSFER", "amount": 500000, "nameOrig": "C100", "nameDest": "C200"}])
        findings = sad.detect_high_value_transfer(df)
        assert findings[0]["score"] == 30

    def test_boundary_100001(self):
        """Amount == 100001 -> score 30."""
        df = _make_df([{"type": "TRANSFER", "amount": 100001, "nameOrig": "C100", "nameDest": "C200"}])
        findings = sad.detect_high_value_transfer(df)
        assert findings[0]["score"] == 30

    def test_boundary_500001(self):
        """Amount == 500001 -> score 40."""
        df = _make_df([{"type": "TRANSFER", "amount": 500001, "nameOrig": "C100", "nameDest": "C200"}])
        findings = sad.detect_high_value_transfer(df)
        assert findings[0]["score"] == 40

    def test_empty_df(self):
        findings = sad.detect_high_value_transfer(_empty_df())
        assert findings == []

    def test_finding_contains_dest(self):
        """Pattern should include the destination account."""
        df = _make_df([{"type": "TRANSFER", "amount": 50000, "nameOrig": "C100", "nameDest": "C999"}])
        findings = sad.detect_high_value_transfer(df)
        assert "C999" in findings[0]["pattern"]


# ===================================================================
# 9. detect_confirmed_fraud_sequences
# ===================================================================
class TestDetectConfirmedFraudSequences:
    def test_is_fraud_only(self):
        """isFraud=1 -> score 15."""
        df = _make_df([{"isFraud": 1, "isFlaggedFraud": 0, "nameOrig": "C100",
                         "type": "TRANSFER", "amount": 5000}])
        findings = sad.detect_confirmed_fraud_sequences(df)
        assert len(findings) == 1
        assert findings[0]["score"] == 15
        assert "confirmed fraud" in findings[0]["explanation"]

    def test_is_flagged_fraud_only(self):
        """isFlaggedFraud=1 -> score 10."""
        df = _make_df([{"isFraud": 0, "isFlaggedFraud": 1, "nameOrig": "C100",
                         "type": "CASH_OUT", "amount": 8000}])
        findings = sad.detect_confirmed_fraud_sequences(df)
        assert len(findings) == 1
        assert findings[0]["score"] == 10
        assert "flagged fraud" in findings[0]["explanation"]

    def test_both_fraud_flags(self):
        """isFraud=1 AND isFlaggedFraud=1 -> score 25."""
        df = _make_df([{"isFraud": 1, "isFlaggedFraud": 1, "nameOrig": "C100",
                         "type": "TRANSFER", "amount": 50000}])
        findings = sad.detect_confirmed_fraud_sequences(df)
        assert len(findings) == 1
        assert findings[0]["score"] == 25
        assert "confirmed fraud" in findings[0]["explanation"]
        assert "flagged fraud" in findings[0]["explanation"]

    def test_no_fraud(self):
        """Neither flag -> no findings."""
        df = _make_df([{"isFraud": 0, "isFlaggedFraud": 0}])
        findings = sad.detect_confirmed_fraud_sequences(df)
        assert findings == []

    def test_multiple_fraud_rows(self):
        df = _make_df([
            {"isFraud": 1, "isFlaggedFraud": 0, "nameOrig": "C100"},
            {"isFraud": 0, "isFlaggedFraud": 1, "nameOrig": "C200"},
            {"isFraud": 1, "isFlaggedFraud": 1, "nameOrig": "C300"},
            {"isFraud": 0, "isFlaggedFraud": 0, "nameOrig": "C400"},
        ])
        findings = sad.detect_confirmed_fraud_sequences(df)
        assert len(findings) == 3
        scores = {f["customer_id"]: f["score"] for f in findings}
        assert scores["C100"] == 15
        assert scores["C200"] == 10
        assert scores["C300"] == 25

    def test_empty_df(self):
        findings = sad.detect_confirmed_fraud_sequences(_empty_df())
        assert findings == []


# ===================================================================
# 10. assign_anomaly_level
# ===================================================================
class TestAssignAnomalyLevel:
    def test_low(self):
        assert sad.assign_anomaly_level(0) == "LOW"
        assert sad.assign_anomaly_level(10) == "LOW"
        assert sad.assign_anomaly_level(39) == "LOW"

    def test_boundary_39_is_low(self):
        assert sad.assign_anomaly_level(39) == "LOW"

    def test_boundary_40_is_medium(self):
        assert sad.assign_anomaly_level(40) == "MEDIUM"

    def test_medium(self):
        assert sad.assign_anomaly_level(40) == "MEDIUM"
        assert sad.assign_anomaly_level(55) == "MEDIUM"
        assert sad.assign_anomaly_level(70) == "MEDIUM"

    def test_boundary_70_is_medium(self):
        assert sad.assign_anomaly_level(70) == "MEDIUM"

    def test_boundary_71_is_high(self):
        assert sad.assign_anomaly_level(71) == "HIGH"

    def test_high(self):
        assert sad.assign_anomaly_level(71) == "HIGH"
        assert sad.assign_anomaly_level(85) == "HIGH"
        assert sad.assign_anomaly_level(100) == "HIGH"

    def test_zero(self):
        assert sad.assign_anomaly_level(0) == "LOW"

    def test_negative(self):
        """Negative scores should still be LOW."""
        assert sad.assign_anomaly_level(-5) == "LOW"

    def test_very_high(self):
        assert sad.assign_anomaly_level(1000) == "HIGH"

    def test_float_boundaries(self):
        assert sad.assign_anomaly_level(39.9) == "LOW"
        assert sad.assign_anomaly_level(40.0) == "MEDIUM"
        assert sad.assign_anomaly_level(70.0) == "MEDIUM"
        assert sad.assign_anomaly_level(70.1) == "HIGH"


# ===================================================================
# 11. generate_report
# ===================================================================
class TestGenerateReport:
    def test_empty_df(self):
        """Empty DataFrame -> empty report."""
        report = sad.generate_report(_empty_df())
        assert report.empty

    def test_no_anomalies(self):
        """All transactions below thresholds -> empty report."""
        df = _make_df([
            {"type": "PAYMENT", "amount": 100, "nameOrig": "C100", "nameDest": "M100",
             "oldbalanceOrg": 50000, "isFraud": 0, "isFlaggedFraud": 0},
        ])
        report = sad.generate_report(df)
        assert report.empty

    def test_report_columns(self):
        """Verify the report has the expected columns."""
        df = _make_df([
            {"type": "CASH_OUT", "amount": 50000, "nameOrig": "C100", "nameDest": "C200"},
        ])
        report = sad.generate_report(df)
        assert set(report.columns) == {
            "customer_id", "sequence_pattern", "anomaly_score",
            "anomaly_level", "explanation",
        }

    def test_score_capped_at_100(self):
        """Multiple high signals for one customer -> score capped at 100."""
        # Create a customer with many high-scoring patterns:
        # - TRANSFER > 500k (40 pts)
        # - Confirmed fraud (15 pts)
        # - Spike (up to 50 pts)
        # Total uncapped > 100
        df = _make_df([
            {"type": "TRANSFER", "amount": 600000, "nameOrig": "C100",
             "nameDest": "C200", "oldbalanceOrg": 600000, "newbalanceOrig": 0,
             "isFraud": 1, "isFlaggedFraud": 1},
        ])
        report = sad.generate_report(df)
        assert not report.empty
        row = report[report["customer_id"] == "C100"]
        assert len(row) == 1
        assert row.iloc[0]["anomaly_score"] <= 100

    def test_sorted_by_score_descending(self):
        """Report should be sorted by anomaly_score descending."""
        df = _make_df([
            {"type": "CASH_OUT", "amount": 50000, "nameOrig": "C100", "nameDest": "C200"},
            {"type": "CASH_OUT", "amount": 300000, "nameOrig": "C200", "nameDest": "C300"},
        ])
        report = sad.generate_report(df)
        if len(report) >= 2:
            scores = report["anomaly_score"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_anomaly_level_assigned(self):
        """Each row should have a valid anomaly_level."""
        df = _make_df([
            {"type": "CASH_OUT", "amount": 50000, "nameOrig": "C100", "nameDest": "C200"},
        ])
        report = sad.generate_report(df)
        for _, row in report.iterrows():
            assert row["anomaly_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_aggregation_per_customer(self):
        """Multiple findings for the same customer are aggregated into one row."""
        # Use a high oldbalanceOrg so the spike detector does NOT fire
        df = _make_df([
            {"type": "CASH_OUT", "amount": 50000, "nameOrig": "C100", "nameDest": "C200",
             "oldbalanceOrg": 5000000, "newbalanceOrig": 4950000,
             "isFraud": 1, "isFlaggedFraud": 0},
        ])
        report = sad.generate_report(df)
        c100_rows = report[report["customer_id"] == "C100"]
        assert len(c100_rows) == 1
        # Score should combine high_value_cashout (30) + confirmed fraud (15) = 45
        assert c100_rows.iloc[0]["anomaly_score"] == 45

    def test_multiple_customers(self):
        """Different customers produce separate rows."""
        df = _make_df([
            {"type": "CASH_OUT", "amount": 50000, "nameOrig": "C100", "nameDest": "C200"},
            {"type": "CASH_OUT", "amount": 150000, "nameOrig": "C200", "nameDest": "C300"},
        ])
        report = sad.generate_report(df)
        customers = set(report["customer_id"].tolist())
        assert "C100" in customers
        assert "C200" in customers

    def test_cap_exactly_at_100(self):
        """Score should be exactly 100, not above."""
        # Build a customer with extremely high combined score:
        # Transfer > 500k (40) + spike with very high ratio (50) + isFraud (15) + isFlaggedFraud (10) = 115
        # Need ratio high enough for spike score to be 50: ratio >= 2.5 -> int(2.5*10)=25, 25+25=50
        df = _make_df([
            {"type": "TRANSFER", "amount": 600000, "nameOrig": "C100",
             "nameDest": "C200", "oldbalanceOrg": 100000, "newbalanceOrig": 0,
             "isFraud": 1, "isFlaggedFraud": 1},
        ])
        report = sad.generate_report(df)
        c100 = report[report["customer_id"] == "C100"]
        assert len(c100) == 1
        assert c100.iloc[0]["anomaly_score"] == 100

    def test_patterns_combined(self):
        """Multiple patterns should be joined in the sequence_pattern field."""
        df = _make_df([
            {"type": "CASH_OUT", "amount": 50000, "nameOrig": "C100", "nameDest": "C200",
             "isFraud": 1, "isFlaggedFraud": 0},
        ])
        report = sad.generate_report(df)
        row = report[report["customer_id"] == "C100"].iloc[0]
        # Should contain both patterns separated by " | "
        assert "HIGH_VALUE_CASHOUT" in row["sequence_pattern"]
        assert "CONFIRMED_FRAUD" in row["sequence_pattern"]

    def test_repeated_high_value_integrated(self):
        """Integration: repeated high-value to same dest triggers in generate_report."""
        df = _make_df([
            {"type": "TRANSFER", "amount": 20000, "nameOrig": "C100",
             "nameDest": "C999", "oldbalanceOrg": 100000, "newbalanceOrig": 80000},
            {"type": "TRANSFER", "amount": 30000, "nameOrig": "C200",
             "nameDest": "C999", "oldbalanceOrg": 100000, "newbalanceOrig": 70000},
        ])
        report = sad.generate_report(df)
        # Both C100 and C200 should be in the report
        customers = set(report["customer_id"].tolist())
        assert "C100" in customers
        assert "C200" in customers

    def test_transfer_cashout_pair_integrated(self):
        """Integration: TRANSFER+CASH_OUT pair at same step and amount triggers."""
        df = _make_df([
            {"type": "TRANSFER", "step": 3, "amount": 50000,
             "nameOrig": "C100", "nameDest": "C200",
             "oldbalanceOrg": 100000, "newbalanceOrig": 50000},
            {"type": "CASH_OUT", "step": 3, "amount": 50000,
             "nameOrig": "C300", "nameDest": "C400",
             "oldbalanceOrg": 100000, "newbalanceOrig": 50000},
        ])
        report = sad.generate_report(df)
        customers = set(report["customer_id"].tolist())
        assert "C100" in customers
        assert "C300" in customers
