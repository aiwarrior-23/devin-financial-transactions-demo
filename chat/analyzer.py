"""
Financial Transaction Analyzer.

Provides rule-based analysis of financial transactions including
risk scoring, anomaly detection, filtering, and statistical summaries.
"""

import os
from typing import Optional

import pandas as pd


DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "Example1.csv",
)

TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]

RISK_LEVEL_THRESHOLDS = {"LOW": 40, "MEDIUM": 70, "HIGH": 100}


def load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load and prepare the transaction dataset."""
    path = filepath or DATA_PATH
    df = pd.read_csv(path)
    df.index.name = "transaction_id"
    df = df.reset_index()
    return df


def compute_amount_risk(amount: float) -> tuple[float, str]:
    """Compute risk from transaction amount."""
    if amount > 500000:
        return 25.0, "Very high transaction amount (>{:,.2f})".format(amount)
    elif amount > 200000:
        return 20.0, "Extremely large transaction amount ({:,.2f})".format(amount)
    elif amount > 100000:
        return 15.0, "Very large transaction amount ({:,.2f})".format(amount)
    elif amount > 10000:
        return 10.0, "High transaction amount ({:,.2f})".format(amount)
    elif amount > 5000:
        return 5.0, "Moderate transaction amount ({:,.2f})".format(amount)
    return 0.0, ""


def compute_type_risk(txn_type: str) -> tuple[float, str]:
    """Compute risk from transaction type."""
    risk_map = {
        "CASH_OUT": (20.0, "High-risk transaction type: CASH_OUT"),
        "TRANSFER": (15.0, "High-risk transaction type: TRANSFER"),
        "DEBIT": (5.0, "Moderate-risk transaction type: DEBIT"),
        "PAYMENT": (0.0, ""),
        "CASH_IN": (0.0, ""),
    }
    return risk_map.get(txn_type, (0.0, ""))


def compute_balance_anomaly_risk(row: pd.Series) -> tuple[float, str]:
    """Detect unusual balance patterns."""
    reasons = []
    score = 0.0

    old_bal = row["oldbalanceOrg"]
    new_bal = row["newbalanceOrig"]
    amount = row["amount"]

    if old_bal > 0 and new_bal == 0:
        score += 10.0
        reasons.append("Origin account fully drained to zero balance")

    expected_new_bal = old_bal - amount
    if abs(expected_new_bal - new_bal) > 0.01 and old_bal > 0:
        score += 5.0
        reasons.append(
            "Balance discrepancy at origin (expected {:.2f}, got {:.2f})".format(
                expected_new_bal, new_bal
            )
        )

    if old_bal == 0 and amount > 0:
        score += 5.0
        reasons.append("Transaction from account with zero initial balance")

    old_dest = row["oldbalanceDest"]
    new_dest = row["newbalanceDest"]
    name_dest = str(row["nameDest"])
    if not name_dest.startswith("M"):
        if old_dest > 0 and new_dest == 0:
            score += 5.0
            reasons.append(
                "Destination balance dropped to zero after receiving funds"
            )

    return min(score, 20.0), "; ".join(reasons)


def compute_repeat_account_risk(
    df: pd.DataFrame,
) -> dict[int, tuple[float, str]]:
    """Identify repeated transactions from the same originating account."""
    account_counts = df.groupby("nameOrig").size()
    repeat_accounts = account_counts[account_counts > 1].index.tolist()

    risk_map = {}
    for idx, row in df.iterrows():
        if row["nameOrig"] in repeat_accounts:
            count = account_counts[row["nameOrig"]]
            score = min(count * 5.0, 15.0)
            risk_map[idx] = (
                score,
                "Account {} has {} transactions (repeated activity)".format(
                    row["nameOrig"], count
                ),
            )
        else:
            risk_map[idx] = (0.0, "")

    return risk_map


def compute_destination_risk(df: pd.DataFrame) -> dict[int, tuple[float, str]]:
    """Assess risk based on destination account patterns."""
    dest_counts = df.groupby("nameDest").size()
    high_traffic_dests = dest_counts[dest_counts > 2].index.tolist()

    risk_map = {}
    for idx, row in df.iterrows():
        name_dest = str(row["nameDest"])
        if name_dest.startswith("M"):
            risk_map[idx] = (0.0, "")
        elif name_dest in high_traffic_dests:
            count = dest_counts[name_dest]
            score = min(count * 3.0, 10.0)
            risk_map[idx] = (
                score,
                "Destination {} received {} transactions (high-traffic)".format(
                    name_dest, count
                ),
            )
        else:
            risk_map[idx] = (0.0, "")

    return risk_map


def compute_cashout_pattern_risk(df: pd.DataFrame) -> dict[int, tuple[float, str]]:
    """Detect large transfer followed by cash-out patterns."""
    risk_map = {idx: (0.0, "") for idx in df.index}

    cashout_dests = set(df[df["type"] == "CASH_OUT"]["nameOrig"].tolist())
    transfer_rows = df[
        (df["type"].isin(["TRANSFER", "CASH_OUT"])) & (df["amount"] > 10000)
    ]

    for idx, row in transfer_rows.iterrows():
        if row["type"] == "TRANSFER" and row["nameDest"] in cashout_dests:
            risk_map[idx] = (
                10.0,
                "Large transfer to account that also performs cash-out (potential layering)",
            )
        elif row["type"] == "CASH_OUT" and row["amount"] > 50000:
            risk_map[idx] = (
                10.0,
                "Large cash-out transaction ({:,.2f}), potential fraud cash-out".format(
                    row["amount"]
                ),
            )

    return risk_map


def assign_risk_level(score: float) -> str:
    """Assign risk category based on score thresholds."""
    if score > 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    return "LOW"


def generate_risk_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a comprehensive transaction-level risk report."""
    repeat_risk = compute_repeat_account_risk(df)
    dest_risk = compute_destination_risk(df)
    cashout_risk = compute_cashout_pattern_risk(df)

    results = []

    for idx, row in df.iterrows():
        explanations = []
        total_score = 0.0

        amt_score, amt_reason = compute_amount_risk(row["amount"])
        total_score += amt_score
        if amt_reason:
            explanations.append(amt_reason)

        type_score, type_reason = compute_type_risk(row["type"])
        total_score += type_score
        if type_reason:
            explanations.append(type_reason)

        bal_score, bal_reason = compute_balance_anomaly_risk(row)
        total_score += bal_score
        if bal_reason:
            explanations.append(bal_reason)

        rep_score, rep_reason = repeat_risk[idx]
        total_score += rep_score
        if rep_reason:
            explanations.append(rep_reason)

        dst_score, dst_reason = dest_risk[idx]
        total_score += dst_score
        if dst_reason:
            explanations.append(dst_reason)

        co_score, co_reason = cashout_risk[idx]
        total_score += co_score
        if co_reason:
            explanations.append(co_reason)

        if row.get("isFraud", 0) == 1:
            total_score += 15.0
            explanations.append("Transaction flagged as fraud in dataset")

        if row.get("isFlaggedFraud", 0) == 1:
            total_score += 10.0
            explanations.append("Transaction flagged by business fraud detection rules")

        final_score = min(round(total_score, 2), 100.0)
        risk_level = assign_risk_level(final_score)
        explanation = (
            "; ".join(explanations)
            if explanations
            else "No significant risk signals detected"
        )

        results.append(
            {
                "transaction_id": row["transaction_id"],
                "risk_score": final_score,
                "risk_level": risk_level,
                "explanation": explanation,
            }
        )

    return pd.DataFrame(results)


def get_enriched_data(df: pd.DataFrame) -> pd.DataFrame:
    """Merge original data with risk report for enriched analysis."""
    report = generate_risk_report(df)
    enriched = df.merge(report, on="transaction_id", how="left")
    return enriched


def get_summary_statistics(enriched: pd.DataFrame) -> dict:
    """Compute summary statistics for the dataset."""
    total = len(enriched)
    risk_dist = enriched["risk_level"].value_counts().to_dict()
    type_dist = enriched["type"].value_counts().to_dict()
    fraud_count = int(enriched["isFraud"].sum())
    flagged_count = int(enriched["isFlaggedFraud"].sum())

    return {
        "total_transactions": total,
        "risk_distribution": risk_dist,
        "type_distribution": type_dist,
        "fraud_count": fraud_count,
        "flagged_fraud_count": flagged_count,
        "total_amount": float(enriched["amount"].sum()),
        "avg_amount": float(enriched["amount"].mean()),
        "max_amount": float(enriched["amount"].max()),
        "min_amount": float(enriched["amount"].min()),
        "avg_risk_score": float(enriched["risk_score"].mean()),
        "max_risk_score": float(enriched["risk_score"].max()),
        "min_risk_score": float(enriched["risk_score"].min()),
    }


def filter_high_risk(enriched: pd.DataFrame) -> pd.DataFrame:
    """Filter transactions with HIGH risk level."""
    return enriched[enriched["risk_level"] == "HIGH"].copy()


def filter_medium_risk(enriched: pd.DataFrame) -> pd.DataFrame:
    """Filter transactions with MEDIUM risk level."""
    return enriched[enriched["risk_level"] == "MEDIUM"].copy()


def filter_low_risk(enriched: pd.DataFrame) -> pd.DataFrame:
    """Filter transactions with LOW risk level."""
    return enriched[enriched["risk_level"] == "LOW"].copy()


def filter_by_type(enriched: pd.DataFrame, txn_type: str) -> pd.DataFrame:
    """Filter transactions by type."""
    return enriched[enriched["type"].str.upper() == txn_type.upper()].copy()


def filter_by_amount_range(
    enriched: pd.DataFrame, min_amt: float = 0, max_amt: float = float("inf")
) -> pd.DataFrame:
    """Filter transactions within an amount range."""
    return enriched[
        (enriched["amount"] >= min_amt) & (enriched["amount"] <= max_amt)
    ].copy()


def filter_fraud_flagged(enriched: pd.DataFrame) -> pd.DataFrame:
    """Filter transactions flagged as fraud."""
    return enriched[
        (enriched["isFraud"] == 1) | (enriched["isFlaggedFraud"] == 1)
    ].copy()


def get_transaction_detail(enriched: pd.DataFrame, txn_id: int) -> Optional[dict]:
    """Get detailed info about a specific transaction."""
    row = enriched[enriched["transaction_id"] == txn_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def get_top_risky_transactions(
    enriched: pd.DataFrame, n: int = 10
) -> pd.DataFrame:
    """Get the top N riskiest transactions."""
    return enriched.nlargest(n, "risk_score").copy()


def get_account_summary(enriched: pd.DataFrame, account_id: str) -> dict:
    """Get summary for a specific account (as originator)."""
    acct_txns = enriched[enriched["nameOrig"] == account_id]
    if acct_txns.empty:
        acct_txns = enriched[enriched["nameDest"] == account_id]
        if acct_txns.empty:
            return {"found": False, "account_id": account_id}

    return {
        "found": True,
        "account_id": account_id,
        "total_transactions": len(acct_txns),
        "total_amount": float(acct_txns["amount"].sum()),
        "avg_risk_score": float(acct_txns["risk_score"].mean()),
        "max_risk_score": float(acct_txns["risk_score"].max()),
        "risk_levels": acct_txns["risk_level"].value_counts().to_dict(),
        "transaction_types": acct_txns["type"].value_counts().to_dict(),
        "transactions": acct_txns,
    }


def get_balance_anomalies(enriched: pd.DataFrame) -> pd.DataFrame:
    """Find transactions with balance anomalies."""
    anomalies = []
    for _, row in enriched.iterrows():
        old_bal = row["oldbalanceOrg"]
        new_bal = row["newbalanceOrig"]
        amount = row["amount"]
        expected = old_bal - amount

        is_drained = old_bal > 0 and new_bal == 0
        has_discrepancy = abs(expected - new_bal) > 0.01 and old_bal > 0
        zero_origin = old_bal == 0 and amount > 0

        if is_drained or has_discrepancy or zero_origin:
            anomalies.append(row["transaction_id"])

    return enriched[enriched["transaction_id"].isin(anomalies)].copy()


def get_large_transactions(
    enriched: pd.DataFrame, threshold: float = 100000
) -> pd.DataFrame:
    """Get transactions above a certain amount threshold."""
    return enriched[enriched["amount"] > threshold].copy()


def get_cashout_transfers(enriched: pd.DataFrame) -> pd.DataFrame:
    """Get CASH_OUT and TRANSFER transactions (higher risk types)."""
    return enriched[enriched["type"].isin(["CASH_OUT", "TRANSFER"])].copy()
