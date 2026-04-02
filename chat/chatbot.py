"""
Rule-based chatbot for financial transaction analysis.

Parses user queries using keyword matching and pattern recognition,
then dispatches to the appropriate analyzer functions.
"""

import re
from typing import Optional

import pandas as pd

from chat.analyzer import (
    filter_by_amount_range,
    filter_by_type,
    filter_fraud_flagged,
    filter_high_risk,
    filter_low_risk,
    filter_medium_risk,
    get_account_summary,
    get_balance_anomalies,
    get_cashout_transfers,
    get_large_transactions,
    get_summary_statistics,
    get_top_risky_transactions,
    get_transaction_detail,
)


def format_currency(value: float) -> str:
    """Format a number as currency."""
    return "${:,.2f}".format(value)


def format_dataframe_response(
    df: pd.DataFrame, title: str, max_rows: int = 20
) -> tuple[str, Optional[pd.DataFrame]]:
    """Format a DataFrame subset into a readable chat response."""
    if df.empty:
        return "No transactions found matching your criteria.", None

    display_cols = [
        "transaction_id",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "risk_score",
        "risk_level",
    ]
    available_cols = [c for c in display_cols if c in df.columns]
    subset = df[available_cols].head(max_rows)

    total = len(df)
    shown = len(subset)

    lines = ["**{}**".format(title)]
    lines.append("Found **{}** transaction(s).".format(total))
    if total > max_rows:
        lines.append("_Showing top {} results._".format(max_rows))
    lines.append("")

    return "\n".join(lines), subset


def format_transaction_detail(txn: dict) -> str:
    """Format a single transaction's details into a readable response."""
    lines = [
        "**Transaction #{} Details**".format(int(txn.get("transaction_id", "N/A"))),
        "",
        "| Field | Value |",
        "|---|---|",
        "| **Type** | {} |".format(txn.get("type", "N/A")),
        "| **Amount** | {} |".format(format_currency(txn.get("amount", 0))),
        "| **Origin Account** | {} |".format(txn.get("nameOrig", "N/A")),
        "| **Destination Account** | {} |".format(txn.get("nameDest", "N/A")),
        "| **Origin Balance (Before)** | {} |".format(
            format_currency(txn.get("oldbalanceOrg", 0))
        ),
        "| **Origin Balance (After)** | {} |".format(
            format_currency(txn.get("newbalanceOrig", 0))
        ),
        "| **Dest Balance (Before)** | {} |".format(
            format_currency(txn.get("oldbalanceDest", 0))
        ),
        "| **Dest Balance (After)** | {} |".format(
            format_currency(txn.get("newbalanceDest", 0))
        ),
        "| **Is Fraud** | {} |".format(
            "Yes" if txn.get("isFraud", 0) == 1 else "No"
        ),
        "| **Is Flagged Fraud** | {} |".format(
            "Yes" if txn.get("isFlaggedFraud", 0) == 1 else "No"
        ),
        "| **Risk Score** | {}/100 |".format(txn.get("risk_score", "N/A")),
        "| **Risk Level** | {} |".format(txn.get("risk_level", "N/A")),
        "",
        "**Risk Explanation:**",
        txn.get("explanation", "No explanation available."),
    ]
    return "\n".join(lines)


def format_summary(stats: dict) -> str:
    """Format summary statistics into a readable response."""
    risk_dist = stats.get("risk_distribution", {})
    type_dist = stats.get("type_distribution", {})

    risk_lines = []
    for level in ["HIGH", "MEDIUM", "LOW"]:
        count = risk_dist.get(level, 0)
        risk_lines.append("  - **{}**: {} transactions".format(level, count))

    type_lines = []
    for t, count in type_dist.items():
        type_lines.append("  - **{}**: {} transactions".format(t, count))

    lines = [
        "**Dataset Summary**",
        "",
        "**General Statistics:**",
        "- Total Transactions: **{}**".format(stats["total_transactions"]),
        "- Total Amount: **{}**".format(format_currency(stats["total_amount"])),
        "- Average Amount: **{}**".format(format_currency(stats["avg_amount"])),
        "- Min Amount: **{}**".format(format_currency(stats["min_amount"])),
        "- Max Amount: **{}**".format(format_currency(stats["max_amount"])),
        "",
        "**Risk Distribution:**",
        *risk_lines,
        "",
        "**Transaction Types:**",
        *type_lines,
        "",
        "**Fraud Indicators:**",
        "- Confirmed Fraud: **{}**".format(stats["fraud_count"]),
        "- Flagged Fraud: **{}**".format(stats["flagged_fraud_count"]),
        "",
        "**Risk Scores:**",
        "- Average Risk Score: **{:.2f}**".format(stats["avg_risk_score"]),
        "- Highest Risk Score: **{:.2f}**".format(stats["max_risk_score"]),
        "- Lowest Risk Score: **{:.2f}**".format(stats["min_risk_score"]),
    ]
    return "\n".join(lines)


def format_account_summary(info: dict) -> str:
    """Format account summary into a readable response."""
    if not info.get("found", False):
        return "Account **{}** not found in the dataset.".format(
            info.get("account_id", "unknown")
        )

    risk_levels = info.get("risk_levels", {})
    txn_types = info.get("transaction_types", {})

    lines = [
        "**Account Summary: {}**".format(info["account_id"]),
        "",
        "- Total Transactions: **{}**".format(info["total_transactions"]),
        "- Total Amount: **{}**".format(format_currency(info["total_amount"])),
        "- Average Risk Score: **{:.2f}**".format(info["avg_risk_score"]),
        "- Max Risk Score: **{:.2f}**".format(info["max_risk_score"]),
        "",
        "**Risk Level Breakdown:**",
    ]
    for level in ["HIGH", "MEDIUM", "LOW"]:
        count = risk_levels.get(level, 0)
        if count > 0:
            lines.append("  - {}: {}".format(level, count))

    lines.append("")
    lines.append("**Transaction Types:**")
    for t, count in txn_types.items():
        lines.append("  - {}: {}".format(t, count))

    return "\n".join(lines)


def extract_number(text: str) -> Optional[float]:
    """Extract the first number from a text string."""
    match = re.search(r"[\d,]+\.?\d*", text.replace(",", ""))
    if match:
        return float(match.group().replace(",", ""))
    return None


def extract_transaction_id(text: str) -> Optional[int]:
    """Extract transaction ID from text."""
    patterns = [
        r"transaction\s*#?\s*(\d+)",
        r"txn\s*#?\s*(\d+)",
        r"id\s*#?\s*(\d+)",
        r"#\s*(\d+)",
        r"number\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_account_id(text: str) -> Optional[str]:
    """Extract account ID (C or M prefix followed by digits) from text."""
    match = re.search(r"[CM]\d+", text)
    if match:
        return match.group()
    return None


def extract_amount_range(text: str) -> tuple[float, float]:
    """Extract amount range from text like 'between 1000 and 5000'."""
    between_match = re.search(
        r"between\s+([\d,.]+)\s+and\s+([\d,.]+)", text, re.IGNORECASE
    )
    if between_match:
        low = float(between_match.group(1).replace(",", ""))
        high = float(between_match.group(2).replace(",", ""))
        return low, high

    above_match = re.search(
        r"(?:above|over|greater than|more than|>)\s*([\d,.]+)", text, re.IGNORECASE
    )
    if above_match:
        return float(above_match.group(1).replace(",", "")), float("inf")

    below_match = re.search(
        r"(?:below|under|less than|lower than|<)\s*([\d,.]+)", text, re.IGNORECASE
    )
    if below_match:
        return 0, float(below_match.group(1).replace(",", ""))

    return 0, float("inf")


def extract_transaction_type(text: str) -> Optional[str]:
    """Extract transaction type from text."""
    text_upper = text.upper()
    for t in ["CASH_OUT", "CASH OUT", "CASHOUT", "TRANSFER", "DEBIT", "PAYMENT", "CASH_IN"]:
        if t in text_upper:
            normalized = t.replace(" ", "_").replace("CASHOUT", "CASH_OUT")
            return normalized
    return None


def process_query(query: str, enriched: pd.DataFrame) -> tuple[str, Optional[pd.DataFrame]]:
    """
    Process a user query and return a response.

    Returns a tuple of (text_response, optional_dataframe).
    """
    query_lower = query.lower().strip()

    # --- Greeting ---
    if query_lower in ("hi", "hello", "hey", "help", "what can you do", "what can you do?"):
        return _help_response(), None

    # --- Summary / Overview ---
    if any(
        kw in query_lower
        for kw in ["summary", "overview", "statistics", "stats", "describe", "dataset info"]
    ):
        stats = get_summary_statistics(enriched)
        return format_summary(stats), None

    # --- Explain specific transaction ---
    if "explain" in query_lower or "why" in query_lower or "detail" in query_lower:
        txn_id = extract_transaction_id(query)
        if txn_id is not None:
            txn = get_transaction_detail(enriched, txn_id)
            if txn:
                return format_transaction_detail(txn), None
            return "Transaction #{} not found in the dataset.".format(txn_id), None

        # If no specific ID, but asking about suspicious transactions generally
        if "suspicious" in query_lower or "fraud" in query_lower:
            high_risk = filter_high_risk(enriched)
            if high_risk.empty:
                return "No high-risk transactions found.", None
            text, df = format_dataframe_response(
                high_risk, "Suspicious/High-Risk Transactions"
            )
            explanation_lines = [text, "", "**Explanations:**", ""]
            for _, row in high_risk.head(5).iterrows():
                explanation_lines.append(
                    "- **Transaction #{}** (Score: {}): {}".format(
                        int(row["transaction_id"]),
                        row["risk_score"],
                        row["explanation"],
                    )
                )
            return "\n".join(explanation_lines), df

    # --- High risk transactions ---
    if "high" in query_lower and ("risk" in query_lower or "risky" in query_lower):
        high_risk = filter_high_risk(enriched)
        text, df = format_dataframe_response(high_risk, "High-Risk Transactions")
        return text, df

    # --- Medium risk transactions ---
    if "medium" in query_lower and ("risk" in query_lower or "risky" in query_lower):
        medium_risk = filter_medium_risk(enriched)
        text, df = format_dataframe_response(medium_risk, "Medium-Risk Transactions")
        return text, df

    # --- Low risk transactions ---
    if "low" in query_lower and ("risk" in query_lower or "risky" in query_lower):
        low_risk = filter_low_risk(enriched)
        text, df = format_dataframe_response(low_risk, "Low-Risk Transactions")
        return text, df

    # --- Top risky transactions ---
    if "top" in query_lower and ("risk" in query_lower or "risky" in query_lower):
        n = extract_number(query)
        n = int(n) if n and n <= 100 else 10
        top = get_top_risky_transactions(enriched, n)
        text, df = format_dataframe_response(
            top, "Top {} Riskiest Transactions".format(n)
        )
        return text, df

    # --- Fraud flagged ---
    if "fraud" in query_lower or "flagged" in query_lower:
        flagged = filter_fraud_flagged(enriched)
        text, df = format_dataframe_response(
            flagged, "Fraud-Flagged Transactions"
        )
        return text, df

    # --- Suspicious transactions ---
    if "suspicious" in query_lower:
        # Suspicious = medium or high risk
        suspicious = enriched[enriched["risk_level"].isin(["MEDIUM", "HIGH"])].copy()
        text, df = format_dataframe_response(
            suspicious, "Suspicious Transactions (Medium & High Risk)"
        )
        return text, df

    # --- Transaction by ID ---
    txn_id = extract_transaction_id(query)
    if txn_id is not None:
        txn = get_transaction_detail(enriched, txn_id)
        if txn:
            return format_transaction_detail(txn), None
        return "Transaction #{} not found. Valid IDs are 0 to {}.".format(
            txn_id, len(enriched) - 1
        ), None

    # --- Filter by transaction type ---
    txn_type = extract_transaction_type(query)
    if txn_type and ("show" in query_lower or "list" in query_lower or "filter" in query_lower or "find" in query_lower or txn_type.lower() in query_lower):
        filtered = filter_by_type(enriched, txn_type)
        text, df = format_dataframe_response(
            filtered, "{} Transactions".format(txn_type)
        )
        return text, df

    # --- Amount-based queries ---
    if any(
        kw in query_lower
        for kw in [
            "amount",
            "above",
            "below",
            "over",
            "under",
            "greater",
            "less",
            "between",
            "large",
            "biggest",
            "largest",
            "smallest",
        ]
    ):
        if "largest" in query_lower or "biggest" in query_lower:
            n = extract_number(query)
            n = int(n) if n and n <= 100 else 10
            top = enriched.nlargest(n, "amount")
            text, df = format_dataframe_response(
                top, "Top {} Largest Transactions by Amount".format(n)
            )
            return text, df

        if "smallest" in query_lower:
            n = extract_number(query)
            n = int(n) if n and n <= 100 else 10
            bottom = enriched.nsmallest(n, "amount")
            text, df = format_dataframe_response(
                bottom, "Top {} Smallest Transactions by Amount".format(n)
            )
            return text, df

        min_amt, max_amt = extract_amount_range(query)
        if min_amt > 0 or max_amt < float("inf"):
            filtered = filter_by_amount_range(enriched, min_amt, max_amt)
            if max_amt == float("inf"):
                title = "Transactions Above {}".format(format_currency(min_amt))
            elif min_amt == 0:
                title = "Transactions Below {}".format(format_currency(max_amt))
            else:
                title = "Transactions Between {} and {}".format(
                    format_currency(min_amt), format_currency(max_amt)
                )
            text, df = format_dataframe_response(filtered, title)
            return text, df

        # Large transactions default
        large = get_large_transactions(enriched)
        text, df = format_dataframe_response(
            large, "Large Transactions (>$100,000)"
        )
        return text, df

    # --- Balance anomalies ---
    if any(
        kw in query_lower
        for kw in ["balance", "anomal", "drained", "discrepancy", "zero balance"]
    ):
        anomalies = get_balance_anomalies(enriched)
        text, df = format_dataframe_response(anomalies, "Balance Anomaly Transactions")
        return text, df

    # --- Account lookup ---
    account_id = extract_account_id(query)
    if account_id:
        info = get_account_summary(enriched, account_id)
        return format_account_summary(info), None

    # --- Cash-out and transfer patterns ---
    if any(
        kw in query_lower
        for kw in ["cash out", "cashout", "cash_out", "transfer", "layering"]
    ):
        co = get_cashout_transfers(enriched)
        text, df = format_dataframe_response(
            co, "CASH_OUT & TRANSFER Transactions"
        )
        return text, df

    # --- Transaction type specific (fallback) ---
    if txn_type:
        filtered = filter_by_type(enriched, txn_type)
        text, df = format_dataframe_response(
            filtered, "{} Transactions".format(txn_type)
        )
        return text, df

    # --- All transactions ---
    if "all" in query_lower and "transaction" in query_lower:
        text, df = format_dataframe_response(enriched, "All Transactions")
        return text, df

    # --- Default: show help ---
    return _unknown_query_response(query), None


def _help_response() -> str:
    """Return the help message."""
    return """**Welcome to the Financial Transaction Analyzer!**

I can help you analyze financial transactions and identify potential fraud. Here are some things you can ask me:

**Risk Analysis:**
- "Show high-risk transactions"
- "Show medium-risk transactions"
- "Show low-risk transactions"
- "Show top 10 riskiest transactions"
- "Show suspicious transactions"

**Transaction Details:**
- "Explain transaction #3"
- "Why is transaction #4 suspicious?"
- "Show details for transaction #16"

**Filtering:**
- "Show all CASH_OUT transactions"
- "Show TRANSFER transactions"
- "Show transactions above 100000"
- "Show transactions between 5000 and 50000"
- "Show the 5 largest transactions"

**Fraud Detection:**
- "Show fraud-flagged transactions"
- "Show balance anomalies"
- "Explain why a transaction is suspicious"

**Statistics:**
- "Show summary statistics"
- "Give me a dataset overview"

**Account Analysis:**
- "Show info for account C1231006815"
- "What transactions involve C840083671?"

Type any question to get started!"""


def _unknown_query_response(query: str) -> str:
    """Return a response for unrecognized queries."""
    return """I'm not sure how to interpret "**{}**".

Here are some example questions you can try:
- "Show high-risk transactions"
- "Explain transaction #3"
- "Show summary statistics"
- "Show transactions above 100000"
- "Show fraud-flagged transactions"
- "Show balance anomalies"
- "Show CASH_OUT transactions"

Type **help** to see all supported queries.""".format(query)
