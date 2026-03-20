"""
Detect Anomalous Transaction Sequences

Analyzes transaction data to identify suspicious sequential patterns by:
1. Grouping transactions by customer (nameOrig).
2. Sorting transactions by time (step).
3. Analyzing sequences of transaction types and amounts - both within
   a single customer's history and across linked accounts (money-flow
   chains where a destination account later originates a transaction).
4. Detecting:
   - Repeated high-value transactions (per customer or per destination)
   - TRANSFER followed by CASH_OUT (layering / money-laundering)
   - Sudden increases in transaction amounts relative to account balance
5. Assigning an anomaly level (LOW / MEDIUM / HIGH).
6. Producing a CSV report with customer_id, sequence_pattern,
   anomaly_level, and explanation.

Uses Python and pandas.
"""

import os

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HIGH_VALUE_THRESHOLD = 10_000
BALANCE_DRAIN_THRESHOLD = 0.95

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INPUT_FILE = os.path.join(DATA_DIR, "Example1.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "sequence_anomaly_report.csv")


def load_and_prepare(filepath):
    df = pd.read_csv(filepath)
    df["transaction_id"] = range(1, len(df) + 1)
    df = df.sort_values(by=["nameOrig", "step"]).reset_index(drop=True)
    return df


def build_dest_index(df):
    dest_index = {}
    for _, row in df.iterrows():
        dest = row["nameDest"]
        if dest not in dest_index:
            dest_index[dest] = []
        dest_index[dest].append(row.to_dict())
    return dest_index


def build_transfer_amounts(df):
    transfers = {}
    for _, row in df[df["type"] == "TRANSFER"].iterrows():
        key = (row["step"], round(row["amount"], 2))
        if key not in transfers:
            transfers[key] = []
        transfers[key].append(row.to_dict())
    return transfers


def detect_repeated_high_value_to_dest(df, dest_index):
    findings = []
    for dest, txns in dest_index.items():
        if dest.startswith("M"):
            continue
        high_txns = [t for t in txns if t["amount"] > HIGH_VALUE_THRESHOLD]
        if len(high_txns) >= 2:
            senders = [t["nameOrig"] for t in high_txns]
            amounts = [t["amount"] for t in high_txns]
            types = [t["type"] for t in high_txns]
            pattern = " -> ".join(
                "{0}({1:,.2f})".format(tp, amt) for tp, amt in zip(types, amounts)
            )
            score = min(30 + 10 * (len(high_txns) - 1), 60)
            for sender in senders:
                findings.append({
                    "customer_id": sender,
                    "pattern": "REPEATED_HIGH_VALUE_TO_{0}: {1}".format(dest, pattern),
                    "score": score,
                    "explanation": (
                        "Destination {0} received {1} high-value transactions "
                        "(>{2:,}) from accounts: {3}; amounts: {4}"
                    ).format(
                        dest, len(high_txns), HIGH_VALUE_THRESHOLD,
                        ", ".join(senders),
                        ", ".join("{0:,.2f}".format(a) for a in amounts)
                    ),
                })
    return findings


def detect_transfer_cashout_pairs(df, transfer_amounts):
    findings = []
    seen_pairs = set()
    cashouts = df[df["type"] == "CASH_OUT"]

    for _, co_row in cashouts.iterrows():
        key = (co_row["step"], round(co_row["amount"], 2))
        if key not in transfer_amounts:
            continue
        for tf in transfer_amounts[key]:
            pair_key = (tf["nameOrig"], co_row["nameOrig"])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            pattern = (
                "TRANSFER_THEN_CASHOUT: TRANSFER({0:,.2f}) by {1} "
                "-> CASH_OUT({2:,.2f}) by {3}"
            ).format(tf["amount"], tf["nameOrig"], co_row["amount"], co_row["nameOrig"])
            explanation_tf = (
                "Transfer of {0:,.2f} from {1} to {2} at step {3} "
                "matched by cash-out of {4:,.2f} from {5}; "
                "potential layering pattern"
            ).format(
                tf["amount"], tf["nameOrig"], tf["nameDest"],
                int(tf["step"]), co_row["amount"], co_row["nameOrig"]
            )
            explanation_co = (
                "Cash-out of {0:,.2f} from {1} at step {2} "
                "matches transfer of {3:,.2f} from {4}; "
                "potential layering pattern"
            ).format(
                co_row["amount"], co_row["nameOrig"],
                int(co_row["step"]), tf["amount"], tf["nameOrig"]
            )
            findings.append({
                "customer_id": tf["nameOrig"],
                "pattern": pattern,
                "score": 45,
                "explanation": explanation_tf,
            })
            findings.append({
                "customer_id": co_row["nameOrig"],
                "pattern": pattern,
                "score": 45,
                "explanation": explanation_co,
            })
    return findings


def detect_sudden_amount_spike(df):
    findings = []
    for _, row in df.iterrows():
        balance = row["oldbalanceOrg"]
        amount = row["amount"]
        if balance <= 0:
            continue
        ratio = amount / balance
        if ratio >= BALANCE_DRAIN_THRESHOLD and amount > HIGH_VALUE_THRESHOLD:
            score = min(25 + int(ratio * 10), 50)
            findings.append({
                "customer_id": row["nameOrig"],
                "pattern": "SUDDEN_INCREASE: {0}({1:,.2f}) vs balance({2:,.2f})".format(
                    row["type"], amount, balance
                ),
                "score": score,
                "explanation": (
                    "Transaction amount {0:,.2f} consumes {1:.1f}% of account "
                    "balance {2:,.2f}; large balance drain above {3:,} "
                    "suggests possible fraud"
                ).format(amount, ratio * 100, balance, HIGH_VALUE_THRESHOLD),
            })
    return findings


def detect_high_value_cashout(df):
    findings = []
    cashouts = df[(df["type"] == "CASH_OUT") & (df["amount"] > HIGH_VALUE_THRESHOLD)]
    for _, row in cashouts.iterrows():
        score = 30
        if row["amount"] > 100_000:
            score = 40
        if row["amount"] > 200_000:
            score = 50
        findings.append({
            "customer_id": row["nameOrig"],
            "pattern": "HIGH_VALUE_CASHOUT: CASH_OUT({0:,.2f})".format(row["amount"]),
            "score": score,
            "explanation": (
                "High-value cash-out of {0:,.2f} from {1}; "
                "CASH_OUT is a high-risk transaction type"
            ).format(row["amount"], row["nameOrig"]),
        })
    return findings


def detect_high_value_transfer(df):
    findings = []
    transfers = df[(df["type"] == "TRANSFER") & (df["amount"] > HIGH_VALUE_THRESHOLD)]
    for _, row in transfers.iterrows():
        score = 20
        if row["amount"] > 100_000:
            score = 30
        if row["amount"] > 500_000:
            score = 40
        findings.append({
            "customer_id": row["nameOrig"],
            "pattern": "HIGH_VALUE_TRANSFER: TRANSFER({0:,.2f}) to {1}".format(
                row["amount"], row["nameDest"]
            ),
            "score": score,
            "explanation": (
                "High-value transfer of {0:,.2f} from {1} to {2}; "
                "TRANSFER is a higher-risk transaction type"
            ).format(row["amount"], row["nameOrig"], row["nameDest"]),
        })
    return findings


def detect_confirmed_fraud_sequences(df):
    findings = []
    fraud_rows = df[(df["isFraud"] == 1) | (df["isFlaggedFraud"] == 1)]
    for _, row in fraud_rows.iterrows():
        score = 0
        reasons = []
        if row["isFraud"] == 1:
            score += 15
            reasons.append("confirmed fraud (isFraud=1)")
        if row["isFlaggedFraud"] == 1:
            score += 10
            reasons.append("flagged fraud (isFlaggedFraud=1)")
        findings.append({
            "customer_id": row["nameOrig"],
            "pattern": "CONFIRMED_FRAUD: {0}({1:,.2f})".format(row["type"], row["amount"]),
            "score": score,
            "explanation": (
                "Transaction {0} of {1:,.2f} from {2}: {3}"
            ).format(row["type"], row["amount"], row["nameOrig"], "; ".join(reasons)),
        })
    return findings


def assign_anomaly_level(score):
    if score < 40:
        return "LOW"
    if score <= 70:
        return "MEDIUM"
    return "HIGH"


def generate_report(df):
    dest_index = build_dest_index(df)
    transfer_amounts = build_transfer_amounts(df)

    all_findings = []
    all_findings.extend(detect_repeated_high_value_to_dest(df, dest_index))
    all_findings.extend(detect_transfer_cashout_pairs(df, transfer_amounts))
    all_findings.extend(detect_sudden_amount_spike(df))
    all_findings.extend(detect_high_value_cashout(df))
    all_findings.extend(detect_high_value_transfer(df))
    all_findings.extend(detect_confirmed_fraud_sequences(df))

    customer_findings = {}
    for f in all_findings:
        cid = f["customer_id"]
        if cid not in customer_findings:
            customer_findings[cid] = []
        customer_findings[cid].append(f)

    records = []
    for customer_id, findings in customer_findings.items():
        total_score = min(sum(f["score"] for f in findings), 100)
        combined_patterns = " | ".join(
            dict.fromkeys(f["pattern"] for f in findings)
        )
        combined_explanations = "; ".join(
            dict.fromkeys(f["explanation"] for f in findings)
        )
        anomaly_level = assign_anomaly_level(total_score)

        records.append({
            "customer_id": customer_id,
            "sequence_pattern": combined_patterns,
            "anomaly_score": total_score,
            "anomaly_level": anomaly_level,
            "explanation": combined_explanations,
        })

    report = pd.DataFrame(records)
    if not report.empty:
        report = report.sort_values(
            "anomaly_score", ascending=False
        ).reset_index(drop=True)
    return report


def main():
    print("Loading dataset...")
    df = load_and_prepare(INPUT_FILE)
    print("  {0} transactions loaded for {1} customers.\n".format(
        len(df), df["nameOrig"].nunique()
    ))

    print("Analyzing transaction sequences...")
    report = generate_report(df)

    print("\n" + "=" * 50)
    print("  Sequence Anomaly Report")
    print("=" * 50)
    print("Anomalous customers found: {0}".format(len(report)))
    if not report.empty:
        for level in ["HIGH", "MEDIUM", "LOW"]:
            count = (report["anomaly_level"] == level).sum()
            if count:
                print("  {0}: {1}".format(level, count))

        print("\nTop anomalies:")
        for _, row in report.head(10).iterrows():
            print("  [{0}] {1}  score={2}".format(
                row["anomaly_level"], row["customer_id"], row["anomaly_score"]
            ))
            print("    pattern: {0}".format(row["sequence_pattern"][:120]))

    report.to_csv(OUTPUT_FILE, index=False)
    print("\nFull report saved to {0}".format(OUTPUT_FILE))


if __name__ == "__main__":
    main()
