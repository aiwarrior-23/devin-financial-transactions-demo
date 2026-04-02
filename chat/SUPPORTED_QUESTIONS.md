# Supported Questions - Financial Transaction Analyzer Chatbot

This document lists all the questions and query types supported by the Financial Transaction Analyzer chatbot.

---

## Risk Analysis

| Question | Description |
|---|---|
| `Show high-risk transactions` | Displays all transactions with a risk level of HIGH (score > 70) |
| `Show medium-risk transactions` | Displays all transactions with a risk level of MEDIUM (score 40-70) |
| `Show low-risk transactions` | Displays all transactions with a risk level of LOW (score < 40) |
| `Show top 10 riskiest transactions` | Shows the N most risky transactions sorted by risk score (default: 10) |
| `Show suspicious transactions` | Displays all medium and high-risk transactions |

---

## Transaction Details & Explanations

| Question | Description |
|---|---|
| `Explain transaction #3` | Shows full details and risk explanation for a specific transaction |
| `Why is transaction #4 suspicious?` | Explains the risk factors contributing to a transaction's score |
| `Show details for transaction #16` | Displays all fields and risk analysis for a transaction |
| `Transaction #0` | Quick lookup of a transaction by its ID |

---

## Filtering by Transaction Type

| Question | Description |
|---|---|
| `Show all CASH_OUT transactions` | Filters and displays only CASH_OUT type transactions |
| `Show TRANSFER transactions` | Filters and displays only TRANSFER type transactions |
| `Show PAYMENT transactions` | Filters and displays only PAYMENT type transactions |
| `Show DEBIT transactions` | Filters and displays only DEBIT type transactions |
| `List CASH_IN transactions` | Filters and displays only CASH_IN type transactions |

---

## Filtering by Amount

| Question | Description |
|---|---|
| `Show transactions above 100000` | Displays transactions with amounts exceeding the specified value |
| `Show transactions below 5000` | Displays transactions with amounts below the specified value |
| `Show transactions between 5000 and 50000` | Displays transactions within the specified amount range |
| `Show the 5 largest transactions` | Shows the N largest transactions by amount |
| `Show the 10 smallest transactions` | Shows the N smallest transactions by amount |
| `Show large transactions` | Shows transactions above $100,000 (default threshold) |

---

## Fraud Detection

| Question | Description |
|---|---|
| `Show fraud-flagged transactions` | Displays transactions where isFraud or isFlaggedFraud is set to 1 |
| `Show balance anomalies` | Shows transactions with unusual balance patterns (account drained, balance discrepancy, zero-balance origin) |
| `Explain why a transaction is suspicious` | Provides detailed risk factor breakdown for high-risk transactions |
| `Show cashout and transfer patterns` | Displays CASH_OUT and TRANSFER transactions (higher risk types) |

---

## Statistics & Summaries

| Question | Description |
|---|---|
| `Show summary statistics` | Displays comprehensive dataset overview including risk distribution, transaction types, and fraud indicators |
| `Give me a dataset overview` | Same as summary statistics |
| `Show stats` | Abbreviated form for summary statistics |

---

## Account Analysis

| Question | Description |
|---|---|
| `Show info for account C1231006815` | Displays summary for a specific originating account |
| `What transactions involve C840083671?` | Shows all transactions related to the specified account |

---

## General

| Question | Description |
|---|---|
| `help` | Shows the full list of supported query types |
| `hello` / `hi` / `hey` | Displays the welcome message with example queries |
| `Show all transactions` | Lists all transactions in the dataset |

---

## Risk Scoring Rules

The chatbot uses the following rule-based risk scoring system:

### Risk Factors

| Factor | Score | Condition |
|---|---|---|
| **Amount Risk** | 0-25 pts | Based on transaction amount thresholds |
| **Type Risk** | 0-20 pts | CASH_OUT (20), TRANSFER (15), DEBIT (5) |
| **Balance Anomaly** | 0-20 pts | Account drained, balance discrepancy, zero-balance origin |
| **Repeat Account** | 0-15 pts | Multiple transactions from same account |
| **Destination Risk** | 0-10 pts | High-traffic destination accounts |
| **Cash-out Pattern** | 0-10 pts | Large transfer followed by cash-out (layering) |
| **Fraud Flag Boost** | +15 pts | If isFraud = 1 |
| **Flagged Fraud Boost** | +10 pts | If isFlaggedFraud = 1 |

### Risk Levels

| Level | Score Range | Description |
|---|---|---|
| **LOW** | 0 - 39 | Minimal risk signals detected |
| **MEDIUM** | 40 - 70 | Moderate risk, warrants attention |
| **HIGH** | 71 - 100 | Significant risk, requires investigation |

### Amount Thresholds

| Amount Range | Risk Score |
|---|---|
| > $500,000 | 25 points |
| > $200,000 | 20 points |
| > $100,000 | 15 points |
| > $10,000 | 10 points |
| > $5,000 | 5 points |
| <= $5,000 | 0 points |

---

## Tips

- You can use natural language - the chatbot understands variations of queries
- Transaction IDs start from 0
- Account IDs start with `C` (customer) or `M` (merchant)
- Combine keywords naturally: "Why is transaction #4 risky?"
- The chatbot will suggest example queries if it doesn't understand your input
