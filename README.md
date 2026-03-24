# Financial Transaction Fraud Detection System

A dual-language fraud detection system that analyzes financial transaction data to identify potentially fraudulent activity. The system is implemented in both **Python** and **Java**, producing equivalent risk reports.

## Project Structure

```
devin-financial-transactions-demo/
├── README.md
├── .gitignore
├── data/
│   └── Example1.csv                          # Input transaction dataset (100 records)
├── python/
│   ├── requirements.txt                      # Python dependencies
│   └── fraudRiskScoring/                     # Python package
│       ├── __init__.py
│       ├── fraudRiskScoring.py               # Fraud risk scoring engine
│       └── sequenceAnomalyDetection.py       # Sequence anomaly detection engine
├── java/
│   ├── pom.xml                               # Maven build configuration
│   └── src/
│       ├── main/java/com/fraudrisk/
│       │   ├── FraudRiskScoring.java         # Main scoring engine
│       │   ├── Transaction.java              # Input data model
│       │   ├── RiskResult.java               # Intermediate scoring result
│       │   └── RiskReport.java               # Final output data model
│       └── test/java/com/fraudrisk/
│           └── FraudRiskScoringTest.java     # JUnit 5 test suite
└── tests/
    ├── test_fraud_risk_scoring.py            # Python unit tests for fraud risk scoring
    └── test_sequence_anomaly_detection.py    # Python unit tests for sequence anomaly detection
```

## Features

- **Fraud Risk Scoring**: Evaluates each transaction against multiple risk signals (amount, type, balance anomalies, repeat accounts, destination patterns, cash-out patterns) and produces a score from 0-100.
- **Sequence Anomaly Detection**: Analyzes transaction sequences to detect suspicious patterns such as repeated high-value transfers, transfer-then-cashout layering, and sudden amount spikes.
- **Risk Levels**: Transactions are categorized as LOW (< 40), MEDIUM (40-70), or HIGH (> 70) based on their risk score.

## Setup

### Python

```bash
cd python
pip install -r requirements.txt

# Run fraud risk scoring
python -m fraudRiskScoring.fraudRiskScoring

# Run sequence anomaly detection
python -m fraudRiskScoring.sequenceAnomalyDetection
```

### Java

```bash
cd java
mvn clean compile

# Run the application
mvn exec:java -Dexec.mainClass="com.fraudrisk.FraudRiskScoring"

# Run tests
mvn test
```

### Running Python Tests

```bash
# From the repository root
pip install pytest
pytest tests/
```

## Input Data

The system reads transaction data from `data/Example1.csv`. Each record includes:
- Transaction step (time period), type, and amount
- Origin and destination account identifiers
- Account balances before and after the transaction
- Fraud indicator labels (`isFraud`, `isFlaggedFraud`)

## Output

Both implementations produce CSV reports in the `data/` directory:
- `fraud_risk_report.csv` — per-transaction risk scores, levels, and explanations
- `sequence_anomaly_report.csv` — per-customer anomaly scores and pattern descriptions

Generated output files are excluded from version control via `.gitignore`.
