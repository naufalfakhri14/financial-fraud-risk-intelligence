# 💳 Financial Fraud & Transaction Risk Intelligence

## Detect → Prioritize → Investigate → Quantify Exposure

An end-to-end fraud detection and transaction risk intelligence project built to answer a practical financial risk question:

> **How can a financial platform identify suspicious transactions while balancing fraud detection against false-positive investigation workload?**

The project goes beyond a simple fraud classifier. It connects:

```text
Business Problem
      ↓
Fraud & Financial Exposure Analysis
      ↓
Transaction Behavior Analysis
      ↓
Feature Engineering
      ↓
Feature Availability / Leakage Audit
      ↓
Benchmark Model
      ↓
Real-Time Risk Model
      ↓
Precision–Recall & PR-AUC Evaluation
      ↓
Threshold Optimization
      ↓
Financial Impact Analysis
      ↓
Risk Scoring
      ↓
Investigation Prioritization
      ↓
Streamlit Fraud Operations Center
```

---

# 1. 🎯 Business Problem

Financial platforms process large volumes of transactions where only a very small proportion may be fraudulent.

The real question is not simply:

> **“Can we classify fraud?”**

It is:

> **“How can we identify enough fraudulent activity to reduce financial exposure without overwhelming investigators with false-positive alerts?”**

Two errors have different consequences.

### False Negative
Fraud is missed.

Potential consequences:
- Financial loss
- Unprotected exposure
- Missed investigation opportunities

### False Positive
A legitimate transaction is flagged.

Potential consequences:
- Investigation workload
- Customer friction
- Operational cost

Therefore, this project treats fraud detection as a **risk-ranking and threshold decision problem**, not an accuracy-only problem.

---

# 2. 📊 Dataset

The project uses the **PaySim / Synthetic Financial Dataset for Fraud Detection**.

The dataset contains transaction-level information including:

- `step`
- `type`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`
- `isFraud`
- `isFlaggedFraud`

The analysis contains approximately:

> **6.36 million transactions**

with:

> **8,213 fraudulent transactions**

---

# 3. 🔎 Data Audit

## Fraud prevalence

| Metric | Result |
|---|---:|
| Total transactions | **6,362,620** |
| Fraud transactions | **8,213** |
| Fraud rate | **0.1291%** |

Fraud is therefore a severe **class-imbalance problem**.

Only about 1 in 775 transactions is labeled as fraud.

### Why accuracy is not enough

A model could obtain very high accuracy by predicting almost every transaction as legitimate.

The project therefore focuses on:

- Precision
- Recall
- F1-score
- ROC-AUC
- **PR-AUC**

---

# 4. 💰 Financial Exposure

The data contains approximately:

> **1.144T total transaction value**

while fraudulent transactions represent:

> **12.056B fraud transaction value**

### Key insight

Although fraud represents only **0.1291% of transactions**, the associated financial exposure is substantial.

This motivates a risk system that considers both:

> **Fraud frequency + financial exposure**

---

# 5. 🚨 Fraud Concentration by Transaction Type

| Transaction Type | Transactions | Fraud | Fraud Rate |
|---|---:|---:|---:|
| **TRANSFER** | 532,909 | 4,097 | **0.77%** |
| **CASH_OUT** | 2,237,500 | 4,116 | **0.18%** |
| CASH_IN | 1,399,284 | 0 | 0.00% |
| DEBIT | 41,432 | 0 | 0.00% |
| PAYMENT | 2,151,495 | 0 | 0.00% |

All observed fraud cases occurred in **TRANSFER** and **CASH_OUT** transactions.

### Business interpretation

These transaction types can receive higher monitoring priority in a risk-scoring workflow.

This is an observed association in the dataset, not causal evidence.

---

# 6. 🧠 Transaction Behavior Analysis

## Transaction amount

| | Legitimate | Fraud |
|---|---:|---:|
| Average amount | **178K** | **1.47M** |

Fraudulent transactions are substantially larger on average.

## Origin balance

| | Legitimate | Fraud |
|---|---:|---:|
| Average old balance | **832.8K** | **1.65M** |
| Average new balance | **856.0K** | **192.4K** |

## Origin balance depletion

We created:

```python
origin_balance_drop = oldbalanceOrg - newbalanceOrig
```

Observed mean:

| | Legitimate | Fraud |
|---|---:|---:|
| Origin balance change | **-23K** | **1.46M** |

### Insight

Fraudulent transactions show unusually large depletion of the originating account balance.

This supported the creation of behavioral features.

---

# 7. 🧮 Feature Engineering

The analysis created behavioral variables including:

### Origin balance depletion

```python
origin_balance_drop = oldbalanceOrg - newbalanceOrig
```

### Destination balance change

```python
destination_balance_change = newbalanceDest - oldbalanceDest
```

### Amount relative to origin balance

```python
amount_to_oldbalance_ratio = amount / oldbalanceOrg
```

The purpose was to capture:

> **Transaction magnitude + account-balance behavior**

rather than relying only on raw columns.

---

# 8. ⚠️ Feature Audit & Leakage Awareness

A balance consistency feature was tested:

```python
balance_inconsistency =
oldbalanceOrg - newbalanceOrig - amount
```

Its distribution was:

- Mean: **-201K**
- Median: **-68.7K**
- Maximum: approximately **0**

Because it largely reflected transaction/accounting mechanics instead of a clean independent anomaly signal, it was excluded from the final feature set.

This illustrates an important principle:

> **A highly predictive feature is not automatically a good production feature.**

The project considered both predictive value and information availability.

---


# 8.1 🔬 Professional Data Analysis Perspective

The exploratory stage was structured as a **business-oriented diagnostic analysis**, not simply a collection of visualizations.

The analysis followed four questions:

```text
1. How common is fraud?
2. Where does fraud concentrate?
3. What transaction behavior differs between fraud and legitimate activity?
4. Which signals can be translated into a predictive risk model?
```

This sequence ensures that the predictive model is grounded in observed data behavior and a clearly defined business problem.

### A. Distribution Analysis

Fraud prevalence was first quantified.

> **Fraud rate = 0.1291%**

This establishes the problem as a **rare-event and highly imbalanced classification problem**.

From a Data Science perspective, this immediately changes the evaluation strategy:

- Accuracy becomes less informative.
- Class imbalance must be considered during model training.
- Precision–recall behavior becomes important.
- Threshold selection becomes an operational decision.

### B. Concentration Analysis

Fraud was segmented by transaction type.

The analysis found that observed fraud was concentrated in:

- `TRANSFER`
- `CASH_OUT`

This is useful for **risk segmentation and monitoring prioritization**.

The analysis deliberately avoids causal interpretation. The correct statement is:

> **Fraud is associated with specific transaction types in this dataset.**

It is not:

> **The transaction type causes fraud.**

### C. Comparative Behavior Analysis

Fraudulent and legitimate transactions were compared using:

- Transaction amount
- Origin account balance
- Destination account balance
- Balance movement

The analysis showed that fraudulent transactions had substantially higher average transaction values and much larger origin-account balance depletion.

This represents a progression from:

```text
Descriptive Analytics
"What happened?"
        ↓
Diagnostic Analytics
"Where and how did it differ?"
        ↓
Predictive Analytics
"Can the observed patterns help predict risk?"
```

### D. Feature Engineering as Analytical Representation

The engineered variables:

```text
origin_balance_drop
destination_balance_change
amount_to_oldbalance_ratio
```

were created to represent **transaction behavior**, not simply to increase the number of model inputs.

The purpose was to translate raw accounting fields into interpretable behavioral signals.

### E. Feature Audit and Information Availability

The `balance_inconsistency` feature was tested and then excluded after its distribution suggested that it largely reflected the simulator's accounting mechanics rather than a stable independent risk signal.

This demonstrates an important Data Science principle:

> **Feature usefulness must be evaluated together with meaning, stability, and information availability at scoring time.**

### F. Model Evaluation as Decision Analysis

Model evaluation was expanded beyond one metric:

```text
Discrimination
→ ROC-AUC / PR-AUC

Classification behavior
→ Precision / Recall / F1

Operational workload
→ False positives / False negatives

Financial consequence
→ Detected / missed transaction value
```

This creates a complete analytical chain:

```text
Data Analysis
      ↓
Feature Engineering
      ↓
Predictive Modeling
      ↓
Threshold Decision
      ↓
Business Consequence
```

---

# 8.2 🧠 Professional Data Science Interpretation

## Why the benchmark score was questioned

The broader-feature Random Forest produced near-perfect performance.

A shallow analysis could simply report:

> **99.9998% accuracy**

However, unexpectedly strong performance should trigger a diagnostic review.

The project therefore followed:

```text
Near-perfect benchmark
        ↓
Ask "Why is it so strong?"
        ↓
Inspect feature importance
        ↓
Review feature availability
        ↓
Build a more conservative real-time feature set
```

This is an important professional modeling practice:

> **Unexpectedly strong performance is a reason to investigate the data-generating process, not only to celebrate the metric.**

The benchmark model is therefore retained as evidence of the strong synthetic patterns in PaySim, while the real-time model becomes the main operational narrative.

## Real-Time Modeling Philosophy

The real-time feature set was deliberately reduced to:

```text
step
type
amount
oldbalanceOrg
oldbalanceDest
```

The objective was to create a more conservative scoring scenario based on information that is more plausibly available during transaction evaluation.

The resulting Random Forest achieved:

> **PR-AUC = 0.5000**

and:

> **ROC-AUC = 0.9858**

At the default threshold of 0.50:

> **Precision = 11.27%**  
> **Recall = 62.18%**

The model therefore has useful ranking ability, but a default threshold creates substantial alert workload.

---

# 8.3 📐 Precision–Recall as the Core Decision Framework

Because fraud is extremely rare, precision and recall represent competing business objectives.

### Recall

> **How much fraud can we catch?**

High recall is useful when the business strongly prioritizes minimizing missed fraud.

### Precision

> **How trustworthy are our alerts?**

High precision is useful when investigator capacity is limited and false alerts are costly.

The threshold therefore becomes an operational policy:

```text
Lower threshold
→ More alerts
→ More fraud captured
→ More false positives

Higher threshold
→ Fewer alerts
→ Higher precision
→ More fraud missed
```

This is why the project does not claim that one threshold is universally optimal.

Instead, it defines operating modes:

| Mode | Threshold | Primary objective |
|---|---:|---|
| Sensitive | 0.50 | Maximize fraud capture |
| Standard Review | 0.6665 | Balance capture and workload |
| High-Confidence | 0.7193 | Reduce false-positive workload |

---

# 8.4 💼 Business Translation

A professional Data Science workflow should convert model probabilities into decisions a business stakeholder can act on.

The project therefore creates:

```text
Fraud probability
        ↓
Risk level
        ↓
Alert priority
        ↓
Transaction value
        ↓
Investigation decision
```

For example:

> **CRITICAL risk → 2,488 transactions → 6.16B transaction value**

This creates a direct connection between:

**model output → operational workload → financial exposure**

instead of presenting an isolated machine-learning metric.

---

# 8.5 📊 Analytical Decision Summary

### Finding 1 — Fraud is rare

**Evidence:** 0.1291% fraud rate.

### Finding 2 — Fraud is financially meaningful

**Evidence:** approximately 12.06B in fraudulent transaction value.

### Finding 3 — Fraud is concentrated

**Evidence:** observed fraud occurs in TRANSFER and CASH_OUT.

### Finding 4 — Fraudulent transactions have different behavior

**Evidence:** higher average transaction values and substantially larger origin-account balance depletion.

### Finding 5 — Accuracy alone is insufficient

**Evidence:** severe class imbalance makes accuracy easy to inflate.

### Finding 6 — Benchmark performance requires scrutiny

**Evidence:** the broader Random Forest produced near-perfect metrics on the synthetic dataset.

### Finding 7 — The real-time feature set provides a more conservative operational test

**Evidence:** Random Forest PR-AUC = 0.5000.

### Finding 8 — Threshold changes operational outcomes

**Evidence:** false positives fall from 20,822 at threshold 0.50 to 746 at 0.7193.

### Finding 9 — Risk scoring helps prioritize financial exposure

**Evidence:** 2,488 CRITICAL transactions represent approximately 6.16B in transaction value.

### Final analytical conclusion

> **The most valuable output is not simply a fraud classifier. It is a risk-prioritization framework that connects transaction behavior, predictive probability, operational thresholds, investigation workload, and financial exposure.**

---


# 9. 🧪 Benchmark Modeling

A broader feature set was first tested as a benchmark.

## Logistic Regression benchmark

| Metric | Result |
|---|---:|
| Accuracy | **97.54%** |
| Precision | **10.88%** |
| Recall | **88.55%** |
| F1 | **19.39%** |
| ROC-AUC | **0.9858** |
| PR-AUC | **0.6775** |

This model had very high recall but low precision at threshold 0.50.

### Interpretation

> It was effective at catching fraud but generated a large number of false alerts.

---

# 10. 🌲 Benchmark Random Forest

The broader-feature Random Forest achieved near-perfect results:

| Metric | Result |
|---|---:|
| Accuracy | **99.9998%** |
| Precision | **100.00%** |
| Recall | **99.95%** |
| F1 | **99.98%** |
| ROC-AUC | **~1.00** |
| PR-AUC | **~1.00** |

Confusion matrix:

```text
True Negative   = 1,268,270
False Positive  = 0
False Negative  = 2
True Positive   = 4,252
```

### Why we did not blindly present this as production performance

The PaySim dataset is synthetic and its fraud-generation rules create very strong patterns.

The benchmark result is therefore treated as:

> **Synthetic benchmark performance**

rather than evidence of production-level fraud detection.

This triggered a second experiment using a more conservative real-time feature set.

---

# 11. 🕐 Real-Time Risk Scoring Model

For a more production-like scenario, the final model used:

```text
step
type
amount
oldbalanceOrg
oldbalanceDest
```

The objective was:

> **Estimate transaction risk using information that is more plausibly available during transaction evaluation.**

---

# 12. 🤖 Real-Time Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 95.39% | 5.78% | **83.45%** | 10.81% | 0.9750 | 0.3302 |
| **Random Forest** | **98.24%** | **11.27%** | 62.18% | **19.08%** | **0.9858** | **0.5000** |

## Final model

> **Random Forest**

The decision was based primarily on:

- PR-AUC
- ROC-AUC
- Precision
- F1

rather than accuracy alone.

Logistic Regression had higher recall, but Random Forest had stronger overall ranking performance and a better precision/F1 profile for an investigation-oriented risk queue.

---

# 13. 🎚️ Threshold Optimization

A fraud probability is not automatically an operational decision.

At threshold `0.50`, the real-time Random Forest produced:

- Precision: **11.27%**
- Recall: **62.18%**
- F1: **19.08%**

The project therefore evaluated multiple thresholds.

---

# 14. ⚖️ Operational Threshold Trade-off

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5000 | 11.27% | **62.18%** | 19.08% | 2,645 | 20,822 | 1,609 |
| 0.6665 | 30.00% | 52.40% | 38.16% | 2,229 | 5,200 | 2,025 |
| **0.7193** | **70.02%** | 40.95% | **51.68%** | 1,742 | 746 | 2,512 |

### Operating modes

**0.50 — Sensitive Screening**
- Higher recall
- Very high false-positive workload
- Suitable for broad monitoring

**0.6665 — Standard Review**
- Middle ground
- Precision ≈ 30%
- Recall ≈ 52%

**0.7193 — High-Confidence Review**
- Precision ≈ 70%
- Recall ≈ 41%
- Only 746 false positives in the evaluated test set

The project uses:

> **0.7193**

as the **high-confidence operating threshold**.

This is an operating scenario, not a universal optimal threshold.

---

# 15. 💵 Financial Impact of Thresholds

Threshold selection was evaluated not only through classification metrics, but also through transaction value.

| Threshold | Detected Fraud Value | Missed Fraud Value | Flagged Transaction Value |
|---:|---:|---:|---:|
| 0.50 | **6.34B** | 349.34M | 10.05B |
| 0.6665 | **6.16B** | 525.36M | 7.13B |
| **0.7193** | **5.98B** | 707.43M | **6.16B** |

### Key business insight

A higher threshold substantially reduces the amount of transaction value sent to review and reduces false positives, but increases missed fraud exposure.

Therefore:

> **Threshold optimization is a business policy decision, not simply a mathematical optimization problem.**

---

# 16. 🧩 Model Explainability

Feature importance for the final real-time Random Forest:

| Feature | Importance |
|---|---:|
| **oldbalanceOrg** | **0.3027** |
| **amount** | **0.2290** |
| **type_TRANSFER** | **0.1003** |
| **type_CASH_IN** | **0.0980** |
| **type_PAYMENT** | **0.0927** |
| **oldbalanceDest** | **0.0793** |
| **type_CASH_OUT** | **0.0720** |
| **step** | **0.0253** |
| **type_DEBIT** | **0.0008** |

### Interpretation

The strongest model signals are:

1. Origin-account balance
2. Transaction amount
3. Transaction type
4. Destination-account balance
5. Transaction timing

This allows the risk score to be explained through transaction characteristics rather than presenting a completely opaque prediction.

Feature importance is interpreted as model behavior, not causal evidence.

---

# 17. 🚦 Operational Risk Scoring

Fraud probability is translated into four practical risk categories:

| Risk | Probability |
|---|---|
| **LOW** | < 0.30 |
| **MEDIUM** | 0.30–0.49 |
| **HIGH** | 0.50–0.7192 |
| **CRITICAL** | ≥ 0.7193 |

This turns the model into an operational prioritization tool.

---

# 18. 🔥 Risk Distribution

On the real-time test set:

| Risk Level | Transactions | Transaction Value | Average Amount |
|---|---:|---:|---:|
| LOW | **1,212,665** | 206.10B | 169,960 |
| MEDIUM | **36,392** | 5.13B | 140,909 |
| HIGH | **20,979** | 3.89B | 185,616 |
| **CRITICAL** | **2,488** | **6.16B** | **2.47M** |

### Important business finding

Only:

> **2,488 transactions**

are classified as CRITICAL.

However, they represent:

> **6.16B transaction value**

and average approximately:

> **2.47M per transaction**

This demonstrates why risk scoring can be more useful than simply labeling transactions as fraud/not-fraud.

The fraud team can prioritize:

> **High-confidence + high-value alerts first.**

---

# 19. 🧭 Fraud Handling Strategy

The model is designed as a **decision-support and investigation prioritization system**, not as an autonomous transaction-blocking system.

Recommended workflow:

```text
Transaction
     ↓
Fraud Probability
     ↓
Risk Level
     ↓
Investigation Priority
```

### LOW
- Normal processing
- Passive monitoring

### MEDIUM
- Additional monitoring
- Additional rules where available

### HIGH
- Investigation queue
- Additional transaction checks

### CRITICAL
- High-priority investigation
- Consider transaction review/hold according to business policy
- Prioritize high-value cases

The final business action should remain subject to operational policies and human/rule-based controls.

---

# 20. 🖥️ Streamlit Fraud Operations Center

The planned application translates the model into an interactive operational dashboard.

## Executive Risk Overview

Shows:

- Total transactions
- Fraud rate
- Fraud transaction value
- Critical transaction count
- Critical transaction value
- Risk distribution
- Fraud concentration by transaction type

## Risk Explorer

Provides interactive exploration of:

- Transaction type
- Transaction amount
- Account balance characteristics
- Risk level
- Financial exposure

## Threshold Simulator

Users can change the risk threshold and immediately see:

- Precision
- Recall
- F1
- True positives
- False positives
- False negatives
- Detected fraud value
- Missed fraud value
- Flagged transaction value

This makes the **prediction → decision trade-off** visible.

## Transaction Risk Simulator

Users can enter a transaction scenario and obtain:

- Fraud probability
- Risk level
- Investigation priority
- Risk signals

## Investigation Queue

High-confidence alerts can be sorted by:

- Fraud probability
- Transaction amount
- Risk level
- Financial exposure

## Model & Insights

Communicates:

- Benchmark model results
- Real-time model results
- PR-AUC
- Threshold rationale
- Feature importance
- Limitations

---

# 21. 📌 Final Outputs

The complete project produces:

### 1. Risk Prediction

A Random Forest model produces fraud probability.

### 2. Operational Risk Score

```text
LOW
MEDIUM
HIGH
CRITICAL
```

### 3. Operating Thresholds

```text
0.50   → Sensitive screening
0.6665 → Standard review
0.7193 → High-confidence review
```

### 4. Financial Exposure

```text
Detected fraud value
Missed fraud value
Flagged transaction value
```

### 5. Investigation Priority

A smaller set of high-confidence transactions can be prioritized for manual or rule-based investigation.

---

# 22. 🏆 Project Impact

The key value of the project is the transformation:

```text
Machine Learning Probability
          ↓
Operational Threshold
          ↓
Risk Category
          ↓
Investigation Priority
          ↓
Financial Exposure
```

Instead of stopping at:

> **“Is this transaction fraud?”**

the system answers:

> **“Which transactions should investigators prioritize, at what confidence level, and what financial exposure is associated with those alerts?”**

This makes the project closer to a **transaction risk intelligence system** than a standalone classification model.

---

# 23. ⚠️ Limitations

This project uses a **synthetic PaySim dataset**, so model performance should not be interpreted as production performance on real banking transactions.

Important limitations include:

- Synthetic fraud-generation patterns may be easier to learn than real fraud behavior.
- The benchmark model achieved near-perfect performance and is therefore treated as a synthetic benchmark, not a production claim.
- The real-time model intentionally uses a more conservative feature set.
- The 0.7193 threshold is an operating scenario for high-confidence review, not a universal optimum.
- False-positive and false-negative costs were not converted into an explicit monetary cost function.
- Real production deployment would require streaming infrastructure, model monitoring, concept-drift detection, device/account/network signals, case management, governance, and human oversight.

---

# 24. 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression
- Random Forest
- One-Hot Encoding
- StandardScaler
- Class-weighted learning
- Precision–Recall Analysis
- PR-AUC
- ROC-AUC
- Matplotlib
- Seaborn
- Plotly
- Streamlit

---

# 25. 📁 Project Structure

```text
financial-fraud-risk-intelligence/
│
├── data/
│   ├── raw/
│   │   └── paysim.csv
│   └── processed/
│
├── notebooks/
│   └── fraud_detection_analysis.ipynb
│
├── models/
│
├── visualizations/
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 26. 🔄 End-to-End Workflow

```text
Business Problem
        ↓
Data Audit
        ↓
Fraud Prevalence & Exposure
        ↓
Transaction Behavior Analysis
        ↓
Feature Engineering
        ↓
Feature Availability / Leakage Audit
        ↓
Benchmark Modeling
        ↓
Real-Time Feature Modeling
        ↓
Model Comparison
        ↓
PR-AUC Evaluation
        ↓
Threshold Optimization
        ↓
Financial Impact Analysis
        ↓
Risk Scoring
        ↓
Investigation Prioritization
        ↓
Streamlit Fraud Operations Center
```

---

# 27. 💼 Interview-Ready Summary

> **“I built an end-to-end transaction fraud-risk intelligence system using the PaySim benchmark dataset. I first quantified severe class imbalance and financial exposure, then analyzed transaction behavior and engineered balance-related features. A benchmark model achieved near-perfect performance on the synthetic data, so I performed a feature-availability audit and evaluated a more conservative real-time feature set. The final Random Forest achieved a PR-AUC of 0.50. I then optimized the operating threshold using precision-recall and financial impact analysis. At a high-confidence threshold of 0.7193, precision reached 70.02% with 746 false positives, while the model identified approximately 5.98B in fraudulent transaction value. Finally, I translated model probabilities into LOW, MEDIUM, HIGH, and CRITICAL risk levels and connected those scores to investigation workload and financial exposure, turning the predictive model into a business-oriented risk-prioritization framework.”**

---

# 👤 Author

**Naufal Fakhri**

Data Science · Machine Learning · Data Analytics · Risk Intelligence
