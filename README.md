# Telecom Customer Churn Prediction

A machine learning solution that transforms reactive customer retention into proactive intervention. By predicting churn before it happens, the model identifies 75.6% of at-risk customers and enables targeted retention efforts that could save approximately $5.4M annually.

## The Problem

**Current State:** A telecom company loses customers and only reacts after they leave. The win-back success rate is 15-30%, requiring significant discounts (20-40% off) as a last resort.

**Goal:** Build a predictive model that identifies customers likely to churn *before* they leave, enabling proactive retention with smaller incentives.

## Approach

### Feature Engineering
Created a composite risk score that combines two strong churn predictors:
```
high_churn_risk = MonthlyCharges / tenure
```
Rationale: Customers paying high monthly rates who are new to the service are at greatest risk. This single feature achieved a 0.39 correlation with churn, stronger than any individual feature.

### Model Selection
- Logistic Regression with StandardScaler preprocessing
- Stratified 5-fold cross-validation (handles 27% churn imbalance)
- Compared against Random Forest baseline

### Threshold Optimization
Instead of optimizing for accuracy or F1, the threshold was selected to maximize business value:

```
Value = (Recall x CLV) - (FPR x Discount Cost)
```

Where:
- CLV (Customer Lifetime Value) = $4,400.30
- Retention Discount = $300
- Net value per saved customer = $4,100.30

## Results

| Metric | Value |
|--------|-------|
| Recall | 75.6% |
| False Positive Rate | 25.5% |
| Optimal Threshold | 0.30 |

### Business Impact

| Scenario | Annual Value |
|----------|--------------|
| Baseline (no model) | -$8.2M loss |
| With model | ~$5.4M saved |
| Remaining loss | ~$2.0M (24.4% missed churners) |

The model saves ~$5.4M compared to doing nothing, with the trade-off of offering discounts to some non-churners (low cost at $300 vs. $4,400 CLV).

## Key Insights

1. **Contract type is the strongest predictor** - Month-to-month contracts churn significantly more
2. **First year is critical** - Low tenure strongly correlates with churn
3. **Add-on services reduce churn** - Tech support, security, and backup services are protective
4. **Price sensitivity matters** - High monthly charges drive churn, but total spending does not

## Quick Start

```bash
pip install -r requirements.txt
```

The notebook uses the [Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). You'll need Kaggle API credentials configured to download automatically, or manually download the CSV.

## Tech Stack

- Python 3.8+
- pandas, numpy
- scikit-learn
- matplotlib

## Project Structure

```
telecom-churn-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── utils.py                 # Plotting and evaluation functions
└── churn_analysis.ipynb     # Main analysis notebook
```

## Methodology Notes

The threshold of 0.30 was chosen because in this business context:
- **False negatives are expensive** - Missing a churner costs $4,400 (full CLV)
- **False positives are cheap** - Unnecessary discount costs only $300

This asymmetry justifies a lower threshold that catches more churners at the cost of some false alarms. The economics clearly favor recall over precision.
