# Customer Churn Prediction for Subscription Services

Predicting which telecom subscribers are likely to churn, and turning the model's findings into concrete retention levers.

## Overview

Using the Telco Customer Churn dataset (7,043 customers, 20+ features), this project identifies at-risk subscribers and the factors that drive churn. The data is imbalanced (~73% retained / 27% churned), so the work focuses on correctly capturing the **minority churn class** rather than chasing accuracy alone.

## Approach

**1. EDA & data cleaning** (`Churn Analysis - EDA.ipynb`)
- Converted `TotalCharges` to numeric and removed the small set of resulting nulls (~0.15%).
- Binned `tenure` into 12-month groups; one-hot encoded categorical features.
- Correlation and univariate/bivariate analysis to surface churn drivers.

**2. Modeling** (`Churn Analysis - Model Building.ipynb`)
- Baseline **Decision Tree** and **Random Forest** classifiers.
- Applied **SMOTEENN** (over-sampling + edited nearest neighbors) to rebalance the classes and lift minority-class recall/F1.
- Tried **PCA** for dimensionality reduction (no improvement, so dropped).
- Final model — **Random Forest + SMOTEENN**, serialized with `pickle` for reuse/serving.

## Key Churn Drivers (from EDA)

Highest churn is concentrated in: **month-to-month contracts**, **no online security / no tech support**, **electronic-check payments**, **fiber-optic internet**, and **low-tenure** customers. Long-term contracts and 5+ year customers churn the least.

→ **Retention implication:** prioritize early-tenure, month-to-month, electronic-check customers for proactive outreach, security/support bundling, and contract-upgrade incentives.

## Results

| Stage | Outcome |
|---|---|
| Baseline (imbalanced) | High accuracy but poor recall on the churn class |
| Random Forest + SMOTEENN | Strong, balanced precision/recall/F1 on the minority class (~92% accuracy on the resampled set) |

## Tech Stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn (SMOTEENN) · Matplotlib · Seaborn · pickle

## Repository Structure

```
├── Churn Analysis - EDA.ipynb            # Cleaning, feature prep, churn-driver analysis
├── Churn Analysis - Model Building.ipynb # Modeling, resampling, evaluation, model export
└── README.md
```

## Future Improvements

- Apply resampling **inside cross-validation folds** (not before the train/test split) so the reported metrics reflect true generalization.
- Add gradient-boosted models (XGBoost/LightGBM) and probability-calibrated thresholds for ranking customers by churn risk.

## Key Takeaway

In churn modeling, the business value is in **recall on churners + interpretable drivers** — knowing *who* will leave and *why* is what enables a retention strategy.
