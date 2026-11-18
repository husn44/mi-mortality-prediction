# Myocardial Infarction Mortality Prediction

Predicting in-hospital mortality after a heart attack using machine learning, outperforming a 2026 published Graph Neural Network benchmark.

## Results
| Model | AUC | F1 |
|---|---|---|
| Ghafari et al. XGBoost (baseline) | 0.7865 | – |
| Guo et al. 2026 GNN (benchmark) | 0.8828 | 0.6059 |
| **XGBoost + SMOTE (ours)** | **0.9082** | **0.6811** |
| **LightGBM + Optuna (ours)** | **0.9094** | **0.6695** |

Results averaged over 10 runs × 5-fold stratified cross-validation (50 fits per model).

## Overview
- **Dataset:** [UCI Myocardial Infarction Complications](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications), 1,700 patients, 123 clinical features
- **Feature engineering:** temporal features (differences, means, trends) from 72-hour vital sign readings
- **Class imbalance:** 16% mortality rate, handled with SMOTE applied inside each training fold to avoid data leakage
- **Models:** XGBoost with threshold tuning; LightGBM with Optuna hyperparameter search (50 trials)

## Key Findings
- Both models broke the 0.90 AUC barrier, beating the published GNN benchmark
- LightGBM ranked patients slightly better (higher AUC), while XGBoost caught more deaths (higher F1)
- Temporal vital-sign trends were among the strongest predictors of mortality

## Tech Stack
Python · Pandas · NumPy · scikit-learn · XGBoost · LightGBM · Optuna · imbalanced-learn · Matplotlib · Seaborn

## Team
WIH2001 Data Analytics, Group 4, Universiti Malaya
[Teammate names / GitHub usernames]
