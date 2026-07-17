# Salifort Motors: Employee Retention Analysis

Predicting which employees are likely to leave Salifort Motors — and why — using classification models trained on internal HR data. Built as the capstone project for Google's Advanced Data Analytics Professional Certificate.

## Business Problem

Salifort Motors wanted to understand why employees were leaving and whether departures could be predicted early enough to intervene. The goal: build a model that flags at-risk employees and identify the underlying drivers of turnover, so HR can act on causes rather than symptoms.

## Dataset

- ~15,000 employee records, 10 variables (satisfaction level, last evaluation score, number of projects, average monthly hours, tenure, work accidents, promotions, department, salary tier, and whether the employee left)
- Cleaned to ~12,000 records: removed 3,008 duplicate rows and validated tenure outliers using the IQR method
- No missing values after cleaning

## Approach

1. **EDA** — explored relationships between workload, satisfaction, tenure, and attrition using box plots, scatter plots, and a correlation heatmap
2. **Modeling, Round 1** — built Logistic Regression, Decision Tree, and Random Forest classifiers
3. **Data leakage check** — identified that `satisfaction_level` risked leaking information not available at prediction time; dropped it and engineered a new `overworked` feature (>175 hours/month) instead
4. **Modeling, Round 2** — retrained all three models on the corrected feature set, tuned Decision Tree and Random Forest via `GridSearchCV` with 4-fold cross-validation across accuracy, precision, recall, F1, and AUC

## Results

| Model | Precision | Recall | Accuracy | AUC |
|---|---|---|---|---|
| Logistic Regression | 80% | 83% | 82% | — |
| Decision Tree (Round 2) | 87% | 90.4% | — | 93.8% |
| **Random Forest (Round 2)** ⭐ | **87%** | **90.4%** | **96.2%** | **93.8%** |

**Champion model: Random Forest (Round 2)** — selected for its combination of high accuracy and AUC, with a false-positive/false-negative tradeoff (67 vs. 48) that errs on the safer side for a retention use case: flagging a few extra employees for a check-in costs less than missing someone who was actually about to leave.

## Key Findings

- **Workload is the strongest signal.** Employees with 7 projects are highly likely to leave; 240–315 hours/month is effectively a burnout cluster
- **The 4-year mark is a danger zone.** Satisfaction drops sharply around year four of tenure
- **Promotions retain people.** Almost no employee who received a promotion left the company
- **This isn't a department problem — it's company-wide.** Attrition drivers were consistent across departments, pointing to systemic policy issues rather than isolated management problems

## Recommendations

1. Cap active projects at 5 per employee; reassign overloaded staff immediately
2. Check in with employees at the 3-year mark, before the 4-year satisfaction drop hits
3. Fix the mismatch between hours worked and rewards — compensate overtime fairly or reduce the expectation, and make the policy transparent
4. Promote more employees; it's one of the strongest retention levers in the data
5. **Next modeling steps:** remove `last_evaluation` to further reduce leakage risk, test XGBoost, and explore K-means clustering to segment at-risk employees

## Tools

Python (pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost) · Jupyter Notebook

## Files in This Repo

- `salifort_motors_capstone.ipynb` — full analysis notebook: EDA, modeling, evaluation
- `executive_summary.pdf` — stakeholder-facing summary of approach, results, and recommendations

---
*Capstone project, Google Advanced Data Analytics Professional Certificate (Course 6/7) — completed May 2026*
