# Explainable Credit Risk Model using Machine Learning in Corporate Banking

> **Predicting loan default on 230,288 Lending Club loans (2007–2014) using four ML models across three paradigms, with SHAP-based explainability for regulatory compliance.**

---

## Project Summary

Banks face an asymmetric cost problem in credit risk — missing a real defaulter costs the entire loan principal, while incorrectly rejecting a good borrower costs one customer. Standard ML optimises for accuracy, which is misleading on imbalanced data (81% non-defaulter). This project builds a complete, production-oriented credit risk framework that prioritises **Recall** as the primary metric, evaluated across five complementary dimensions, and explained using SHAP for individual borrower-level audit trails.

---

## Final Results

| Model | Recall | Precision | Accuracy | F1 | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| Logistic Regression | 71.83% | 28.21% | 59.99% | 0.412 | **70.06%** | 0.2397 |
| XGBoost | 71.15% | 27.77% | 59.43% | 0.407 | 69.60% | 0.2356 |
| RF Full (20 feat) | 69.57% | 27.03% | 58.61% | 0.398 | 68.64% | 0.2380 |
| RF Reduced (14 feat) | 72.32% | 26.44% | 56.59% | 0.398 | 68.47% | 0.2386 |
| **Gradient Boosting** | 65.97% | **29.04%** | **62.98%** | 0.407 | 69.61% | **0.2212** |

**Final models selected:**
- **Gradient Boosting** — primary deployment model (best calibration, Brier 0.2212, ranked 1st in McNemar's test)
- **Logistic Regression** — interpretability model (highest F1 0.412, ROC-AUC 70.06%, regulatory audit ready)

---

## Repository Structure

```
credit-risk-explainable-ml/
├── README.md
├── Phase1_Data_Cleaning/
│   ├── README.md
│   ├── Data_cleaning.ipynb
│   ├── data_cleaning_report.tex
│   └── Data_cleaning_report.pdf
├── Phase2_EDA/
│   ├── README.md
│   ├── eda.ipynb
│   ├── EDA_Report.pdf
│   └── EDA_Report_With_Visuals.docx
├── Phase3_Modelling/
│   ├── README.md
│   ├── modelling_preprocessing.ipynb
│   └── modelling_2.ipynb
├── Phase4_Model_Evaluation/
│   ├── README.md
│   ├── final_model_evaluation.ipynb
│   └── final_testing_on_unlabelled_data/
│       └── Testing_data_created.ipynb
├── Phase5_Explainability/
│   ├── README.md
│   └── shap_analysis.ipynb
├── .gitignore
└── requirements.txt
```

---

## Pipeline

```
Raw Data (466,285 rows x 75 features)
         |
Phase 1 - Data Cleaning
  - Dropped 18 null/irrelevant columns
  - Treated missing values in 20 columns
  - Output: 230,288 resolved loans x 55 features
         |
Phase 2 - Exploratory Data Analysis
  - Ranked all features by default rate
  - grade/sub_grade strongest (6.5% to 43.7%)
  - revol_bal and emp_length confirmed useless
         |
Phase 3 - Modelling
  - Chi-Square + Mann-Whitney + Anderson-Darling feature selection
  - Binned 4 continuous variables
  - No SMOTE - class weights only
  - 4 models: LR, XGBoost, RF, GBM
         |
Phase 4 - Model Evaluation
  - 6 dimensions: metrics, importance, ROC, agreement on
    235,399 active loans, calibration, McNemars test
  - Final ranking: GBM > LR > XGBoost > RF Full > RF Reduced
         |
Phase 5 - Explainability
  - SHAP on GBM (TreeExplainer) and LR (LinearExplainer)
  - 15 plots: bar, beeswarm, waterfall, dependence, comparison
  - All EDA findings confirmed - no contradictions
```

---

## Key Findings

**sub_grade is the single dominant predictor**
Grade A defaults at 6.5%, Grade G at 43.7% — a 7x gap, perfectly monotonic across all 35 sub-grades. Confirmed by EDA, feature importance, and SHAP.

**All five models converge to the same ceiling**
70-72% recall and 68-70% AUC. The ceiling is in the data signal, not the algorithms.

**Revolving balance has zero predictive power**
Identical distributions for both groups. Revolving utilization shows 3x increase from low to high — same raw data, completely different information.

**Calibration separates GBM from LR**
GBM Brier 0.2212 vs LR 0.2397. When GBM predicts 70% default probability, reality is closer to 70% than any other model.

**Two-model approach justified**
Deployment needs calibrated probabilities (GBM). Regulatory compliance needs transparent explanations (LR).

---

## Why Recall Not Accuracy

A model predicting "Not Defaulter" for everyone gets 81% accuracy but catches zero defaults. Recall minimises false negatives. All tuning optimised on Recall with AUC as secondary.

---

## Why Not SMOTE

- Empirical: SMOTE + class_weight collapsed to 99.88% recall / 19.62% accuracy
- Fundamental: SMOTE creates artificial interpolated points. Class weights upweight real defaulters — model learns genuine patterns

---

## Statistical Feature Selection

| Feature Type | Test | Rule |
|---|---|---|
| Categorical vs binary target | Chi-Square | p < 0.05 keep |
| Numerical non-normal | Mann-Whitney | p < 0.05 keep |
| Normality check | Anderson-Darling | A2 > critical = non-normal |

All 7 numerical columns rejected normality. application_type failed Chi-Square (p=1.0) and was dropped.

---

## SHAP Summary

| Feature | GBM SHAP | LR SHAP | Agreement |
|---|---|---|---|
| sub_grade | 0.19 (1st) | 0.19 (3rd) | Both top 3 |
| bin_annual_inc | 0.06 (2nd) | 0.21 (2nd) | Both top 3 |
| bin_int_rate | 0.04 (5th) | 0.18 (4th) | Both top 5 |
| bin_dti | 0.05 (3rd) | 0.16 (5th) | Both top 6 |
| term | 0.05 (4th) | 0.14 (6th) | Both top 6 |
| total_acc_bin | 0.00 | ~0.00 | Both confirm useless |

---

## Dataset

- **Source:** Lending Club — [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Period:** 2007-2014 (spans 2008 financial crisis)
- **Raw:** 466,285 rows x 75 columns
- **Cleaned:** 230,288 resolved loans x 55 features
- Raw CSV files not included — download from Kaggle

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data processing | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Statistical tests | scipy, statsmodels |
| ML models | scikit-learn, xgboost |
| Explainability | shap |

---

## How to Run

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-explainable-ml.git
cd credit-risk-explainable-ml
pip install -r requirements.txt
# Download data from Kaggle, place in Datasets/
# Run notebooks in phase order 1 through 5
```

---

## Future Work

- **Survival Analysis** — Kaplan-Meier and Cox PH for time-to-default
- **Macro Stress Testing** — Monte Carlo simulation under adverse conditions
- **Model monitoring** — SHAP value drift as retraining signal

---

*Academic data science project demonstrating end-to-end ML pipeline, statistical feature selection, multi-criteria model evaluation, and regulatory-grade explainability.*
