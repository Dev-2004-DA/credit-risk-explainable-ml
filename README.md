# Explainable Credit Risk Model using Machine Learning in Corporate Banking

> **Predicting loan default on 230,288 Lending Club loans (2007–2014) — binary classification, survival analysis, SHAP explainability, and a unified Streamlit dashboard across 6 phases.**

---

## Project Summary

Banks face an asymmetric cost problem in credit risk — missing a real defaulter costs the entire loan principal, while incorrectly rejecting a good borrower costs one customer. Standard ML optimises for accuracy, which is misleading on imbalanced data (81% non-defaulter). This project builds a complete, production-oriented credit risk framework across two core research questions:

| Question | Method |
|---|---|
| **Who will default?** | LR, XGBoost, Random Forest, Gradient Boosting |
| **When will they default?** | Kaplan-Meier, Cox PH, XGBoost Survival |

Both are explained using SHAP and combined into a Streamlit dashboard for real-time borrower-level risk assessment.

---

## Final Results — Classification

| Model | Recall | Precision | Accuracy | F1 | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| Logistic Regression | 71.83% | 28.21% | 59.99% | 0.412 | **70.06%** | 0.2397 |
| XGBoost | 71.15% | 27.77% | 59.43% | 0.407 | 69.60% | 0.2356 |
| RF Full (20 feat) | 69.57% | 27.03% | 58.61% | 0.398 | 68.64% | 0.2380 |
| RF Reduced (14 feat) | 72.32% | 26.44% | 56.59% | 0.398 | 68.47% | 0.2386 |
| **Gradient Boosting** | 65.97% | **29.04%** | **62.98%** | 0.407 | 69.61% | **0.2212** |

**Final models selected:**
- **Gradient Boosting** — primary deployment model (best calibration Brier 0.2212, ranked 1st in McNemar's test)
- **Logistic Regression** — interpretability model (highest F1 0.412, ROC-AUC 70.06%, regulatory audit ready)

---

## Final Results — Survival Analysis

| Model | Purpose | Key Output |
|---|---|---|
| Kaplan-Meier | Non-parametric survival curves | Default peaks months 5-10, Grade A vs G clearly separated |
| Cox PH | Semi-parametric hazard estimation | Hazard ratios — int_rate, sub_grade, income, term dominant |
| XGBoost Survival | Non-linear survival prediction | int_rate importance 0.3212 — top survival predictor |

**Sample borrower (C3, 60-month, $25K, income $75K):**

| Model | Prediction |
|---|---|
| Logistic Regression | 80.17% default probability |
| Gradient Boosting | 61.48% default probability |
| Cox PH | Risk score 1.03 — above baseline hazard |
| XGBoost Survival | Risk score 1.28 — 28% higher relative risk |

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
├── Phase6_Survival_Analysis/
│   ├── README.md
│   ├── survival_analysis.ipynb
│   └── model_evaluation.ipynb
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
  - Treated missing values in 20 columns — every decision documented
  - Verified accounting identity: total_pymnt = components
  - Output: 230,288 resolved loans x 55 features
         |
Phase 2 - Exploratory Data Analysis
  - Ranked all features by default rate
  - grade/sub_grade strongest (6.5% to 43.7% — 7x gap)
  - revol_bal and emp_length confirmed useless
  - Key finding: default peaks months 5-10 then drops
         |
Phase 3 - Modelling
  - Chi-Square + Mann-Whitney + Anderson-Darling feature selection
  - Binned 4 continuous variables — interpretability + empirical evidence
  - No SMOTE — class weights only
  - 4 models trained: LR, XGBoost, RF, GBM
  - Optimised on Recall — business cost asymmetry justification
         |
Phase 4 - Model Evaluation
  - 6 dimensions: metrics, importance, ROC, agreement on
    235,399 active loans, calibration, McNemars test
  - Final ranking: GBM > LR > XGBoost > RF Full > RF Reduced
  - Monotonicity validated on active portfolio — all models pass
         |
Phase 5 - Explainability (XAI)
  - SHAP on GBM (TreeExplainer) and LR (LinearExplainer)
  - 15 plots: bar, beeswarm, waterfall, dependence, comparison
  - sub_grade dominates classification SHAP (0.19)
  - All EDA findings confirmed — no contradictions
         |
Phase 6 - Survival Analysis
  - Kaplan-Meier survival curves by grade, term, income, DTI
  - Log-Rank test — all group differences p < 0.001
  - Cox PH — hazard ratios for time-to-default
  - XGBoost Survival — int_rate dominates (0.42 SHAP)
  - Unified prediction function + Streamlit dashboard
```

---

## Key Findings

**1. sub_grade is the dominant classification predictor**
Grade A defaults at 6.5%, Grade G at 43.7% — a 7x gap, perfectly monotonic. Confirmed by EDA, feature importance, and SHAP across all models.

**2. int_rate is the dominant survival predictor**
While sub_grade predicts WHETHER a borrower will default, int_rate predicts WHEN. Survival SHAP: int_rate 0.42 (1st) vs classification SHAP: int_rate 0.04 (5th). This divergence is a genuine finding — rate carries timing information that grade does not.

**3. All classification models converge to the same ceiling**
70-72% recall and 68-70% AUC across all models. The ceiling is in the data signal, not the algorithms.

**4. Default risk peaks months 5-10 then drops**
Confirmed by both Phase 2 EDA (visual) and Phase 6 Kaplan-Meier (formal statistical quantification). Borrowers surviving the first year are far more likely to fully repay.

**5. Two-model classification approach justified**
Deployment requires calibrated probabilities (GBM, Brier 0.2212). Regulatory compliance requires transparent explanations (LR, F1 0.412). No single model excels at both simultaneously.

**6. Survival analysis validates all EDA findings independently**
Every risk factor identified in Phase 2 (grade, term, income, DTI, int_rate) is confirmed as a statistically significant time-to-default predictor by Cox PH (p < 0.001 for all).

---

## Statistical Methods Applied

| Test / Method | Phase | Purpose |
|---|---|---|
| Chi-Square Test | Phase 3 | Association between categorical features and binary target |
| Anderson-Darling Test | Phase 3 | Normality check before choosing T-Test vs Mann-Whitney |
| Mann-Whitney U Test | Phase 3 | Non-parametric association for numerical features vs binary target |
| Z-Test | Phase 2 | DTI difference significance across purpose subgroups |
| Log-Rank Test | Phase 6 | Survival curve differences between risk groups |
| Cox PH Model | Phase 6 | Hazard ratios — time-to-default feature attribution |
| Brier Score | Phase 4 | Probability calibration quality |
| McNemar's Test | Phase 4 | Pairwise model significance testing |
| ROC-AUC | Phase 3 & 4 | Threshold-independent discrimination |
| Youden's J Statistic | Phase 4 | Optimal classification threshold selection |
| SHAP (Shapley Values) | Phase 5 & 6 | Feature attribution for classification and survival models |
| Concordance Index | Phase 6 | Survival model discrimination ability |
| Mutual Information (SelectKBest) | Phase 3 | Feature selection inside pipeline |

---

## SHAP Summary — Classification vs Survival

| Feature | Classification GBM SHAP | Survival XGB SHAP | Agreement |
|---|---|---|---|
| sub_grade | **0.19 (1st)** | ~0.05 (5th) | Predicts WHETHER |
| int_rate | 0.04 (5th) | **0.42 (1st)** | Predicts WHEN |
| bin_annual_inc | 0.06 (2nd) | 0.19 (2nd) | Both top 3 |
| bin_dti | 0.05 (3rd) | ~0.04 | Both top 6 |
| term | 0.05 (4th) | Significant | Both confirm |
| inq_last_6mths | Absent | 0.10 (3rd) | Timing-specific signal |
| total_acc_bin | 0.00 | ~0.00 | Both confirm useless |

---

## Why Recall Not Accuracy

A model predicting "Not Defaulter" for everyone gets 81% accuracy but catches zero defaults. Recall minimises false negatives. All tuning optimised on Recall with AUC as secondary.

---

## Why Not SMOTE

- **Empirical:** SMOTE + class_weight collapsed to 99.88% recall / 19.62% accuracy — model predicting everyone as defaulter
- **Fundamental:** SMOTE creates artificial interpolated points. Class weights upweight real defaulters — model learns genuine patterns, not synthetic noise

---

## Complete Risk Framework (All Phases Combined)

A borrower now receives a four-dimensional risk assessment:

```
1. Binary default probability    (LR + GBM)           → Will they default?
2. Time-to-default risk score    (Cox + XGBoost)       → When will they default?
3. SHAP explanation              (Classification + Survival) → Why are they flagged?
4. Survival curve                (Cox PH)              → Month-by-month default probability
```

All delivered through a **Streamlit dashboard** combining all 6 phases into a single interface.

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
| ML — Classification | scikit-learn, xgboost |
| ML — Survival | lifelines, scikit-survival, xgboost (survival objective) |
| Explainability | shap |
| Dashboard | streamlit |
| Reports | LaTeX, python-docx |

---

## How to Run

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-explainable-ml.git
cd credit-risk-explainable-ml
pip install -r requirements.txt
# Download data from Kaggle, place in Datasets/
# Run notebooks in phase order 1 through 6
# For dashboard: streamlit run Phase6_Survival_Analysis/model_evaluation.ipynb
```

---

*Academic data science project demonstrating end-to-end ML pipeline, statistical feature selection, multi-criteria model evaluation, survival analysis, and regulatory-grade explainability across 6 phases.*
