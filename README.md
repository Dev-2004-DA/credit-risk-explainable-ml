# Explainable Credit Risk Model using Machine Learning in Corporate Banking

**Dataset:** Lending Club Loan Dataset — 2007 to 2014
**Domain:** Financial Risk Analytics | Credit Default Prediction

---

## Project Overview

This project builds a complete, production-oriented credit risk framework
for corporate banking using the Lending Club loan dataset (2007–2014).
It addresses two interconnected objectives:

| Objective | Question | Method |
|---|---|---|
| Classification | Who will default? | LR, XGBoost, RF, GBM |
| Explainability | Why did the model decide this? | SHAP |

---

## Dataset

| Property | Value |
|---|---|
| Source | Lending Club public loan data |
| Period | 2007 – 2014 |
| Raw records | 466,285 loans × 75 features |
| After cleaning | 230,288 resolved loans × 49 features |
| Target variable | `loan_status` — Defaulter (19%) vs Not Defaulter (81%) |
| Class imbalance | 81% Not Defaulter, 19% Defaulter |

**Note:** Raw dataset and intermediate CSVs are not included in this
repository due to file size. Download the raw data from
[Kaggle — Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club).

---

## Repository Structure

```
credit-risk-explainable-ml/
│
├── README.md                          ← You are here
│
├── Phase1_Data_Cleaning/
│   ├── README.md                      ← Phase 1 documentation
│   ├── Data_cleaning.ipynb            ← Main cleaning notebook
│   └── Data_cleaning_report.pdf       ← Compiled PDF report
│
├── Phase2_EDA/
│   ├── README.md                      ← Phase 2 documentation
│   ├── eda.ipynb                      ← EDA notebook
│   ├── EDA_Report.pdf                 ← EDA report PDF
│   └── EDA_Report_With_Visuals.docx   ← Word report with embedded plots
│
├── Phase3_Modelling/
│   ├── README.md                      ← Phase 3 documentation
│   ├── modelling_preprocessing.ipynb  ← Feature selection and engineering
│   └── modelling_training.ipynb       ← Model training and tuning
│
├── Phase4_Model_Evaluation/
│   ├── README.md                      ← Phase 4 documentation
│   ├── final_model_evaluation.ipynb   ← Comprehensive evaluation notebook
│   └── final_testing_on_unlabelled_data/
│       └── Testing_data_created.ipynb ← Agreement analysis on active loans
│
├── Phase5_Explainability/
│   ├── README.md                      ← Phase 5 documentation
│   └── xai_shap.ipynb                 ← SHAP analysis (in progress)
│
├── .gitignore
└── requirements.txt
```

---

## Pipeline Summary

```
Raw Data (466K rows × 75 features)
        │
        ▼
Phase 1 — Data Cleaning
  • Dropped 18 irrelevant/null columns
  • Treated missing values in 20 columns
  • Verified accounting identity
  • Output: 466,285 rows × 57 features
        │
        ▼
Phase 2 — EDA
  • Analysed all features vs default rate
  • Ranked predictive strength
  • Identified key patterns and data quality issues
        │
        ▼
Phase 3 — Modelling
  • Statistical feature selection (Chi-Square, Mann-Whitney)
  • Feature engineering (binning, binary flags, purpose grouping)
  • 4 models trained: LR, XGBoost, RF, GBM
  • Optimised on Recall — catching defaulters is the priority
  • Output: 5 trained model files
        │
        ▼
Phase 4 — Model Evaluation
  • Standard metrics, calibration, threshold optimization
  • Agreement analysis on 235K active loans
  • McNemar's significance testing
  • Final model: Gradient Boosting (Brier 0.2212)
        │
        ▼
Phase 5 — Explainability (XAI)
  • SHAP global and local explanations on GBM
  • Feature interaction analysis
  • Regulatory-ready prediction explanations
```

---

## Future Work

- **Survival Analysis** — Kaplan-Meier and Cox Proportional Hazards
  to answer *when* a borrower is likely to default within the loan
  lifecycle (months 5–10 identified as the danger zone in EDA)
- **Macro Stress Testing** — Monte Carlo simulation under normal vs
  stressed economic conditions using the 2008 crisis as calibration

---

## Key Results

### Classification Performance (Final Models on Test Set)

| Model | Recall | Precision | Accuracy | F1 | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| Logistic Regression | 71.83% | 28.21% | 59.99% | 0.412 | 70.06% | 0.2397 |
| XGBoost | 71.15% | 27.77% | 59.43% | 0.407 | 69.60% | 0.2356 |
| RF Full | 69.57% | 27.03% | 58.61% | 0.398 | 68.64% | 0.2380 |
| RF Reduced | 72.32% | 26.44% | 56.59% | 0.398 | 68.47% | 0.2386 |
| **Gradient Boosting** | 65.97% | **29.04%** | **62.98%** | 0.407 | 69.61% | **0.2212** |

**Final model selected: Gradient Boosting** — best probability
calibration (Brier 0.2212), validated on 235,399 active loans.

### Top Predictive Features (Consistent Across All Models)

1. **sub_grade** — 60–74% importance in tree models
2. **bin_int_rate** — 10–35% importance
3. **term** — 60-month loans default at 2× the rate of 36-month
4. **bin_annual_inc** — higher income = lower default, monotonic
5. **bin_dti** — independent signal, low multicollinearity

### Key Finding

> All five models from three ML paradigms converged to the same
> performance ceiling (~70-72% recall, ~68-70% AUC). This confirms
> the ceiling is in the **data signal** — Lending Club's own credit
> assessment (grade, sub_grade, int_rate) is the primary default
> driver, leaving limited room for additional discrimination through
> model complexity alone.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data Processing | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Statistical Tests | scipy, statsmodels |
| ML Models | scikit-learn, xgboost |
| Imbalance Handling | class_weight, scale_pos_weight, compute_sample_weight |
| Explainability | shap |
| Reports | LaTeX, python-docx |

---

## How to Run

**1 — Clone the repository**
```bash
git clone https://github.com/yourusername/credit-risk-explainable-ml.git
cd credit-risk-explainable-ml
```

**2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**3 — Download the raw data**
Download `loan_data_2007_2014.csv` from Kaggle and place it in a
local `Datasets/` folder (not tracked by git).

**4 — Run notebooks in order**
```
Phase1_Data_Cleaning/Data_cleaning.ipynb
Phase2_EDA/eda.ipynb
Phase3_Modelling/modelling_preprocessing.ipynb
Phase3_Modelling/modelling_training.ipynb
Phase4_Model_Evaluation/final_model_evaluation.ipynb
Phase5_Explainability/xai_shap.ipynb
```

---

## Academic Context

This project was developed as part of an academic data science
programme. It demonstrates:

- End-to-end ML pipeline from raw data to explainable deployment
- Statistical rigour in feature selection — every feature justified
  with a formal hypothesis test (Chi-Square, Mann-Whitney, Anderson-Darling)
- Multi-criteria model evaluation — calibration, threshold optimization,
  agreement analysis, and statistical significance testing
- Business-oriented design — recall-first objective, probability
  calibration for risk scoring, regulatory explainability via SHAP

---

*Master README will be finalised after Phase 5 (Explainability) is complete.*
