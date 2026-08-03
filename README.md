# Explainable Credit Risk & Survival Analysis Model in Corporate Banking

> An end-to-end, production-grade credit risk framework evaluating 466,285 Lending Club loans (2007–2014) through a six-phase pipeline combining cost-sensitive classification, regulatory-compliant SHAP explainability, and temporal survival modeling for IFRS 9 Expected Credit Loss (ECL) compliance.

---

## Project Executive Summary

The 2008 financial crisis catalyzed a paradigm shift under Basel III and IFRS 9, pushing financial institutions away from backward-looking incurred-loss models toward forward-looking, lifetime Expected Credit Loss frameworks:

$$ECL = PD \times LGD \times EAD$$

In consumer and corporate lending, machine learning pipelines face severe economic cost asymmetries: failing to identify a true defaulter (a False Negative) destroys the entire principal, whereas rejecting a creditworthy applicant (a False Positive) only sacrifices interest margin. Standard accuracy metrics are deeply misleading on highly imbalanced credit data (an 81:19 class split).

This project builds an enterprise-grade analytics pipeline that:
- Prioritizes **Recall** as the core optimization target
- Uses non-parametric statistical tests for feature selection
- Rejects invalid synthetic oversampling in favor of native cost-sensitive learning
- Benchmarks models across six evaluation dimensions
- Secures regulatory transparency through SHAP-based explainability
- Extends into temporal time-to-default survival analysis

---

## Final Performance & Dual-Model Deployment

| Model | Imbalance Strategy | Test Recall | Precision | Accuracy | F1 | ROC-AUC | Brier Score |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Class Weights `{0:1, 1:5}` | **71.83%** | 28.21% | 59.99% | 0.412 | **70.06%** | 0.2397 |
| **XGBoost** | `scale_pos_weight = 5.5` | 71.15% | 27.77% | 59.43% | 0.407 | 69.60% | 0.2356 |
| **Random Forest (Full)** | Class Weights `{0:1, 1:4.7}` | 69.57% | 27.03% | 58.61% | 0.398 | 68.64% | 0.2380 |
| **Random Forest (Reduced)** | Class Weights `{0:1, 1:4.7}` | 72.32% | 26.44% | 56.59% | 0.398 | 68.47% | 0.2386 |
| **Gradient Boosting (GBM)** | Sample Weights `{0:1, 1:4.75}` | 65.97% | **29.04%** | **62.98%** | 0.407 | 69.61% | **0.2212** |

### Strategic Dual-Model Architecture

- **Gradient Boosting — Primary Production Engine.** Selected for deployment on the strength of its industry-leading probability calibration (Brier Score: 0.2212) and statistically confirmed superiority via McNemar's test (p < 0.05), giving trustworthy absolute probabilities for financial provisioning and IFRS 9 ECL calculations.
- **Logistic Regression — Regulatory Interpretability Model.** Retained as a secondary governance engine for its high ROC-AUC (70.06%) and transparent, coefficient-based structure, enabling direct regulatory audits and adverse action reporting.

---

## Detailed Six-Phase Architecture

```
Raw Data (466,285 rows × 75 columns)
│
Phase 1 — Data Architecture & Integrity Protocols
├── Validated payment accounting identity
├── Dropped 20 leak-prone / irrelevant features
├── Excluded 236,239 unresolved loans to eliminate label noise
└── Contextually imputed missing values → 230,288 resolved rows × 55 features
│
Phase 2 — Exploratory Signal Extraction & Feature Strength
├── Mapped a 7x risk escalation across sub_grades (6.5% → 43.7%)
├── Debunked absolute revolving balances (near-identical distributions across classes)
└── Confirmed revolving utilization triples default risk (11% → 30%+)
│
Phase 3 — Methodological Modeling & Feature Engineering
├── Hypothesis testing: Chi-Square, Anderson-Darling, Mann-Whitney U
├── Binned skewed continuous variables (income, interest rate, DTI, utilization)
├── Rejected SMOTE outright (collapsed model accuracy to 19.62%)
└── Implemented native cost-sensitive learning weights
│
Phase 4 — Multi-Dimensional Evaluation & Calibration Diagnostics
├── Threshold optimization via Youden's J & F1 maximization (optimal: 0.53–0.57)
├── Brier Score calibration reliability analysis
├── Out-of-time business validation on 235,399 active loans (risk monotonicity confirmed)
└── Pairwise model significance testing via McNemar's Test (p < 0.05)
│
Phase 5 — Explainable AI (SHAP) Integration
├── TreeExplainer for GBM (sub_grade dominant, |SHAP| = 0.19)
├── LinearExplainer for Logistic Regression
└── Local waterfall analytics for individualized adverse-action auditing (ECOA compliance)
│
Phase 6 — Temporal Risk Dynamics & Survival Analysis
├── Kaplan-Meier non-parametric survival estimation (asymptotic stabilization at ~0.73)
├── Stratified log-rank testing confirming cohort survival variance (p < 0.05)
├── Cox Proportional Hazards modeling (C-index: 0.67; Schoenfeld residual checks)
├── Quantified key hazard ratios (interest rate +70.15%, annual income −19.41%)
└── XGBoost Survival engine for non-linear risk ranking and lifetime ECL horizon mapping
```

---

## Key Methodological Insights

- **The dominance of proprietary grading.** Bivariate exploration showed that Lending Club's `sub_grade` accounts for the vast majority of risk variance, mapping a near-perfect monotonic risk curve from 6.5% (Grade A) to 43.7% (Grade G).
- **Absolute balances vs. ratios.** Absolute monetary figures like `revol_bal` carry little discriminative power due to overlapping distributions, whereas relative ratios like `revol_util` successfully capture financial stress and liquidity exhaustion.
- **The rejection of synthetic oversampling (SMOTE).** Synthetic data generation failed both empirically and theoretically — combining SMOTE with class weights collapsed model accuracy to 19.62% by interpolating implausible credit profiles. Native cost-sensitive weighting preserved real borrower behavior and performed far better.
- **The predictive ceiling.** Across all ensemble and linear algorithms, performance converged at 68–70% ROC-AUC and ~72% Recall, indicating the predictive boundary is set by the latent informational signal in the data rather than algorithmic complexity.
- **Survival dynamics & time-to-default.** Default hazard is heavily front-loaded, peaking acutely between months 5–7 for 36-month loans and months 8–10 for 60-month loans. Kaplan-Meier baseline survival stabilizes asymptotically at ~0.73 over a 70-month tracking period.
- **Hazard risk multipliers.** Cox Proportional Hazards isolated the primary risk escalators: a +70.15% instantaneous default risk for rising interest rates, a +11.60% jump for recent credit inquiries, a +12.21% elevated baseline risk for renters, and a −19.41% protective effect from higher annual income.

---

## Statistical Testing & Validation Matrix

| Statistical Test / Method | Phase | Purpose & Application |
|---|---|---|
| **Chi-Square Test of Independence** | 3 | Tested categorical features against binary `loan_status` (p < 0.05 threshold); eliminated `application_type` (p = 1.0). |
| **Anderson-Darling Test** | 3 | Tested numerical feature normality; rejected normality across all columns (e.g., `annual_inc`: A² = 14,953.29, far above the 0.787 critical value). |
| **Mann-Whitney U Test** | 3 | Non-parametric alternative to the t-test; confirmed significant distributional shifts in DTI and interest rates between defaulters and non-defaulters. |
| **Youden's J Statistic & F1 Maximization** | 4 | Optimized decision thresholds, establishing operational cutoffs between 0.53 and 0.57 to limit capital loss. |
| **Brier Score** | 4 | Evaluated mean squared error of probability outputs; established GBM as the best-calibrated engine (0.2212). |
| **McNemar's Test** | 4 | Run across 10 pairwise model configurations (p < 0.05); confirmed error distributions between models were statistically distinct. |
| **Log-Rank Tests** | 6 | Confirmed statistically significant survival-distribution variance across categorical cohorts (e.g., employment verification status, p < 0.05). |
| **Schoenfeld Residuals** | 6 | Validated the proportional hazards assumption for the Cox PH model, prompting stratification where the assumption failed. |

---

## Repository Structure

```
credit-risk-explainable-ml/
├── README.md
├── Phase1_Data_Cleaning/
│   ├── README.md
│   ├── Data_cleaning.ipynb
│   └── Data_cleaning_report.pdf
├── Phase2_EDA/
│   ├── README.md
│   ├── eda.ipynb
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
│   └── survival_model_evaluation.ipynb
├── .gitignore
└── requirements.txt
```

---

## Tech Stack

- **Language:** Python 3.x
- **Data Processing & Curation:** Pandas, NumPy
- **Statistical Inference:** SciPy, Statsmodels
- **Machine Learning & Ensembles:** Scikit-learn, XGBoost
- **Survival Modeling:** Lifelines
- **Explainable AI:** SHAP (`TreeExplainer`, `LinearExplainer`)
- **Visualization:** Matplotlib, Seaborn

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/credit-risk-explainable-ml.git
cd credit-risk-explainable-ml

# Install required dependencies
pip install -r requirements.txt

# Download the Lending Club dataset from Kaggle and place source files in the working directory
# Execute notebooks sequentially from Phase 1 through Phase 6
```
