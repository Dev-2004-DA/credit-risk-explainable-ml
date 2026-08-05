# Phase 6 — Survival Analysis & Model Evaluation

## Objective
Answer the second core research question — **When will a borrower default?**
While Phase 3-5 addressed *who* will default using classification, this phase
uses survival analysis to model *time-to-default* within the loan lifecycle.
Additionally, a unified prediction system is built combining all trained models
into a single borrower-level inference pipeline with a Streamlit dashboard.

---

## Two Notebooks in This Phase

| Notebook | Purpose |
|---|---|
| `survival_analysis.ipynb` | Kaplan-Meier, Log-Rank, Cox PH, XGBoost Survival |
| `model_evaluation.ipynb` | SHAP on survival model, unified prediction function, Streamlit app |

---

## What is Survival Analysis?

Unlike classification which predicts a binary outcome (default / no default),
survival analysis models **time-to-event** — how long until a borrower defaults.
It handles **censored observations** — borrowers who did not default within the
observation window (Fully Paid loans) are not missing data, they are censored
at their repayment duration.

| Concept | Definition |
|---|---|
| Duration | Months from loan issue to last payment |
| Event | 1 = Defaulted, 0 = Fully Paid (censored) |
| Survival Function S(t) | Probability of surviving beyond time t without default |
| Hazard Function h(t) | Instantaneous default rate at time t given survival to t |

---

## Models Applied

### 1. Kaplan-Meier Estimator (Non-Parametric)
- Estimates survival curve without any distributional assumptions
- Applied overall and by risk groups: grade, term, sub_grade
- **Log-Rank Test** used to confirm survival curves differ
  significantly between groups (p < 0.05)

**Key findings:**
- Grade A borrowers survive significantly longer than Grade G
- 36-month loans show higher early survival than 60-month loans
- Survival probability drops sharply in months 5-10 — consistent
  with Phase 2 EDA finding that default peaks in months 5-10

### 2. Cox Proportional Hazards Model (Semi-Parametric)
- Models hazard ratio — how much each feature accelerates or
  decelerates time to default relative to a baseline
- Features with hazard ratio > 1 increase default risk
- Features with hazard ratio < 1 are protective

**Key findings:**
- int_rate — strongest predictor of accelerated default
- sub_grade — confirmed hazard increases monotonically A→G
- annual_inc — protective, higher income delays default
- term — 60-month loans show higher hazard than 36-month

### 3. XGBoost Survival Model
- Gradient boosting with survival objective (Cox loss)
- Captures non-linear interactions missed by Cox PH
- Outputs relative risk score — higher score = earlier expected default

**Feature Importance (XGBoost Survival):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | int_rate | 0.3212 |
| 2 | sub_grade | 0.1039 |
| 3 | inq_last_6mths | 0.0625 |
| 4 | purpose_group_Business_Education | ~0.05 |

---

## SHAP on XGBoost Survival Model

SHAP TreeExplainer applied to explain individual survival predictions:

**Bar Plot (Mean |SHAP|):**
- int_rate (0.42) — highest contribution, dominant survival predictor
- annual_inc (0.19) — second, higher income delays default
- inq_last_6mths (0.10) — recent inquiries accelerate default
- loan_amnt (0.07) — larger loans have higher hazard

**Beeswarm Plot:**
- High interest rate (red) → positive SHAP → earlier default
- High annual income (red) → negative SHAP → delayed default
- More credit inquiries → positive SHAP → higher hazard

**Key difference from Phase 5 classification SHAP:**
- int_rate dominates survival SHAP (0.42) vs sub_grade
  dominating classification SHAP (0.19)
- Survival model is more sensitive to the continuous rate signal
  while classification model routes it through sub_grade

---

## Unified Prediction Function

A single prediction function was built combining all trained models
across both classification (Phase 3-5) and survival (Phase 6):

```python
# Input: borrower profile (raw features)
# Output: all model predictions in one call

LR model:          default probability (%)
GBM model:         default probability (%)
Cox model:         risk score (hazard ratio)
XGBoost Survival:  risk score (relative ranking)
```

**Sample Borrower Output (C3, 60-month, $25,000, income $75,000):**

| Model | Prediction | Interpretation |
|---|---|---|
| Logistic Regression | 80.17% | High default probability |
| Gradient Boosting | 61.48% | High default probability |
| Cox Risk Score | 1.03 | Above baseline hazard |
| XGBoost Survival Score | 1.28 | Higher relative default risk |

All four models agree — this borrower is high risk.

---

## Streamlit Dashboard

A live credit risk dashboard was built using Streamlit combining
all models into a single interface:

- Input borrower profile through interactive form
- Returns LR and GBM default probabilities
- Returns Cox and XGBoost survival risk scores
- Displays SHAP waterfall plot for individual explanation
- Shows survival curve for the borrower's risk profile

---

## Connection to Previous Phases

| Finding | Phase 2 EDA | Phase 6 Survival |
|---|---|---|
| Default peaks months 5-10 | ✅ Observed visually | ✅ Confirmed by KM curves |
| Grade A safest, Grade G riskiest | ✅ 6.5% vs 43.7% default rate | ✅ KM curves clearly separated |
| int_rate strong predictor | ✅ Ranked ★★★★★ | ✅ Top SHAP feature (0.42) |
| 60-month riskier than 36-month | ✅ 32% vs 16% default rate | ✅ Higher hazard confirmed by Cox |
| Higher income = lower risk | ✅ 24% → 13% gradient | ✅ Protective in Cox + SHAP |

---

## Key Conclusion

Survival analysis answers what classification cannot — not just whether a
borrower will default, but when. The danger zone identified in Phase 2 EDA
(months 5-10) is formally confirmed by Kaplan-Meier curves. Cox PH and
XGBoost survival models both agree that interest rate, sub_grade, income,
and term are the dominant time-to-default predictors — fully consistent
with all previous phases.

---

## Files in This Phase

| File | Description |
|---|---|
| `survival_analysis.ipynb` | KM, Log-Rank, Cox PH, XGBoost Survival |
| `model_evaluation.ipynb` | SHAP on survival model, unified prediction function, Streamlit dashboard |
