# Phase 5 — Explainability (XAI)

## Objective
Apply SHAP (SHapley Additive exPlanations) to both final models
to explain predictions at global (portfolio) and local (individual
borrower) level — making the credit risk model auditable,
interpretable, and regulatory-ready.

---

## Models Explained
| Model | Explainer | Role |
|---|---|---|
| Gradient Boosting | shap.TreeExplainer | Primary deployment model |
| Logistic Regression | shap.LinearExplainer | Interpretability companion |

---

## Why Two Models for XAI?
- **GBM** — best calibrated probabilities (Brier 0.2212), used for
  actual risk scoring. SHAP TreeExplainer gives exact Shapley values
- **LR** — highest F1 (0.412) and ROC-AUC (70.06%), coefficients
  directly readable by regulators. SHAP LinearExplainer aligns
  with coefficient-based explanations

---

## Plots Generated (15 Total)

| # | Plot | Model |
|---|---|---|
| 1 | Bar — Global Importance | GBM |
| 2 | Beeswarm — Direction & Distribution | GBM |
| 3 | Waterfall — Non-Defaulter | GBM |
| 4 | Waterfall — Defaulter | GBM |
| 5 | Dependence — sub_grade | GBM |
| 6 | Dependence — bin_int_rate | GBM |
| 7 | Dependence — term | GBM |
| 8 | Bar — Global Importance | LR |
| 9 | Beeswarm — Direction & Distribution | LR |
| 10 | Waterfall — Non-Defaulter | LR |
| 11 | Waterfall — Defaulter | LR |
| 12 | Dependence — sub_grade | LR |
| 13 | Dependence — bin_int_rate | LR |
| 14 | Dependence — term | LR |
| 15 | Side-by-side comparison | GBM + LR |

---

## Key Findings

### GBM — Global Importance
- **sub_grade (0.19)** dominates — 3× more than any other feature
- **bin_annual_inc, bin_dti, term** all at 0.05-0.06 — second tier
- **bin_int_rate (0.04)** lower than expected — GBM routes the
  correlated rate signal through sub_grade
- **total_acc_bin = 0.00** — completely ignored, confirms EDA

### LR — Global Importance
- No single dominant feature — top 9 all within 0.08-0.21
- **purpose_group_Debt Management (0.21)** ranks 1st — LR captures
  loan purpose signal directly; GBM absorbs it through interactions
- **bin_int_rate (0.18)** near-equal to sub_grade — LR cannot
  separate collinear features, uses both independently

### GBM vs LR Agreement
- sub_grade, bin_annual_inc, bin_int_rate, bin_dti, term appear
  in top 6 of **both** models — genuine signals confirmed
- Where they agree → data truth | Where they differ → model
  architecture difference, not contradictions

### Dependence Plots — All Monotonic
- sub_grade: perfectly monotonic A1 (-0.45) → G5 (+0.25/+0.80)
- bin_int_rate: clean steps from -0.10/<-0.40 to +0.09/+0.60
- term: hard binary — 36m negative SHAP, 60m positive SHAP

### Individual Predictions (same borrower, both models)
| | GBM | LR |
|---|---|---|
| Non-Defaulter probability | 44% | 36% |
| Defaulter probability | 56% | 67% |
| LR vs GBM | More conservative on safe loans | More aggressive on risky loans |

---

## Files in This Phase

| File | Description |
|---|---|
| `shap_analysis.ipynb` | Full SHAP analysis — both models, all 15 plots |
