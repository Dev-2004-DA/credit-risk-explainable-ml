# Phase 4 — Model Evaluation

## Objective
Comprehensively evaluate all five trained models beyond standard test
metrics — using calibration analysis, threshold optimization, agreement
analysis on active loans, and statistical significance testing to
select the final deployment model.

---

## Models Evaluated
- Logistic Regression
- XGBoost
- Random Forest (Full — 20 features)
- Random Forest (Reduced — 14 features)
- Gradient Boosting

---

## Evaluation 1 — Standard Test Metrics

| Model | Recall | Precision | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 71.83% | 28.21% | 59.99% | 0.412 | **70.06%** |
| XGBoost | 71.15% | 27.77% | 59.43% | 0.407 | 69.60% |
| RF Full | 69.57% | 27.03% | 58.61% | 0.398 | 68.64% |
| RF Reduced | **72.32%** | 26.44% | 56.59% | 0.398 | 68.47% |
| Gradient Boosting | 65.97% | **29.04%** | **62.98%** | 0.407 | 69.61% |

**Key observation:** All models converged to the same performance range
— confirming the ceiling is in the data signal, not the algorithms.

---

## Evaluation 2 — Agreement Analysis on Active Loans

Applied all models to **235,399 unresolved loans** (Current, In Grace
Period, Late 16-30, Late 31-120) at threshold 0.7.

### Monotonicity Check — Mean Predicted Probability

| Loan Status | XGB | GBM | LR | RF Full | RF Reduced |
|---|---|---|---|---|---|
| Current | 0.54 | 0.52 | 0.61 | 0.51 | 0.50 |
| In Grace Period | 0.60 | 0.58 | 0.67 | 0.57 | 0.57 |
| Late (16–30 days) | 0.60 | 0.58 | 0.67 | 0.57 | 0.57 |
| Late (31–120 days) | 0.61 | 0.59 | 0.68 | 0.57 | 0.57 |

**Monotonicity confirmed across all models** — default probability
increases consistently as payment delay worsens. All models learned
genuine patterns, not training data noise.

### Predicted Defaults at Threshold 0.7

| Model | Flagged as Default | % of Portfolio |
|---|---|---|
| Logistic Regression | 81,248 | 34.5% |
| XGBoost | 33,656 | 14.3% |
| Gradient Boosting | 27,979 | 11.9% |
| RF Reduced | 26,388 | 11.2% |
| RF Full | 22,407 | 9.5% |

LR is most aggressive — flags 34.5% at threshold 0.7. RF Full most
conservative at 9.5%. XGBoost and GBM provide balanced middle ground.

---

## Evaluation 3 — Threshold Optimization (Precision-Recall Curve)

Optimal threshold found per model by maximising F1 score:

| Model | Best Threshold | F1 |
|---|---|---|
| **Logistic Regression** | 0.546 | **0.412** |
| XGBoost | 0.566 | 0.407 |
| Gradient Boosting | 0.537 | 0.407 |
| RF Full | 0.552 | 0.398 |
| RF Reduced | 0.565 | 0.398 |

All optimal thresholds cluster at 0.53–0.57 — confirms 0.5 (default)
and 0.7 (strict) are both suboptimal operating points.

---

## Evaluation 4 — Calibration Analysis (Brier Score)

**What Brier Score measures:** Average squared error between predicted
probability and actual outcome. Lower = better calibrated probabilities.

| Model | Brier Score | Calibration Quality |
|---|---|---|
| **Gradient Boosting** | **0.2212** | Best |
| XGBoost | 0.2356 | Good |
| RF Full | 0.2380 | Acceptable |
| RF Reduced | 0.2386 | Acceptable |
| Logistic Regression | 0.2397 | Acceptable |

**GBM produces the most trustworthy probability estimates** — when it
assigns a risk score, that score most accurately reflects true default
likelihood. Critical for downstream decisions: loan pricing, capital
reserve calculations, portfolio risk monitoring.

**Calibration curve observation:** All models sit below the diagonal
at higher predicted probabilities — overestimating risk at the upper
end. This is a common pattern in imbalanced credit datasets.

---

## Evaluation 5 — Statistical Significance (McNemar's Test)

Pairwise McNemar's tests confirmed all differences between models are
statistically significant.

**Final ranking:**
```
GBM > LR > XGBoost > RF Full > RF Reduced
```

---

## Final Model Selection — Gradient Boosting

Although Logistic Regression achieved the highest F1 (0.412) and
ROC-AUC (70.06%), the improvement over GBM was marginal and came at
the cost of weaker calibration (0.2397 vs 0.2212).

**Gradient Boosting selected as primary model because:**
- Best probability calibration (Brier 0.2212) — most trustworthy
  risk scores for downstream decisions
- Monotonicity validated on 235,399 active loans
- Statistically superior on McNemar's test
- Best precision (29.04%) and accuracy (62.98%) on test set
- Stable threshold behaviour (optimal threshold 0.537 — closest
  to natural 0.5 of all models)

**Logistic Regression retained as interpretability model:**
- Highest F1 (0.412) and ROC-AUC (70.06%)
- Coefficients directly explainable to regulators
- Used for XAI analysis in Phase 5

**Random Forest (both versions) — eliminated:**
- Weakest F1 and calibration across all evaluations
- RF Reduced oversimplification confirmed — sub_grade alone at 73.9%
  importance with 9 out of 14 features at zero importance

---

## Key Finding

> All five models from three different ML paradigms (linear, bagging,
> boosting) converged to the same performance ceiling — 70-72% recall
> and 68-70% AUC. This confirms the ceiling is in the **data signal**,
> not the algorithms. grade, sub_grade and int_rate dominate every
> model's feature importance — Lending Club's own risk assessment is
> the primary driver of default prediction.

---

## Files in This Phase

| File | Description |
|---|---|
| `final_model_evaluation.ipynb` | All 5 evaluation dimensions |
| `final_testing_on_unlabelled_data/Testing_data_created.ipynb` | Agreement analysis on 235K active loans |
| `Evaluation_Ready_data.csv` | Preprocessed test set used for evaluation |
| `final_testing_on_unlabelled_data/data_for_testing.csv` | Active loan portfolio for agreement analysis |
