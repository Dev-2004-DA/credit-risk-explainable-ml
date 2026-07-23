# Phase 5 — Explainability AI (XAI)

## Status: In Progress

This phase applies SHAP (SHapley Additive exPlanations) to the final
selected Gradient Boosting model to explain individual and global
predictions.

---

## Objective

Credit risk models deployed in regulated banking environments must be
explainable — not just accurate. This phase answers:

- **Which features drive the model's predictions globally?**
- **Why did the model flag a specific borrower as a defaulter?**
- **How does each feature push the prediction toward or away from default?**

---

## Why SHAP?

SHAP is grounded in cooperative game theory — it assigns each feature
a contribution value (Shapley value) representing its marginal impact
on the prediction. Unlike permutation importance or coefficient-based
explanations, SHAP:

- Is **model-agnostic** — works on any black-box model
- Provides **both global and local** explanations
- Guarantees **consistency** — a feature that matters more always gets
  a higher SHAP value
- Is **additive** — individual SHAP values sum to the final prediction

---

## Planned Analysis

### Global Explanations
- **SHAP Summary Plot** — feature importance ranked by mean |SHAP|
  across all predictions
- **SHAP Bar Plot** — compact view of top features by average impact
- **SHAP Beeswarm Plot** — distribution of SHAP values per feature,
  coloured by feature value (high/low)

### Local Explanations
- **SHAP Waterfall Plot** — for individual loans showing how each
  feature pushed the prediction above or below the baseline
- **SHAP Force Plot** — visual explanation for a single prediction

### Feature Interaction Analysis
- **SHAP Dependence Plots** — how SHAP value of one feature changes
  across its value range, with interaction colouring
- Focus on: sub_grade × int_rate, bin_dti × bin_annual_inc

---

## Expected Findings (Based on Modelling Phase)

From feature importance analysis across all models, SHAP is expected
to confirm:

| Feature | Expected SHAP Contribution |
|---|---|
| sub_grade | Highest — 60-74% importance across tree models |
| bin_int_rate | Second — 10-35% depending on model |
| term | Third — 6-12% |
| bin_annual_inc | Moderate — 5-10% |
| bin_dti | Moderate — 3-7% |
| emp_status_unverifiable | Small but consistent |
| bin_revol_util | Small — grade captures most of this signal |

---

## Business Value of XAI

For a bank using this model in production:

- **Regulatory compliance** — Basel III and IFRS 9 require explainable
  credit decisions. SHAP outputs can be used directly in loan rejection
  letters and audit documentation
- **Loan officer training** — SHAP waterfall plots show exactly which
  borrower characteristics triggered the risk flag
- **Model monitoring** — drift in SHAP values over time signals when
  the model needs retraining
- **Bias detection** — SHAP values for protected attributes (e.g.
  home_ownership categories) can be inspected for fairness concerns

---

## Files in This Phase

| File | Description |
|---|---|
| `xai_shap.ipynb` | SHAP analysis notebook — coming soon |

---

*This README will be updated with full results once the XAI phase
is complete.*
