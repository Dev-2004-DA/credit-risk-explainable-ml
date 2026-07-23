# Phase 2 — Exploratory Data Analysis

## Objective
Understand the structure of the cleaned dataset, identify which features
are associated with default, rank their predictive strength, and guide
feature engineering decisions for modelling.

---

## Dataset
- **Input:** `data_for_eda.csv` (cleaned output from Phase 1)
- **Records:** 230,288 resolved loans
- **Target:** `loan_status` — Defaulter (19%) vs Not Defaulter (81%)

---

## Analysis Structure

Each feature was analysed through:
- Distribution plots (histogram, KDE, boxplot, countplot)
- Default rate comparison between groups
- Correlation with other features where relevant
- Interpretive observations documented in markdown cells

---

## Feature-by-Feature Findings

### Loan Amount
- Right-skewed distribution with peaks at round amounts (5K, 10K, 15K, 20K, 35K)
- Modest positive association with default — higher loan amounts carry
  marginally higher risk (16.7% at <10K vs 24.4% at 30K+)
- Weak standalone predictor due to substantial overlap between groups

### Funded Amount & Investor Funded Amount
- `funded_amnt` correlates at 0.997 with `loan_amnt` — nearly redundant
- 19.5% of loans (44,934) have a gap between funded amount and investor
  funded amount — gap widens for larger loans
- `inv_funding_gap_pct` derived as a potential engineered feature

### Term
- 78.6% of loans are 36-month, 21.4% are 60-month
- **Strong predictor** — 60-month loans default at ~32% vs ~16% for
  36-month loans — nearly 2× difference

### Interest Rate
- Range: 5.42% to 26.06%, mean 13.78%
- **Strongest continuous predictor** — non-defaulters concentrate at
  8–13%, defaulters shift to 14–20%+ with clear visual separation
- Discrete peaks in distribution confirm sub-grade rate tiers

### Installment
- Highly correlated with funded amount (r=0.96) — driven by loan size
- Only modest difference between defaulters and non-defaulters
  (mean $438 vs $410 — ~7% gap)
- Dropped in modelling phase due to multicollinearity with loan amount

### Grade & Sub-Grade
- **Strongest predictor in entire dataset**
- Grade A defaults at 6.5%, Grade G at 43.7% — **7× increase**
- Perfectly monotonic across all 7 grades and all 35 sub-grades
- Interest rate is set by grade — both encode the same risk signal
  in different forms

### Employment Length
- **No meaningful pattern** — default rate flat at 18–20% across all
  categories from < 1 year to 10+ years
- UNKNOWN category (27.1%) is the only elevated group
- Confirmed as a weak predictor — collapsed to binary flag in modelling

### Home Ownership
- Mortgage holders default least (17.09%), renters most (20.99%)
- Moderate signal — 3-4 percentage point range across categories
- OTHER, NONE, ANY merged into UNSPECIFIED (~0.1% of data)

### Annual Income
- Highly right-skewed — 95% of borrowers earn under $150K
- Log transformation applied for better visualisation
- Clear negative gradient — higher income = lower default risk
  (24.05% at <$40K vs 13.16% at >$120K)

### Loan Purpose
- 13 original categories grouped into 5:
  - Debt Management, Home & Property, Personal Expenses,
    Business & Education, Vehicle & Major Purchases
- Moderate signal — Business & Education highest default (28.7%),
  Vehicle & Major Purchases lowest (13.6%)

### DTI (Debt-to-Income Ratio)
- Defaulters consistently show higher DTI across all purpose groups
  (mean 18.19% vs 15.92% for non-defaulters)
- Z-test confirmed statistically significant difference (p < 0.001)
  across all purpose subgroups
- Low correlation with other features (max r=0.31) — provides
  independent signal

### Delinquencies (delinq_2yrs)
- 84.2% of borrowers have zero delinquencies
- Binary flag approach confirmed — 20.5% default with any delinquency
  vs 18.7% without — only 1.8pp difference
- Weak predictor but retained as binary flag

### Revolving Balance
- **No discriminative power** — defaulters and non-defaulters have
  nearly identical distributions (mean $15,170 vs $15,244)
- Log transformation does not help — distributions remain overlapping
- Dropped from modelling; `revol_util` used instead

### Revolving Utilization
- Clear monotonic signal — 11.26% default at 0–20% utilization,
  rising to 30.3% at 100%+
- Pre-loan characteristic from credit bureau — no data leakage risk
- Weak correlation with revol_bal (r=0.22) — captures different info
- One extreme outlier at 892.3% identified — data quality issue

---

## Feature Predictive Strength Summary

| Feature | Default Rate Range | Strength |
|---|---|---|
| grade / sub_grade | 6.5% – 43.7% | ⭐⭐⭐⭐⭐ Strongest |
| int_rate | 8–13% vs 14–20%+ | ⭐⭐⭐⭐⭐ Very Strong |
| term | ~16% vs ~32% | ⭐⭐⭐⭐ Strong |
| revol_util | 11.3% – 30.3% | ⭐⭐⭐ Moderate-Strong |
| annual_inc | 24.1% → 13.2% | ⭐⭐⭐ Moderate |
| dti | Higher for defaulters | ⭐⭐⭐ Moderate |
| purpose_group | 13.6% – 28.7% | ⭐⭐ Moderate-Weak |
| home_ownership | 17.1% – 21.0% | ⭐⭐ Weak-Moderate |
| delinq_2yrs | 18.7% vs 20.5% | ⭐ Weak |
| revol_bal | ~Equal both groups | ✗ None |
| emp_length | 18–20% flat | ✗ None |

---

## Time-Based Analysis

### Loan Issuance Trend
- Strong growth in loan issuances from 2007 to 2014
- Acceleration from 2012 onward — platform expansion period
- Default proportion did not increase with volume — underwriting
  quality remained consistent across the expansion

### Default Risk by Tenure
- **36-month loans:** Default risk peaks in months 5–7 then declines
- **60-month loans:** Default risk peaks in months 8–10 then declines
- A borrower who survives the first year is significantly more likely
  to fully repay — key insight motivating survival analysis

---

## Key EDA Conclusions

1. Grade and interest rate together are the dominant default signals —
   both encode Lending Club's own risk assessment
2. Revolving balance has zero discriminative power — utilization rate
   does not (three times the default rate from low to high utilization)
3. Employment length is useless as a predictor — only UNKNOWN status
   carries meaningful signal
4. Default risk is front-loaded in the loan lifecycle — months 5–10
   are the danger zone for both loan terms
5. The 2007–2014 window spanning the 2008 financial crisis provides
   ideal conditions for macro stress testing

---

## Files in This Phase

| File | Description |
|---|---|
| `eda.ipynb` | Main EDA notebook — all feature analyses with plots |
| `EDA_Report.pdf` | Compiled PDF report of EDA findings |
| `EDA_Report_With_Visuals.docx` | Word document with embedded notebook plots |
