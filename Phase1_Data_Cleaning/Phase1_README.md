# Phase 1 — Data Cleaning

## Objective
Clean and prepare the raw Lending Club loan dataset (2007–2014) for
downstream analysis. This phase handles irrelevant columns, missing
values, and target variable definition before any modelling begins.

---

## Dataset
- **Source:** Lending Club loan dataset 2007–2014
- **Raw shape:** 466,285 rows × 75 columns
- **Output:** `Cleaned_lending_club_2007_2014.csv`

---

## Target Variable — `loan_status`

The raw dataset contains 9 loan status categories. For binary
classification only resolved loans are kept:

| Status | Label |
|---|---|
| Fully Paid | 0 — Not Defaulter |
| Charged Off | 1 — Defaulter |

Removed from modelling (unresolved outcomes):
- Current, Late (16-30 days), Late (31-120 days), In Grace Period,
  Default, Does not meet credit policy statuses

---

## Accounting Identity Verified

Before cleaning, the following identity was confirmed to hold across
all records — validating data integrity:

```
total_pymnt = total_rec_prncp + total_rec_int
            + total_rec_late_fee + recoveries
```

---

## Step 1 — Irrelevant Columns Dropped (18 columns)

Columns dropped immediately on inspection:

| Column | Reason |
|---|---|
| `Unnamed: 0` | Row index artifact |
| `url` | Just a web link to borrower profile |
| `desc` | Unstructured free text — 73% missing, no NLP in scope |
| `annual_inc_joint`, `dti_joint`, `verification_status_joint` | All null — joint applications not present |
| `open_acc_6m`, `open_il_6m`, `open_il_12m`, `open_il_24m` | All null — instalment loan fields |
| `mths_since_rcnt_il`, `total_bal_il`, `il_util` | All null |
| `open_rv_12m`, `open_rv_24m`, `max_bal_bc` | All null |
| `all_util`, `inq_fi`, `total_cu_tl`, `inq_last_12m` | All null |

**Shape after drop:** 466,285 × 57

---

## Step 2 — Missing Value Treatment (20 columns)

Each column was individually analysed using a custom `missing_values()`
function that reports count, percentage, and sample values before
deciding on treatment strategy.

| Column | Missing | % | Strategy | Reason |
|---|---|---|---|---|
| `emp_title` | 27,588 | 5.9% | Fill → `UNKNOWN` | 205K unique values — cannot categorise |
| `emp_length` | 21,008 | 4.5% | Fill → `UNKNOWN` for those with unknown emp_title; drop remaining 186 rows | 20,822 of missing emp_length belong to UNKNOWN emp_title — structural missingness |
| `annual_inc` | 4 | 0.001% | Fill → median (63,000) | Only 4 rows — median safe for right-skewed income |
| `title` | 21 | 0.004% | Fill → mode | Negligible count |
| `delinq_2yrs` | 29 | 0.006% | Fill → median | Right-skewed count variable |
| `earliest_cr_line` | 29 | 0.006% | Fill → `issue_d` | No prior credit history — current loan is earliest known account |
| `inq_last_6mths` | 29 | 0.006% | Fill → median | Right-skewed count variable |
| `mths_since_last_delinq` | 250,223 | 53.7% | Fill → 0 | Missing = no delinquency ever occurred |
| `mths_since_last_record` | 403,490 | 86.6% | Fill → 0 | Missing = no public record ever reported |
| `open_acc` | 29 | 0.006% | Fill → median | Right-skewed distribution |
| `pub_rec` | 29 | 0.006% | Fill → 0 | Missing = no derogatory public records |
| `revol_util` | 339 | 0.07% | Computed from `revol_bal / total_rev_hi_lim × 100` where limit available; fill 0 where `revol_bal = 0`; drop remaining 6 rows | Mathematical derivation preferred over imputation |
| `total_acc` | 29 | 0.006% | Fill → `open_acc` | Verified that total_acc = open_acc is valid for new borrowers |
| `last_pymnt_d` | 376 | 0.08% | Impute from `next_pymnt_d` (12 rows); drop remaining | Structural missing for resolved loans |
| `next_pymnt_d` | 226,733 | 48.7% | Retained as NaN | Structural — Fully Paid and Charged Off loans have no next payment |
| `last_credit_pull_d` | 42 | 0.009% | Drop rows | Official bank record — missing suggests data integrity issue |
| `collections_12_mths_ex_med` | 143 | 0.03% | Fill → 0 | All missing belong to resolved loans — no collections expected |
| `mths_since_last_major_derog` | 366,803 | 78.8% | Fill → 0 | Missing = no major derogatory event ever occurred |
| `acc_now_delinq` | 28 | 0.006% | Fill → 0 | Missing = no current delinquency |
| `tot_coll_amt` | 70,085 | 15.1% | Retained / exported as-is | Not used in downstream modelling features |

---

## Key Decisions

**`emp_length` — UNKNOWN category created**
Analysis revealed 20,822 out of 21,008 missing `emp_length` values
belonged to borrowers who also had no `emp_title`. These were labelled
`UNKNOWN` — a deliberate category representing unverifiable employment
status. The remaining 186 rows were dropped as they were negligible
(0.04%).

**`revol_util` — Computed not imputed**
For 66 borrowers with missing revolving utilization but known revolving
balance and credit limit, the value was computed directly:
`revol_util = revol_bal × 100 / total_rev_hi_lim`
This preserves accuracy over median imputation.

**`mths_since_last_delinq` and `mths_since_last_record` — Zero fill**
Both columns use 0 to encode "this event never happened" — a meaningful
value rather than a placeholder. Imputing with mean or median would
incorrectly imply a delinquency or public record occurred.

---

## Output

**File:** `Cleaned_lending_club_2007_2014.csv`

All missing values resolved. Data ready for EDA and feature selection.

---

## Files in This Phase

| File | Description |
|---|---|
| `Data_cleaning.ipynb` | Main cleaning notebook — all steps documented |
| `data_cleaning_report.tex` | LaTeX source of the formal cleaning report |
| `Data_cleaning_report.pdf` | Compiled PDF report of all cleaning decisions |
