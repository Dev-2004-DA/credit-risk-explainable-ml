# Phase 3 — Modelling

## Objective
Build and tune binary classification models to predict loan default
(Who will default?). Evaluate on Recall as the primary metric —
missing a real defaulter costs the bank far more than a false alarm.

---

## Two Notebooks in This Phase

| Notebook | Purpose |
|---|---|
| `modelling_preprocessing.ipynb` | Feature selection, engineering, binning, encoding |
| `modelling_2.ipynb` | Model training and hyperparameter tuning |

---

## Part 1 — Preprocessing (`modelling_preprocessing.ipynb`)

### Statistical Feature Selection

**For categorical features — Chi-Square Test**
- H₀: Feature is independent of loan_status
- H₁: Feature is associated with loan_status
- Decision rule: p-value < 0.05 → keep

Results:
- All ordinal features (term, grade, sub_grade, emp_length) → **Reject H₀ — keep**
- Nominal features: home_ownership, verification_status, purpose_group → **Reject H₀ — keep**
- `application_type` → **Fail to reject H₀ (p=1.0) — dropped**

**For numerical features — Anderson-Darling + Mann-Whitney**
- Anderson-Darling first to check normality
- All 7 numerical columns rejected normality → Mann-Whitney used
- All 7 columns (loan_amnt, int_rate, annual_inc, dti, revol_bal,
  revol_util, installment) showed significant association → keep
- `installment` dropped after correlation check (r=0.96 with loan_amnt)

### Feature Engineering

**Binary flags created:**
- `delinq_2yrs_flag` — 0/1 binary from delinq_2yrs count
- `emp_status_unverifiable` — 0 = employment stated, 1 = UNKNOWN
- `mths_since_last_delinq_flag` — 0/1
- `mths_since_last_record_flag` — 0/1
- `pub_rec_flag` — 0/1

**Binned continuous variables (4 schemes tested per feature):**

| Feature | Final Bins | Reason |
|---|---|---|
| `annual_inc` | <18K, 18K-40K, 40K-80K, 80K-120K, >120K | Monotonic default rate, natural breakpoints |
| `int_rate` | <8%, 8-12%, 12-15%, 15-21%, 21-24%, >24% | Aligns with sub-grade rate tiers |
| `dti` | <5, 5-10, 10-15, 15-20, 20-25, 25-30, 30-35, >35 | Most granular monotonic pattern |
| `revol_util` | <20%, 20-40%, 40-60%, 60-80%, 80-100%, >100% | Natural equal-width breaks, monotonic |

Binning rationale: converts noisy continuous values into statistically
validated risk tiers — improves model interpretability and robustness.

**Purpose grouping:** 13 categories → 5 groups
- Debt Management, Home & Property, Personal Expenses,
  Business & Education, Vehicle & Major Purchases

**Features dropped from modelling:**
- `grade` — redundant with sub_grade
- `mths_since_last_delinq_flag` — <1pp default rate difference
- `mths_since_last_record_flag` — <1pp default rate difference
- `pub_rec_flag` — <1pp default rate difference
- `open_acc_bin` — essentially flat default rate

### Final Feature Set (15 columns)

| Type | Features |
|---|---|
| Ordinal (8) | term, sub_grade, inq_last_6mths_bin, total_acc_bin, bin_annual_inc, bin_int_rate, bin_dti, bin_revol_util |
| Nominal (3) | home_ownership, verification_status, purpose_group |
| Binary (2) | delinq_2yrs_flag, emp_status_unverifiable |
| Numerical (1) | loan_amnt |
| Target | loan_status |

**Output:** `modelling_ready_data.csv` — 230,288 rows × 15 columns

---

## Part 2 — Model Training (`modelling_2.ipynb`)

### Pipeline Setup
```
OrdinalEncoder → OneHotEncoder → StandardScaler → SelectKBest → Model
```
- `StratifiedKFold (n_splits=3)` used throughout — preserves 81/19
  class ratio in every fold
- Recall as primary refit metric — catching defaulters is the priority
- Scored on: Recall, Precision, Accuracy, ROC-AUC, F-beta (β=1.5)

### Class Imbalance Strategy
- **LR & RF:** `class_weight` parameter
- **XGBoost:** `scale_pos_weight` (natural ratio = 4.27)
- **GBM:** `compute_sample_weight` passed to `fit()`
- **No SMOTE** — class weights are cleaner and avoid the probability
  collapse problem (99% recall with meaningless precision)

---

## Models Trained

### 1. Logistic Regression

| Round | Type | Best Params | Recall | Accuracy | Precision |
|---|---|---|---|---|---|
| RCV1 | RandomizedSearchCV | weight={0:1,1:5}, l1=0.1 | 71.90% | 60.15% | 28.32% |
| GCV1 | GridSearchCV | weight={0:1,1:5}, l1=0.1 | 71.90% | 60.15% | 28.32% |

**Conclusion:** All 9 GCV1 combinations produced identical recall —
model converged. Linear decision boundary insufficient for non-linear
credit risk patterns. Moved to tree-based models.

---

### 2. XGBoost

| Round | Key Change | Recall | Accuracy | AUC |
|---|---|---|---|---|
| RCV1 | Wide search — n_estimators typo fixed | 73.35% | — | — |
| GCV1 | scale_pos_weight {3,4.27,5,7}, max_depth {9} | 67.00% | 59.78% | 67.52% |
| GCV2 | max_depth {7,9,13}, n_estimators {200,300,500} | 81.72% | 49.35% | 68.53% |
| GCV3 | weight {4.27,5,5.5}, max_depth {3,5,7} | 76.25% | 54.08% | 68.28% |
| GCV4 | subsample {0.6,0.8,1.0}, colsample_bytree added | 76.78% | 53.26% | 67.60% |
| GCV5 | select__k=15 (increased from 7) | 76.49% | 55.66% | **69.76%** |
| GCV6 | refit=roc_auc, colsample=0.5, subsample=0.7 | 76.40% | 55.66% | 69.67% |

**Final XGBoost model (GCV5):**
```python
XGBClassifier(
    learning_rate    = 0.1,
    max_depth        = 3,
    n_estimators     = 200,
    scale_pos_weight = 5.5,
    subsample        = 0.8,
    colsample_bytree = 0.6
)
```

**Key insight:** select__k=15 (vs k=7) was the missing piece for AUC —
more features improve probability calibration.

---

### 3. Random Forest

| Round | Key Change | Recall | Accuracy |
|---|---|---|---|
| RCV1 | Wide search — 15 iterations | 74.17% | 56.44% |
| GCV1 | max_depth {2,3}, n_estimators {300,400,500} | 74.39% | 56.28% |
| GCV2 | weight {4.27,4.5,4.7,4.9}, min_samples_leaf {10,20,30} | 73.20% | 57.06% |

**Key findings:**
- `min_samples_leaf` had zero impact at max_depth=3
- `class_weight` is the only meaningful parameter
- Two versions trained: 14 features (dropped XGB zero-importance) vs
  full 20 features — reduced version gave higher recall

**Final RF model (manually selected from GCV2):**
```python
RandomForestClassifier(
    class_weight      = {0:1, 1:4.7},
    max_depth         = 3,
    max_features      = 11,
    min_samples_leaf  = 10,
    min_samples_split = 51,
    n_estimators      = 300
)
```

---

### 4. Gradient Boosting

| Round | Key Change | Recall | Accuracy | AUC |
|---|---|---|---|---|
| RCV1 | Wide search — max_depth=9, lr=0.001 won | 85.46% CV | 44.24% test | 68.89% |
| GCV1 | max_depth {3,5}, lr {0.01,0.1}, subsample=0.8 | 71.98% | 64.67% | 69.61% |
| GCV2 | weight reduced to 4.27 | 66.51% | 64.47% | — |

**Overfitting detected in RCV1:** max_depth=9 + lr=0.001 gave 85.46%
CV recall but test precision collapsed from 59.92% to 23.63%. Classic
deep tree memorisation pattern.

**Key finding — CV precision inflation:** `sample_weight` inflates
cross-validation precision to 60-65% but test precision always returns
to ~27-28%. This is a known sklearn limitation with sample_weight
scoring in cross-validation.

**Final GBM model (GCV1):**
```python
GradientBoostingClassifier(
    n_estimators     = 300,
    learning_rate    = 0.01,
    loss             = 'exponential',
    max_depth        = 3,
    subsample        = 0.8,
    min_samples_leaf = 20
)
sample_weight = compute_sample_weight(
    class_weight = {0:1, 1:4.75},
    y = y_train
)
```

---

## Saved Model Files

| File | Model |
|---|---|
| `Final_lr_model` | Logistic Regression |
| `Final_xgb_model` | XGBoost |
| `Final_rf_model` | Random Forest (14 features) |
| `Final_Rf_model_trained_on_full_features` | Random Forest (20 features) |
| `Final_gb_model` | Gradient Boosting |
| `Ct_for_preprocessing` | ColumnTransformer — preprocessing pipeline |

---

## Files in This Phase

| File | Description |
|---|---|
| `modelling_preprocessing.ipynb` | Feature selection, engineering, binning |
| `modelling_2.ipynb` | All 4 models — training and hyperparameter tuning |
