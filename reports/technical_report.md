# Technical Report: Credit Default Risk (Part 1)

*Churn and segmentation sections will be added once `Churn_Modelling.csv` has been run through `src/train_churn.py`.*

## 1. Data
Give Me Some Credit: 150,000 applicants, 10 features, target `SeriousDlqin2yrs` (serious delinquency within 2 years). After removing 767 duplicate rows: 149,233 rows, 6.70% positive. Supplied file was already imputed (see `data/README.md`).

## 2. Cleaning (`src/data_cleaning.py`)
- Past-due counts of 96/98 are placeholder codes, set to missing (0.14% of rows).
- Ages below 18 set to missing.
- Utilization and debt ratio capped at 5 (flags `utilization_extreme`, `debt_ratio_extreme` keep the information); income capped at 50,000.
- Only fixed domain rules are used, so no statistics leak from test data. Remaining missing values are median-imputed inside the model pipeline, fitted on training data only.

## 3. EDA findings
| Driver | Observation |
|---|---|
| Age | Default rate 11.9% (18-29) falling to 2.3% (70+) |
| Utilization | 1.8% (<10%) rising to 37.3% (>100%) |
| 90+ days late | 4.7% (none) vs 33.7% (once), 66% (4+ times) |
| Income | 9.2% (lowest quintile) vs 4.6% (highest); weak next to payment history |
| Dependents | 5.8% (none) rising to about 10% (4+) |
| Correlation with target | late_90 0.32, late_30_59 0.27, late_60_89 0.27, utilization 0.25, age -0.12 |

## 4. Features (`src/feature_engineering.py`)
total_late, any_late, severe_late, log_income, debt_payment_est (debt ratio x income), income_per_person, util_over_100, credit_lines_per_age.

## 5. Modelling
Stratified split 60/20/20 (train 89,539 / validation 29,847 / test 29,847). Class imbalance handled with class weights. Threshold chosen to maximise F1 on validation. Final numbers on the untouched test set. Model selected by validation PR-AUC.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.856 | 0.369 | 0.373 | 0.502 | 0.428 |
| Random Forest | 0.860 | 0.390 | 0.399 | 0.487 | 0.439 |
| **Gradient Boosting (tuned, selected)** | **0.864** | **0.399** | 0.369 | 0.532 | 0.436 |

Tuning: randomized search (10 candidates, 3-fold CV, scored on PR-AUC). Accuracy (about 91%) is deliberately not used for selection: predicting "no default" for everyone already scores 93.3%.

**Calibration.** Class weights distort raw scores, so an isotonic calibrator was fitted on validation data. Test Brier score 0.049. Calibrated probabilities are bounded to 0.5%-95%, because an isotonic calibrator can otherwise output exactly 0% or 100%, which is not credible for any individual. Risk bands from calibrated probability:

| Band | Applicants (test) | Actual default rate |
|---|---|---|
| Low (<5%) | 20,926 | 1.7% |
| Medium (5-20%) | 5,975 | 9.6% |
| High (>20%) | 2,946 | 36.6% |

The riskiest 10% of applicants contain 54% of all defaults (36.4% default rate in that group vs 6.7% overall).

## 6. Drivers
Permutation importance (PR-AUC drop): total_late 0.209, utilization 0.060, late_90 0.018, then debt ratio, open credit lines, age (each below 0.01). Payment history and credit utilization dominate; demographics add little.

## 7. Limitations
- Model gaps: PR-AUC around 0.40 means roughly 6 in 10 flagged applicants do not default; use as a screening tool, not an automatic decision.
- The income field was imputed before it reached us; re-run on the original file.
- Dataset reflects one historical period; no fairness audit was performed (age is a feature, which may be restricted in real lending).
- No SHAP/XGBoost (not installed in the build environment); permutation importance used instead.
- Dashboard was syntax-checked and its scoring logic tested, but Streamlit itself was not available here, so run it locally and report any errors.

---

# Technical Report: Churn and Segmentation (Part 2)

## 8. Data
Bank Customer Churn (`Churn_Modelling.csv`): 10,000 customers, 14 columns, no missing values, no duplicates. Target `Exited`: 20.4% churned. Identifier columns (RowNumber, Surname) dropped. Customers are a **different population** from the credit dataset.

## 9. Churn EDA
| Driver | Observation |
|---|---|
| Country | Germany 32.4%, France 16.2%, Spain 16.7% |
| Age | 7.6% (18-29), 10.9% (30-39), 30.8% (40-49), 56.0% (50-59), 28.0% (60+) |
| Products | 1 product 27.7%, 2 products 7.6%, 3-4 products 85.9% (326 customers only) |
| Activity | Inactive 26.9% vs active 14.3% |
| Gender | Female 25.1% vs male 16.5% |

## 10. Churn model
Same protocol as the credit model: stratified 60/20/20 split (6,000 / 2,000 / 2,000), class weights, threshold chosen on validation, results on the test set, model selected by validation PR-AUC. Features added: balance_salary_ratio, zero_balance, germany_multi.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.787 | 0.505 | 0.415 | 0.681 | 0.516 |
| Random Forest | 0.862 | 0.688 | 0.626 | 0.604 | 0.615 |
| **Gradient Boosting (selected)** | **0.866** | **0.713** | 0.621 | 0.629 | 0.625 |

Logistic regression is clearly weaker because churn depends on non-linear patterns (age peaks around 50-59; 2 products is safe but 1 or 3+ is not).

**Scoring every customer.** Each customer's churn score is an out-of-fold prediction (5-fold), so no customer is scored by a model that trained on them. Raw scores were then calibrated (isotonic, bounded to 0.5%-95%). Calibration was fitted on these same out-of-fold scores, so the band rates below are slightly optimistic.

| Churn band | Customers | Actual churn rate |
|---|---|---|
| Low (<15%) | 6,183 | 5.8% |
| Medium (15-40%) | 2,410 | 26.4% |
| High (>40%) | 1,407 | 74.1% |

The High band (14% of customers) contains 51% of all churners.

**Drivers** (permutation importance, PR-AUC drop): Age 0.271, NumOfProducts 0.261, IsActiveMember 0.106, Geography 0.038, Balance 0.029.

## 11. Segmentation
K-Means on 7 standardised variables (Age, CreditScore, Balance, NumOfProducts, Tenure, EstimatedSalary, IsActiveMember); k=6 chosen by best silhouette over k=3..6.

| Segment | Customers | Churn rate | Recommended action |
|---|---|---|---|
| High-balance, inactive | 1,979 | 32.4% | Top priority: personal outreach, re-engagement |
| Senior customers | 961 | 32.0% | Relationship manager, retirement products |
| Multi-product, inactive | 1,102 | 31.4% | Review product fit and fees |
| Dormant, near-zero balance | 1,728 | 16.4% | Low-cost digital nudges |
| Active, high-balance | 2,505 | 12.6% | Protect and cross-sell |
| Active, low-balance | 1,725 | 8.2% | Loyal base; grow balances |

Segment names come from rules applied to each segment's average profile (`name_segment` in `src/train_churn.py`). Actions are recommendations from the patterns above, not tested interventions.

## 12. Combined view and limitations
- Default risk (loan applicants) and churn/segments (existing customers) come from separate datasets. They are reported side by side and never joined per person.
- The silhouette scores are low (0.14 to 0.16): segments overlap. They are useful groupings, not sharp natural clusters.
- The 3-4 product group is small (326); the 86% churn rate suggests a product or data-quality issue worth investigating, not a reliable rule.
- Churn data is a snapshot with no time dimension, so "likely to churn soon" is relative risk, not a dated forecast.
- Gender and geography are used by the model; a real deployment would need a fairness review.
