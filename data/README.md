# Data

| Dataset | Source | Role | Status |
|---|---|---|---|
| Give Me Some Credit (`cs-training.csv`) | Kaggle competition, Give Me Some Credit | Default prediction | Used. Supplied as a pre-cleaned file `raw/credit_input_df_clean.csv` (150,000 rows, 11 columns, target `SeriousDlqin2yrs`) |
| Bank Customer Churn (`Churn_Modelling.csv`) | Kaggle | Churn prediction + segmentation | Used (10,000 rows, 14 columns, target `Exited`) |

Licence: Kaggle competition/dataset terms apply; used for educational prototyping only. Confirm exact licence text on the Kaggle page before publishing the repo.

Notes on the credit file: it arrived with missing `MonthlyIncome`/`NumberOfDependents` already filled (no NaNs), apparently using full-dataset statistics. That is a small leakage risk and also creates a spike of identical incomes (about 5,400) visible in the income quintiles. Re-run from the original `cs-training.csv` to remove it.
