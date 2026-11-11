"""Reusable cleaning for the Give Me Some Credit dataset.

Rules are fixed, documented domain rules (no statistics learned from the data),
so applying them to the full dataset cannot leak information into the test set.
Imputation of remaining NaNs happens later INSIDE the model pipeline.
"""
import numpy as np
import pandas as pd

TARGET = "SeriousDlqin2yrs"
PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
RENAME = {
    "RevolvingUtilizationOfUnsecuredLines": "utilization",
    "NumberOfTime30-59DaysPastDueNotWorse": "late_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "late_90",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "late_60_89",
    "NumberOfDependents": "dependents",
}
UTIL_CAP, DEBT_CAP = 5.0, 5.0


def clean_credit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.drop_duplicates()
    # 96 / 98 are placeholder codes in the source data, not real counts
    for c in PAST_DUE_COLS:
        df[c] = df[c].where(~df[c].isin([96, 98]), np.nan)
    df.loc[df["age"] < 18, "age"] = np.nan          # impossible ages
    df = df.rename(columns=RENAME)
    # flag extreme values before capping so the information is not lost
    df["utilization_extreme"] = (df["utilization"] > UTIL_CAP).astype(int)
    df["debt_ratio_extreme"] = (df["debt_ratio"] > DEBT_CAP).astype(int)
    df["utilization"] = df["utilization"].clip(upper=UTIL_CAP)
    df["debt_ratio"] = df["debt_ratio"].clip(upper=DEBT_CAP)
    df["monthly_income"] = df["monthly_income"].clip(upper=50_000)
    return df.reset_index(drop=True)
