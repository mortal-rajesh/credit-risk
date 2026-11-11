"""Feature engineering for the credit-default model (stateless, leakage-safe)."""
import numpy as np
import pandas as pd

def add_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["total_late"] = d[["late_30_59", "late_60_89", "late_90"]].sum(axis=1, min_count=1)
    d["any_late"] = (d["total_late"] > 0).astype(int)
    d["severe_late"] = (d["late_90"] > 0).astype(int)
    d["log_income"] = np.log1p(d["monthly_income"])
    d["debt_payment_est"] = d["debt_ratio"] * d["monthly_income"]
    d["income_per_person"] = d["monthly_income"] / (1 + d["dependents"])
    d["util_over_100"] = (d["utilization"] > 1).astype(int)
    d["credit_lines_per_age"] = d["open_credit_lines"] / d["age"]
    return d

def age_group(s: pd.Series) -> pd.Series:
    return pd.cut(s, [17, 29, 39, 49, 59, 69, 120],
                  labels=["18-29", "30-39", "40-49", "50-59", "60-69", "70+"])
