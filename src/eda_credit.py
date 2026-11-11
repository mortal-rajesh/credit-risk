"""EDA for the credit dataset: figures + a findings JSON used in the reports."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, pandas as pd, numpy as np
from src.data_cleaning import TARGET
from src.feature_engineering import age_group
from src.utils import ROOT, save_json

FIG = ROOT / "reports" / "figures"
sns.set_theme(style="whitegrid")

def rate_by(df, col):
    return df.groupby(col, observed=True)[TARGET].agg(["mean", "size"]).rename(columns={"mean": "default_rate"})

def run(df):
    FIG.mkdir(parents=True, exist_ok=True)
    f = {"rows": len(df), "default_rate": float(df[TARGET].mean())}

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    df[TARGET].value_counts().sort_index().plot.bar(ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_xticklabels(["No default", "Default"], rotation=0); ax.set_title("Target distribution")
    fig.tight_layout(); fig.savefig(FIG / "01_target.png", dpi=130); plt.close(fig)

    d = df.assign(age_group=age_group(df["age"]))
    r = rate_by(d, "age_group"); f["by_age_group"] = r["default_rate"].round(4).to_dict()
    util_bin = pd.cut(df["utilization"], [-0.01, 0.1, 0.3, 0.6, 0.9, 1.0, 5], labels=["<10%", "10-30%", "30-60%", "60-90%", "90-100%", ">100%"])
    ru = rate_by(df.assign(util_bin=util_bin), "util_bin"); f["by_utilization"] = ru["default_rate"].round(4).to_dict()
    inc_bin = pd.qcut(df["monthly_income"], 5, duplicates="drop")
    ri = rate_by(df.assign(inc=inc_bin), "inc"); f["by_income_quintile"] = {str(k): round(v, 4) for k, v in ri["default_rate"].items()}
    late = df["late_90"].clip(upper=4)
    rl = rate_by(df.assign(late90=late), "late90"); f["by_90day_late"] = rl["default_rate"].round(4).to_dict()
    dep = df["dependents"].clip(upper=5)
    rd = rate_by(df.assign(dep=dep), "dep"); f["by_dependents"] = rd["default_rate"].round(4).to_dict()

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (r_, t) in zip(axes.ravel(), [(r, "Age group"), (ru, "Credit utilization"), (ri, "Income quintile"),
                                          (rl, "Times 90+ days late (4 = 4+)"), (rd, "Dependents (5 = 5+)")]):
        ax.bar(range(len(r_)), r_["default_rate"] * 100, color="#4C72B0")
        ax.set_xticks(range(len(r_))); ax.set_xticklabels([str(i) if not hasattr(i, "mid") else f"{i.left:.0f}-{i.right:.0f}" for i in r_.index], rotation=30, fontsize=8)
        ax.set_title(f"Default rate by {t}"); ax.set_ylabel("%")
    axes[1, 2].axis("off"); fig.tight_layout(); fig.savefig(FIG / "02_default_rate_by_driver.png", dpi=130); plt.close(fig)

    corr = df.drop(columns=["utilization_extreme", "debt_ratio_extreme"]).corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6.5)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation matrix"); fig.tight_layout(); fig.savefig(FIG / "03_correlation.png", dpi=130); plt.close(fig)
    f["corr_with_target"] = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).round(3).to_dict()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, c in zip(axes, ["age", "monthly_income", "utilization"]):
        sns.histplot(df[c].dropna(), bins=40, ax=ax, color="#4C72B0"); ax.set_title(c)
    fig.tight_layout(); fig.savefig(FIG / "04_distributions.png", dpi=130); plt.close(fig)
    f["extreme_utilization_pct"] = float(df["utilization_extreme"].mean() * 100)
    f["extreme_debt_ratio_pct"] = float(df["debt_ratio_extreme"].mean() * 100)
    f["missing_pct"] = (df.isna().mean() * 100).round(2)[lambda s: s > 0].to_dict()
    save_json(f, ROOT / "reports" / "eda_findings.json")
    return f
