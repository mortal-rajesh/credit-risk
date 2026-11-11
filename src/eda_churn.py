"""Churn EDA: figures + findings JSON (run after src.train_churn)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd
from src.utils import ROOT, save_json

FIG = ROOT / "reports" / "figures"

def main():
    df = pd.read_csv(ROOT / "data/processed/churn_scored.csv")
    t = "Exited"; f = {"customers": len(df), "churn_rate": float(df[t].mean())}
    d = df.assign(age_group=pd.cut(df["Age"], [17, 29, 39, 49, 59, 100], labels=["18-29", "30-39", "40-49", "50-59", "60+"]),
                  products=df["NumOfProducts"].clip(upper=3).map({1: "1", 2: "2", 3: "3-4"}))
    groups = {"Geography": "Country", "age_group": "Age group", "products": "Number of products", "IsActiveMember": "Active member (1=yes)",
              "Gender": "Gender", "zero_balance": "Zero balance (1=yes)"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (c, title) in zip(axes.ravel(), groups.items()):
        r = d.groupby(c, observed=True)[t].mean() * 100; f["by_" + c] = r.round(2).to_dict()
        ax.bar([str(i) for i in r.index], r.values, color="#4C72B0"); ax.set_title(f"Churn rate by {title}"); ax.set_ylabel("%")
    fig.tight_layout(); fig.savefig(FIG / "08_churn_eda.png", dpi=130); plt.close(fig)
    # combined business view: segments (churn) -- credit risk is a separate population, see README
    prof = pd.read_csv(ROOT / "reports/segment_profiles.csv")
    f["segments"] = prof[["segment_name", "customers", "actual_churn_rate"]].to_dict("records")
    fig, ax = plt.subplots(figsize=(8, 4)); p = prof.sort_values("actual_churn_rate")
    ax.barh(p["segment_name"], p["actual_churn_rate"] * 100, color="#C44E52"); ax.set_xlabel("Churn rate (%)"); ax.set_title("Churn rate by customer segment")
    fig.tight_layout(); fig.savefig(FIG / "09_segment_churn.png", dpi=130); plt.close(fig)
    save_json(f, ROOT / "reports/churn_eda_findings.json"); return f

if __name__ == "__main__":
    import json; print(json.dumps(main(), indent=1))
