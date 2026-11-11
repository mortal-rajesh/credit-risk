"""Churn model + customer segmentation on the Bank Customer Churn dataset (Churn_Modelling.csv).

Usage:  python -m src.train_churn data/raw/Churn_Modelling.csv
Outputs: models/churn_model.joblib, data/processed/churn_scored.csv, reports/churn_results.json, figures.
Segmentation is done on the SAME customers so each segment gets a real churn rate.
"""
import sys, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, precision_recall_curve
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from src.utils import ROOT, metrics_at, save_json

SEED, TARGET = 42, "Exited"
FIG = ROOT / "reports" / "figures"
NUM = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary", "balance_salary_ratio"]
BIN = ["HasCrCard", "IsActiveMember", "zero_balance", "germany_multi"]
CAT = ["Geography", "Gender"]
SEG_VARS = ["Age", "CreditScore", "Balance", "NumOfProducts", "Tenure", "EstimatedSalary", "IsActiveMember"]


def load_clean(path):
    df = pd.read_csv(path).drop(columns=["RowNumber", "Surname"], errors="ignore").drop_duplicates()
    df["balance_salary_ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["zero_balance"] = (df["Balance"] == 0).astype(int)
    df["germany_multi"] = ((df["Geography"] == "Germany") & (df["NumOfProducts"] >= 2)).astype(int)
    return df.reset_index(drop=True)


def name_segment(r):
    """Plain-English label from a segment's average profile (rule order matters)."""
    if r["age"] > 50: return "Senior customers"
    if r["active"] < 0.2 and r["products"] >= 2.0: return "Multi-product, inactive"
    if r["active"] < 0.2 and r["balance"] > 50000: return "High-balance, inactive"
    if r["active"] < 0.2: return "Dormant, near-zero balance"
    if r["balance"] > 50000: return "Active, high-balance"
    return "Active, low-balance"


def prep():
    return ColumnTransformer([("n", StandardScaler(), NUM), ("b", "passthrough", BIN),
                              ("c", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])


def best_f1_thr(y, p):
    pr, rc, th = precision_recall_curve(y, p); f1 = 2 * pr * rc / (pr + rc + 1e-12)
    return float(th[np.argmax(f1[:-1])])


def main(path):
    df = load_clean(path); FIG.mkdir(parents=True, exist_ok=True)
    X, y = df[NUM + BIN + CAT], df[TARGET]
    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.25, stratify=ytv, random_state=SEED)
    models = {
        "Logistic Regression": Pipeline([("p", prep()), ("m", LogisticRegression(class_weight="balanced", max_iter=2000))]),
        "Random Forest": Pipeline([("p", prep()), ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=10, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))]),
        "Gradient Boosting": Pipeline([("p", prep()), ("m", HistGradientBoostingClassifier(learning_rate=0.05, max_depth=4, max_iter=250, min_samples_leaf=30, l2_regularization=2, class_weight="balanced", random_state=SEED))]),
    }
    res, vp, tp = {}, {}, {}
    for n, m in models.items():
        m.fit(Xtr, ytr); vp[n], tp[n] = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        thr = best_f1_thr(yva, vp[n]); res[n] = {"validation": metrics_at(yva, vp[n], thr), "test": metrics_at(yte, tp[n], thr)}
        print(n, {k: round(v, 3) for k, v in res[n]["test"].items()})
    best = max(res, key=lambda k: res[k]["validation"]["pr_auc"]); model = models[best]
    thr = res[best]["validation"]["threshold"]; print("SELECTED:", best)

    pi = permutation_importance(model, Xte, yte, scoring="average_precision", n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.5)); imp[::-1].plot.barh(ax=ax, color="#4C72B0"); ax.set_title("What drives churn (permutation importance)")
    fig.tight_layout(); fig.savefig(FIG / "10_churn_importance.png", dpi=130); plt.close(fig)

    # out-of-fold churn probability for EVERY customer (no customer is scored by a model that saw them)
    oof = cross_val_predict(models[best], X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    # class weights inflate raw scores -> calibrate the out-of-fold scores so they behave like real probabilities
    calib = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.95).fit(oof, y)
    raw_oof, oof = oof, calib.predict(oof)
    thr_cal = float(calib.predict([thr])[0])
    joblib.dump({"model": model, "threshold": thr, "calibrator": calib, "features": list(X.columns)}, ROOT / "models/churn_model.joblib")

    # ---------------- segmentation (K-Means) ----------------
    Z = StandardScaler().fit_transform(df[SEG_VARS]); sil = {}
    for k in range(3, 7):
        sil[k] = float(silhouette_score(Z, KMeans(k, n_init=10, random_state=SEED).fit_predict(Z), sample_size=min(5000, len(Z)), random_state=SEED))
    k = max(sil, key=sil.get); km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
    df["segment"] = km.labels_; df["churn_probability"] = oof; df["churn_score_raw"] = raw_oof
    prof = df.groupby("segment").agg(customers=("segment", "size"), age=("Age", "mean"), credit_score=("CreditScore", "mean"), balance=("Balance", "mean"),
                                     products=("NumOfProducts", "mean"), tenure=("Tenure", "mean"), salary=("EstimatedSalary", "mean"),
                                     active=("IsActiveMember", "mean"), actual_churn_rate=(TARGET, "mean"), avg_predicted_churn=("churn_probability", "mean")).round(3)
    prof["share_of_customers"] = (prof["customers"] / len(df)).round(3)
    prof.insert(0, "segment_name", [name_segment(r) for _, r in prof.iterrows()])
    df["segment_name"] = df["segment"].map(prof["segment_name"])
    prof.to_csv(ROOT / "reports/segment_profiles.csv")
    pc = PCA(2, random_state=SEED).fit_transform(Z); s = np.random.RandomState(SEED).choice(len(df), min(4000, len(df)), replace=False)
    fig, ax = plt.subplots(figsize=(6.5, 5)); sns.scatterplot(x=pc[s, 0], y=pc[s, 1], hue=df["segment"].values[s], palette="tab10", s=12, ax=ax)
    ax.set_title(f"Customer segments (K-Means, k={k}, PCA view)"); fig.tight_layout(); fig.savefig(FIG / "11_segments.png", dpi=130); plt.close(fig)
    df["churn_flag"] = (df["churn_score_raw"] >= thr).astype(int)
    df["churn_band"] = pd.cut(df["churn_probability"], [-0.01, 0.15, 0.40, 1.0], labels=["Low", "Medium", "High"])
    band = df.groupby("churn_band", observed=True)[TARGET].agg(customers="size", actual_churn_rate="mean").round(4)
    print(band)
    df.to_csv(ROOT / "data/processed/churn_scored.csv", index=False)
    save_json({"selected_model": best, "results": res, "permutation_importance": imp.round(5).to_dict(), "silhouette_by_k": sil, "chosen_k": k,
               "churn_rate": float(y.mean()), "n_customers": len(df), "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
               "churn_bands": band.reset_index().astype({"churn_band": str}).to_dict("records"),
               "high_band_capture": float(df.loc[df.churn_band == "High", TARGET].sum() / df[TARGET].sum())}, ROOT / "reports/churn_results.json")
    print(prof.to_string())

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/raw/Churn_Modelling.csv")
