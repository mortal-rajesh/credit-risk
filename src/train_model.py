"""Train + evaluate credit-default models (LogReg baseline, RandomForest, HistGradientBoosting).

Split: 60% train / 20% validation / 20% test (stratified).
 - imputation / scaling live inside the Pipeline (fitted on train only)
 - decision threshold is chosen on VALIDATION, final numbers reported on untouched TEST
 - class imbalance handled with class weights
"""
import time, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
                             PrecisionRecallDisplay, precision_recall_curve)
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from src.data_cleaning import TARGET
from src.feature_engineering import add_credit_features
from src.utils import ROOT, metrics_at, save_json

SEED = 42
FIG = ROOT / "reports" / "figures"

def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    return float(th[np.argmax(f1[:-1])])

def main():
    df = add_credit_features(pd.read_csv(ROOT / "data/processed/credit_clean.csv"))
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.25, stratify=ytv, random_state=SEED)
    print("train/val/test:", len(Xtr), len(Xva), len(Xte))

    imp = lambda: SimpleImputer(strategy="median")
    models = {
        "Logistic Regression": Pipeline([("i", imp()), ("s", StandardScaler()),
            ("m", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5))]),
        "Random Forest": Pipeline([("i", imp()), ("m", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED))]),
    }
    hgb = Pipeline([("i", imp()), ("m", HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED))])
    search = RandomizedSearchCV(
        hgb, {"m__learning_rate": [0.03, 0.05, 0.1], "m__max_depth": [3, 4, 6], "m__max_leaf_nodes": [15, 31],
              "m__min_samples_leaf": [20, 50, 100], "m__l2_regularization": [0, 1, 5], "m__max_iter": [200, 400]},
        n_iter=10, scoring="average_precision", cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
        n_jobs=-1, random_state=SEED)
    t = time.time(); search.fit(Xtr, ytr); print("HGB search", round(time.time() - t), "s", search.best_params_)
    models["Gradient Boosting (tuned)"] = search.best_estimator_
    cv_pr_auc = float(search.best_score_)

    results, val_p, test_p = {}, {}, {}
    for name, m in models.items():
        if name != "Gradient Boosting (tuned)":
            m.fit(Xtr, ytr)
        val_p[name] = m.predict_proba(Xva)[:, 1]; test_p[name] = m.predict_proba(Xte)[:, 1]
        thr = best_f1_threshold(yva, val_p[name])
        results[name] = {"validation": metrics_at(yva, val_p[name], thr), "test": metrics_at(yte, test_p[name], thr)}
        print(name, {k: round(v, 3) for k, v in results[name]["test"].items()})

    best = max(results, key=lambda k: results[k]["validation"]["pr_auc"])
    print("SELECTED (by validation PR-AUC):", best)
    model, thr = models[best], results[best]["validation"]["threshold"]
    # class weights inflate raw scores -> calibrate on VALIDATION so the dashboard shows real probabilities
    calib = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.95).fit(val_p[best], yva)
    joblib.dump({"model": model, "threshold": thr, "calibrator": calib, "features": list(X.columns)}, ROOT / "models/default_model.joblib")

    # --- figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for n in models:
        RocCurveDisplay.from_predictions(yte, test_p[n], name=n, ax=ax[0])
        PrecisionRecallDisplay.from_predictions(yte, test_p[n], name=n, ax=ax[1])
    ax[0].set_title("ROC (test)"); ax[1].set_title("Precision-Recall (test)")
    fig.tight_layout(); fig.savefig(FIG / "05_roc_pr.png", dpi=130); plt.close(fig)
    cm = confusion_matrix(yte, (test_p[best] >= thr).astype(int))
    fig, ax = plt.subplots(figsize=(4.5, 4)); ConfusionMatrixDisplay(cm, display_labels=["No default", "Default"]).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"{best}\nconfusion matrix (test)"); fig.tight_layout(); fig.savefig(FIG / "06_confusion.png", dpi=130); plt.close(fig)

    # --- drivers: permutation importance on test (PR-AUC drop)
    pi = permutation_importance(model, Xte, yte, scoring="average_precision", n_repeats=5, random_state=SEED, n_jobs=-1)
    imp_s = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5)); imp_s.head(12)[::-1].plot.barh(ax=ax, color="#4C72B0")
    ax.set_title("What drives default risk (permutation importance)"); ax.set_xlabel("Drop in PR-AUC when feature is shuffled")
    fig.tight_layout(); fig.savefig(FIG / "07_importance.png", dpi=130); plt.close(fig)
    lr = models["Logistic Regression"]
    coefs = pd.Series(lr.named_steps["m"].coef_[0], index=X.columns).sort_values(key=abs, ascending=False)

    # --- risk bands + score the full test set for the dashboard
    scored = Xte.copy(); scored[TARGET] = yte.values; scored["risk_score_raw"] = test_p[best]
    scored["default_probability"] = calib.predict(test_p[best])
    scored["risk_band"] = pd.cut(scored["default_probability"], [-0.01, 0.05, 0.20, 1.0], labels=["Low (<5%)", "Medium (5-20%)", "High (>20%)"])
    scored["applicant_id"] = [f"A{100000 + i}" for i in scored.index]
    scored.to_csv(ROOT / "data/processed/credit_test_scored.csv", index=False)
    band = scored.groupby("risk_band", observed=True)[TARGET].agg(["size", "mean"]).rename(columns={"size": "applicants", "mean": "actual_default_rate"})
    print(band)
    top = scored.nlargest(int(len(scored) * 0.1), "risk_score_raw")
    top_decile, capture = top[TARGET].mean(), top[TARGET].sum() / scored[TARGET].sum()
    from sklearn.metrics import brier_score_loss
    brier = float(brier_score_loss(yte, scored["default_probability"]))

    save_json({"split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)}, "selected_model": best,
               "selection_rule": "highest validation PR-AUC; threshold = max-F1 on validation", "tuned_cv_pr_auc": cv_pr_auc, "calibrated_brier_score_test": brier,
               "best_params": search.best_params_, "results": results,
               "risk_bands": band.round(4).reset_index().astype({"risk_band": str}).to_dict("records"),
               "top_decile_default_rate": float(top_decile), "top_decile_capture_of_defaults": float(capture),
               "permutation_importance": imp_s.round(5).head(12).to_dict(), "logreg_coefficients": coefs.round(3).head(10).to_dict()},
              ROOT / "reports/credit_model_results.json")

if __name__ == "__main__":
    main()
