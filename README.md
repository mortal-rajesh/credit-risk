<<<<<<< HEAD
# FinTrust Bank: Credit Risk & Retention Analytics

Internship project: default prediction, churn prediction, customer segmentation, Streamlit dashboard.

## Status
| Component | Status |
|---|---|
| Credit cleaning, EDA, features | Done |
| Default model (LogReg / Random Forest / tuned Gradient Boosting, calibrated) | Done |
| Churn model + K-Means segmentation | Done (real data) |
| Dashboard (credit + churn/segment pages) | Written; run locally to verify |
| Executive summary (PDF), slides (PPTX) | Done |

## Run
```bash
pip install -r requirements.txt
python -m src.train_model            # trains credit models (uses data/processed/credit_clean.csv)
python -m src.train_churn data/raw/Churn_Modelling.csv   # once the churn file is added
streamlit run dashboard/app.py      # main dashboard (Risk Radar design: 5 tabs, full ID lookup, risk calculator)
```
`dashboard/risk_radar.html` is a static, shareable preview of the same design (watchlists only).
To rebuild the credit data: `from src.data_cleaning import clean_credit` on the raw CSV and save to `data/processed/credit_clean.csv`.

## Important limitation
The credit and churn datasets describe different people. The platform therefore reports default risk for loan applicants and churn risk + segment for existing customers separately; it never claims one person has all three scores.
=======
# credit-risk
>>>>>>> c84ba79dd4b12b4415cfe650b6b4dcce9b5737e4
