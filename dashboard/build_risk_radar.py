import json, pandas as pd
R="/home/claude/fintrust-bank-risk/"
cr=json.load(open(R+"reports/credit_model_results.json")); ch=json.load(open(R+"reports/churn_results.json"))
ed=json.load(open(R+"reports/eda_findings.json")); ce=json.load(open(R+"reports/churn_eda_findings.json"))
seg=pd.read_csv(R+"reports/segment_profiles.csv")
c=pd.read_csv(R+"data/processed/credit_test_scored.csv"); n=pd.read_csv(R+"data/processed/churn_scored.csv")
g=cr["results"][cr["selected_model"]]["test"]; t=ch["results"][ch["selected_model"]]["test"]
D=dict(
 c=dict(rate=ed["default_rate"],n=len(c),auc=g["roc_auc"],pr=g["pr_auc"],rec=g["recall"],pre=g["precision"],cap=cr["top_decile_capture_of_defaults"],
  models={k:[v["test"]["roc_auc"],v["test"]["pr_auc"]] for k,v in cr["results"].items()},
  bands=[[b["risk_band"],b["applicants"],b["actual_default_rate"]] for b in cr["risk_bands"]],
  age=ed["by_age_group"],util=ed["by_utilization"],late=ed["by_90day_late"],
  imp=cr["permutation_importance"]),
 h=dict(rate=ch["churn_rate"],n=ch["n_customers"],auc=t["roc_auc"],pr=t["pr_auc"],cap=ch["high_band_capture"],
  bands=[[b["churn_band"],b["customers"],b["actual_churn_rate"]] for b in ch["churn_bands"]],
  age=ce["by_age_group"],geo=ce["by_Geography"],prod=ce["by_products"],act=ce["by_IsActiveMember"],imp=dict(list(ch["permutation_importance"].items())[:5]),
  seg=[[r.segment_name,int(r.customers),float(r.actual_churn_rate),float(r.balance),float(r.age)] for r in seg.itertuples()]),
 cw=[[r.applicant_id,round(r.default_probability,3),int(r.age),round(r.utilization,2),int(r.late_90) if r.late_90==r.late_90 else 0] for r in c.nlargest(60,"default_probability").itertuples()],
 nw=[[int(r.CustomerId),r.segment_name,round(r.churn_probability,3),int(r.Age),int(r.Balance)] for r in n.nlargest(60,"churn_probability").itertuples()])
tpl=open("/tmp/dash/tpl.html").read()
open("/mnt/user-data/outputs/risk_radar.html","w").write(tpl.replace("__DATA__",json.dumps(D,separators=(",",":"))))
print(len(json.dumps(D)))
