"""FinTrust Risk Radar: credit default & churn analytics (Streamlit).
Run from the repo root:  streamlit run dashboard/app.py
All numbers are read from the saved results in reports/ and data/processed/.
"""
import sys, json
from pathlib import Path
import joblib, pandas as pd, streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.feature_engineering import add_credit_features  # noqa: E402

st.set_page_config(page_title="FinTrust Risk Radar", page_icon="📊", layout="wide")

# Colours use transparent greys + currentColor so the design works in Streamlit's light AND dark themes.
CSS = """<style>
:root{--ac:#19b88a;--rd:#e5566a;--am:#eba23a;--bl:#4c8dff;--line:rgba(127,127,127,.3);--card:rgba(127,127,127,.07);--hi:rgba(127,127,127,.2);--mut:rgba(127,127,127,1)}
.block-container{max-width:1240px;padding-top:1.6rem}
#MainMenu,footer{visibility:hidden}header[data-testid="stHeader"]{background:transparent}
.lg{display:flex;gap:12px;align-items:center;margin-bottom:14px}.lg i{width:40px;height:40px;border-radius:11px;background:var(--ac);display:grid;place-items:center;font-style:normal;font-size:20px}
.lg b{font-size:19px;display:block;line-height:1.2}.lg span{color:var(--mut);font-size:12px}
.stTabs [data-baseweb="tab-list"]{gap:4px;border:1px solid var(--line);border-radius:12px;padding:4px;background:var(--card);width:fit-content;max-width:100%}
.stTabs [data-baseweb="tab"]{border-radius:9px;padding:6px 14px;height:auto}
.stTabs [aria-selected="true"]{background:var(--hi)}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
h1.t1{font-size:30px;font-weight:700;margin:14px 0 2px;letter-spacing:-.5px;padding:0}p.s1{color:var(--mut);margin:0 0 18px}
.c{border:1px solid var(--line);border-radius:16px;padding:18px;background:var(--card);margin-bottom:14px;position:relative;min-width:0}
.c h3{margin:0;font-size:15px;font-weight:600;padding:0}.c p.sb{margin:2px 0 14px;color:var(--mut);font-size:12px}
.k small{letter-spacing:.12em;text-transform:uppercase;color:var(--mut);font-size:11px}.k svg{position:absolute;right:16px;top:20px}
.k .v{font-size:34px;font-weight:700;margin:6px 0 10px;letter-spacing:-1px;line-height:1.1}
.b{display:inline-block;font-size:12px;padding:3px 9px;border-radius:7px;background:var(--hi);color:var(--mut)}.b.g1{color:var(--ac)}.b.r1{color:var(--rd)}
.r{display:grid;grid-template-columns:minmax(90px,150px) 1fr 56px;gap:10px;align-items:center;margin:9px 0;font-size:13px}
.r .l{color:var(--mut);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.r .t{height:9px;background:var(--hi);border-radius:5px}.r .t i{display:block;height:100%;border-radius:5px}.r em{font-style:normal;text-align:right;font-weight:600}
.dn{display:flex;gap:22px;align-items:center;flex-wrap:wrap}.dn svg{width:150px;height:150px}
.pill{padding:2px 9px;border-radius:99px;font-size:12px;font-weight:600;background:var(--hi)}
.tb{width:100%;border-collapse:collapse;font-size:13px}.tb th{text-align:left;color:var(--mut);font-weight:500;font-size:11px;letter-spacing:.08em;text-transform:uppercase;padding:6px 8px;border-bottom:1px solid var(--line)}.tb td{padding:9px 8px;border-bottom:1px solid var(--line)}
.nt{color:var(--mut);font-size:12px;margin-top:10px}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

NM = {"total_late": "Total late payments", "utilization": "Credit usage", "late_90": "90+ days late", "debt_ratio": "Debt ratio",
      "open_credit_lines": "Open credit lines", "age": "Age", "Age": "Age", "NumOfProducts": "Number of products", "IsActiveMember": "Active member",
      "Geography": "Country", "Balance": "Balance", "late_30_59": "30-59 days late", "late_60_89": "60-89 days late", "debt_payment_est": "Est. debt payment",
      "log_income": "Income", "monthly_income": "Income", "income_per_person": "Income per person"}
ACT = {"High-balance, inactive": "Personal outreach (top priority)", "Senior customers": "Relationship manager",
       "Multi-product, inactive": "Review product fit and fees", "Dormant, near-zero balance": "Low-cost digital nudges",
       "Active, high-balance": "Protect and cross-sell", "Active, low-balance": "Help balances grow"}
BANDS = ["Low (<5%)", "Medium (5-20%)", "High (>20%)"]


@st.cache_data
def load():
    rd = lambda p: json.loads((ROOT / p).read_text())
    return dict(credit=pd.read_csv(ROOT / "data/processed/credit_test_scored.csv"), churn=pd.read_csv(ROOT / "data/processed/churn_scored.csv"),
                cr=rd("reports/credit_model_results.json"), ch=rd("reports/churn_results.json"), ed=rd("reports/eda_findings.json"),
                ce=rd("reports/churn_eda_findings.json"), seg=pd.read_csv(ROOT / "reports/segment_profiles.csv"))


@st.cache_resource
def model():
    return joblib.load(ROOT / "models/default_model.joblib")


# ---------- small HTML builders (single-line strings: Streamlit markdown breaks on blank lines) ----------
P = lambda x, d=1: f"{x * 100:.{d}f}%"
N = lambda x: f"{x:,.0f}"
col = lambda v: "var(--rd)" if v > .3 else "var(--am)" if v > .1 else "var(--ac)"


def show(html, target=st):
    target.markdown(html, unsafe_allow_html=True)


def bars(items, fmt=P, color=None, mx=None):
    items = list(items.items()) if isinstance(items, dict) else list(items)
    mx = mx or max(v for _, v in items) or 1
    return "".join(f'<div class="r"><span class="l">{NM.get(k, k)}</span><span class="t"><i style="width:{max(2, v / mx * 100):.0f}%;'
                   f'background:{color(v) if color else "var(--bl)"}"></i></span><em>{fmt(v)}</em></div>' for k, v in items)


def spark(vals, c):
    w, h, vals = 96, 34, list(vals); lo, hi = min(vals), max(vals)
    pts = [(i * w / (len(vals) - 1), h - 3 - (y - lo) / ((hi - lo) or 1) * (h - 8)) for i, y in enumerate(vals)]
    l = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return (f'<svg width="{w}" height="{h}"><polygon points="0,{h} {l} {w},{h}" style="fill:{c};opacity:.15"/>'
            f'<polyline points="{l}" style="fill:none;stroke:{c};stroke-width:2"/></svg>')


def kpi(label, value, badge, cls="", sp=""):
    return f'<div class="c k"><small>{label}</small>{sp}<div class="v">{value}</div><span class="b {cls}">{badge}</span></div>'


def card(title, sub, inner):
    return f'<div class="c"><h3>{title}</h3><p class="sb">{sub}</p>{inner}</div>'


def donut(bands):
    tot, off, arcs, leg, cs, C = sum(b[1] for b in bands), 0, "", "", ["var(--ac)", "var(--am)", "var(--rd)"], 289.03
    for (name, n, *_), c in zip(bands, cs):
        ln = n / tot * C
        arcs += f'<circle cx="60" cy="60" r="46" fill="none" stroke-width="16" style="stroke:{c}" stroke-dasharray="{ln:.2f} {C - ln:.2f}" stroke-dashoffset="{-off:.2f}" transform="rotate(-90 60 60)"/>'
        leg += f'<div style="margin:6px 0"><span style="color:{c}">●</span> {name} <b>{N(n)}</b></div>'; off += ln
    return (f'<div class="dn"><svg viewBox="0 0 120 120">{arcs}<text x="60" y="58" text-anchor="middle" font-size="15" font-weight="700" fill="currentColor">{N(tot)}</text>'
            f'<text x="60" y="73" text-anchor="middle" font-size="8" fill="currentColor" opacity=".6">applicants</text></svg><div>{leg}</div></div>')


def table(head, rows):
    return ('<table class="tb"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>" +
            "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>")


def pill(v): return f'<span class="pill" style="color:{col(v)}">{P(v)}</span>'
def title(t, s): show(f'<h1 class="t1">{t}</h1><p class="s1">{s}</p>')
def band_of(p): return BANDS[0] if p < .05 else BANDS[1] if p <= .20 else BANDS[2]


# ---------- app ----------
D = load(); cr, ch, ed, ce = D["cr"], D["ch"], D["ed"], D["ce"]
g, t = cr["results"][cr["selected_model"]]["test"], ch["results"][ch["selected_model"]]["test"]
credit, churn = D["credit"], D["churn"]
seg = D["seg"].sort_values("actual_churn_rate", ascending=False)
hb = ch["churn_bands"]; n_high = hb[2]["customers"]

show('<div class="lg"><i>📊</i><div><b>FinTrust Risk Radar</b><span>Credit default &amp; churn analytics · real project data</span></div></div>')
tabs = st.tabs(["Overview", "Credit risk", "Churn & segments", "Lookup", "Risk calculator"])

with tabs[0]:
    title("Risk &amp; retention overview", f"{N(len(credit))} applicants scored · {N(len(churn))} customers analysed · {N(n_high)} flagged high churn risk")
    k = st.columns(4)
    show(kpi("Default rate", P(ed["default_rate"]), "1 in 15 applicants", "", spark(ed["by_age_group"].values(), "var(--rd)")), k[0])
    show(kpi("Default model AUC", f"{g['roc_auc']:.3f}", f"Top 10% hold {P(cr['top_decile_capture_of_defaults'], 0)} of defaults", "g1"), k[1])
    show(kpi("Churn rate", P(ch["churn_rate"]), "Peaks at age 50-59", "r1", spark(ce["by_age_group"].values(), "var(--am)")), k[2])
    show(kpi("High-risk churners", N(n_high), f"{P(hb[2]['actual_churn_rate'], 0)} of them left", "r1"), k[3])
    a, b = st.columns(2)
    show(card("Actual default rate by risk band", "Held-out applicants, calibrated bands", bars([(x["risk_band"], x["actual_default_rate"]) for x in cr["risk_bands"]], color=col, mx=.4)), a)
    show(card("Applicants by risk band", "Share of the test set", donut([(x["risk_band"], x["applicants"]) for x in cr["risk_bands"]])), b)
    a, b = st.columns(2)
    show(card("Churn rate by predicted risk band", "Customers per band in brackets", bars([(f"{x['churn_band']} ({N(x['customers'])})", x["actual_churn_rate"]) for x in hb], color=col, mx=.8)), a)
    show(card("What drives default risk", "Permutation importance (drop in PR-AUC)", bars(list(cr["permutation_importance"].items())[:7], fmt=lambda v: f"{v:.3f}")), b)

with tabs[1]:
    title("Credit default risk", "Which applicants are likely to default, and why")
    k = st.columns(4)
    show(kpi("Test applicants", N(len(credit)), "never used in training"), k[0])
    show(kpi("ROC-AUC", f"{g['roc_auc']:.3f}", f"baseline {cr['results']['Logistic Regression']['test']['roc_auc']:.3f}", "g1"), k[1])
    show(kpi("PR-AUC", f"{g['pr_auc']:.3f}", "rare-event score"), k[2])
    show(kpi("Recall / precision", f"{P(g['recall'], 0)} / {P(g['precision'], 0)}", "6 in 10 flags are false alarms", "r1"), k[3])
    a, b = st.columns(2)
    show(card("Default rate by credit usage", "Share of credit limit in use", bars(ed["by_utilization"], color=col, mx=.4)), a)
    show(card("Default rate by age group", "Younger applicants default more", bars(ed["by_age_group"], color=col, mx=.4)), b)
    a, b = st.columns(2)
    show(card("Default rate by 90+ day late count", "4 = four or more", bars(ed["by_90day_late"], color=col, mx=.7)), a)
    show(card("Model comparison (test set)", "ROC-AUC · PR-AUC · recall", table(["Model", "ROC", "PR", "Recall"],
         [[k_, f"{v['test']['roc_auc']:.3f}", f"{v['test']['pr_auc']:.3f}", P(v['test']['recall'], 0)] for k_, v in cr["results"].items()])), b)

with tabs[2]:
    title("Churn &amp; customer segments", "Who is leaving, and which group needs which action")
    k = st.columns(4)
    show(kpi("Customers", N(len(churn)), "Bank Customer Churn data"), k[0])
    show(kpi("Churn rate", P(ch["churn_rate"]), "2 products: 7.6%"), k[1])
    show(kpi("Churn model AUC", f"{t['roc_auc']:.3f}", f"PR-AUC {t['pr_auc']:.3f}", "g1"), k[2])
    show(kpi("Leavers in High band", P(ch["high_band_capture"], 0), "in 14% of customers", "r1"), k[3])
    pc = lambda v: col(v / 100); f1 = lambda v: f"{v:.1f}%"
    a, b = st.columns(2)
    show(card("Churn by age group", "Percent of customers who left", bars(ce["by_age_group"], fmt=f1, color=pc, mx=60)), a)
    mix = {**ce["by_Geography"], **{f"{k_} product(s)": v for k_, v in ce["by_products"].items()}, **{("Active member" if k_ == "1" else "Inactive member"): v for k_, v in ce["by_IsActiveMember"].items()}}
    show(card("Churn by country, products and activity", "Percent of customers who left", bars(mix, fmt=f1, color=pc, mx=90)), b)
    a, b = st.columns(2)
    show(card("Churn drivers", "Permutation importance", bars(list(ch["permutation_importance"].items())[:5], fmt=lambda v: f"{v:.3f}")), a)
    show(card("Segments and suggested action", "Six K-Means groups, highest churn first", table(["Segment", "Cust.", "Churn", "Action"],
         [[r.segment_name, N(r.customers), pill(r.actual_churn_rate), ACT.get(r.segment_name, "")] for r in seg.itertuples()])), b)

with tabs[3]:
    title("Lookup", "Search any applicant or customer by ID")
    a, b = st.columns(2)
    with a:
        show('<div class="c"><h3>Loan applicant risk</h3><p class="sb">Held-out test applicants (IDs like A171190)</p></div>')
        aid = st.text_input("Applicant ID", placeholder="A171190", key="aid").strip().upper()
        if aid:
            r = credit[credit["applicant_id"] == aid]
            if r.empty:
                st.error("ID not found. Applicant IDs look like A171190: copy one from the table below.")
            else:
                x = r.iloc[0]; show(kpi("Estimated default probability", P(x["default_probability"]), f"Risk band: {band_of(x['default_probability'])}", "r1" if x["default_probability"] > .2 else "g1"))
                show(card("Applicant profile", "", table(["Age", "Income", "Usage", "Debt ratio", "90d late"], [[int(x["age"]), N(x["monthly_income"]), P(x["utilization"], 0), f"{x['debt_ratio']:.2f}", int(x["late_90"]) if pd.notna(x["late_90"]) else "n/a"]])))
        bands = st.multiselect("Show risk bands", BANDS, default=[BANDS[2]], key="bands")
        top = credit[credit["risk_band"].isin(bands)].nlargest(200, "default_probability")
        st.dataframe(top[["applicant_id", "default_probability", "risk_band", "age", "monthly_income", "utilization", "late_90"]], use_container_width=True, hide_index=True)
    with b:
        show('<div class="c"><h3>Customer churn risk</h3><p class="sb">All 10,000 customers (numeric Customer ID)</p></div>')
        cid = st.text_input("Customer ID", placeholder="15634602", key="cid").strip()
        if cid:
            if not cid.isdigit():
                st.error("Customer ID must be a number, for example 15634602.")
            else:
                r = churn[churn["CustomerId"] == int(cid)]
                if r.empty:
                    st.error("Customer not found.")
                else:
                    x = r.iloc[0]; show(kpi("Estimated churn probability", P(x["churn_probability"]), f"Segment: {x['segment_name']}", "r1" if x["churn_probability"] > .4 else "g1"))
                    show(card("Suggested action", "", f"<b>{ACT.get(x['segment_name'], '')}</b>"))
        cb = st.multiselect("Churn-risk band", ["High", "Medium", "Low"], default=["High"], key="cb")
        topc = churn[churn["churn_band"].isin(cb)].nlargest(200, "churn_probability")
        st.dataframe(topc[["CustomerId", "segment_name", "churn_probability", "churn_band", "Age", "Balance", "NumOfProducts"]], use_container_width=True, hide_index=True)

with tabs[4]:
    title("Risk calculator", "Estimate default risk for a new loan applicant")
    c = st.columns(3)
    age = c[0].number_input("Age", 18, 100, 40); income = c[1].number_input("Monthly income", 0, 50000, 5000, step=100)
    util = c[2].number_input("Credit usage (1.0 = 100% of limits)", 0.0, 5.0, 0.3, step=0.05)
    debt = c[0].number_input("Debt ratio (monthly debt / income)", 0.0, 5.0, 0.35, step=0.05); dep = c[1].number_input("Dependents", 0, 20, 1)
    lines = c[2].number_input("Open credit lines & loans", 0, 60, 8); re_ = c[0].number_input("Real-estate loans", 0, 30, 1)
    l30 = c[1].number_input("Times 30-59 days late", 0, 20, 0); l60 = c[2].number_input("Times 60-89 days late", 0, 20, 0); l90 = c[0].number_input("Times 90+ days late", 0, 20, 0)
    if st.button("Calculate risk", type="primary"):
        row = pd.DataFrame([{"utilization": util, "age": age, "late_30_59": l30, "debt_ratio": debt, "monthly_income": income, "open_credit_lines": lines,
                             "late_90": l90, "real_estate_loans": re_, "late_60_89": l60, "dependents": dep, "utilization_extreme": 0, "debt_ratio_extreme": 0}])
        m = model(); raw = m["model"].predict_proba(add_credit_features(row)[m["features"]])[:, 1]; p = float(m["calibrator"].predict(raw)[0])
        k = st.columns(2)
        show(kpi("Estimated default probability", P(p), "statistical estimate", "r1" if p > .2 else "g1"), k[0]); show(kpi("Risk band", band_of(p), "supports, not replaces, underwriting"), k[1])

show('<p class="nt">Loan-applicant data (150,000 rows) and bank-customer data (10,000 rows) are different people, so default risk and churn are never combined per person. '
     'Credit figures come from a 29,847-applicant held-out test set. Segment actions are recommendations, not tested interventions.</p>')
