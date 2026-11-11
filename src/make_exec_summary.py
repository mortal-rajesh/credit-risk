"""Builds reports/executive_summary.pdf from the saved result files."""
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from src.utils import ROOT

cr = json.loads((ROOT / "reports/credit_model_results.json").read_text())
ch = json.loads((ROOT / "reports/churn_results.json").read_text())
cb = {b["risk_band"]: b for b in cr["risk_bands"]}
nb = {b["churn_band"]: b for b in ch["churn_bands"]}

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=19, textColor=colors.HexColor("#12355B"), alignment=0, spaceAfter=2)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, textColor=colors.HexColor("#12355B"), spaceBefore=6, spaceAfter=2)
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.2, leading=12.2)
SM = ParagraphStyle("sm", parent=B, fontSize=8.3, leading=11, textColor=colors.HexColor("#555555"))

def table(data, widths):
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12355B")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.8), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F9")]), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t

def P(t, s=B): return Paragraph(t, s)
pct = lambda x: f"{x*100:.1f}%"

doc = SimpleDocTemplate(str(ROOT / "reports/executive_summary.pdf"), pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm, topMargin=1.2*cm, bottomMargin=1.0*cm,
                        title="Executive Summary: Credit Risk & Retention Analytics", author="Analytics team")
s = []
s += [P("Executive Summary", H1), P("<b>FinTrust Bank: Credit Risk &amp; Retention Analytics</b> (prototype)", B), Spacer(1, 4)]
s += [P("What we were asked", H2),
      P("Reduce losses from loans that are not repaid, and stop valuable customers from leaving. We built a working prototype that (1) rates loan applicants by default risk, "
        "(2) flags existing customers likely to leave, and (3) groups customers into segments with a suggested action for each. A dashboard lets the risk and retention teams use it.")]
s += [P("1. Loan default risk", H2),
      P(f"We studied 150,000 past borrowers; about 1 in 15 (6.7%) ran into serious repayment trouble. Our model sorts applicants into three bands. "
        f"On applicants the model had never seen, the bands performed as follows:"),
      Spacer(1, 3),
      table([["Risk band", "Applicants tested", "Actually defaulted"],
             ["Low (under 5% estimated risk)", f"{cb['Low (<5%)']['applicants']:,}", pct(cb['Low (<5%)']['actual_default_rate'])],
             ["Medium (5-20%)", f"{cb['Medium (5-20%)']['applicants']:,}", pct(cb['Medium (5-20%)']['actual_default_rate'])],
             ["High (over 20%)", f"{cb['High (>20%)']['applicants']:,}", pct(cb['High (>20%)']['actual_default_rate'])]], [7*cm, 4.5*cm, 4.5*cm]),
      Spacer(1, 3),
      P(f"The riskiest 10% of applicants account for {cr['top_decile_capture_of_defaults']*100:.0f}% of all defaults. "
        "<b>What drives risk:</b> a history of late payments (especially 90+ days late) and using nearly all of the available credit limit. Younger applicants also default more often (11.9% aged 18-29 vs 2.3% aged 70+). Income matters much less than payment behaviour.")]
s += [P("2. Customer churn (customers leaving)", H2),
      P(f"About 1 in 5 customers (20.4%) in the 10,000-customer file left the bank. The model ranks customers by their likelihood of leaving: the {nb['High']['customers']:,} customers in the High band (14%) "
        f"actually left {pct(nb['High']['actual_churn_rate'])} of the time and contain {ch['high_band_capture']*100:.0f}% of all leavers, while the Low band left only {pct(nb['Low']['actual_churn_rate'])} of the time. "
        "Patterns we found: customers aged 50-59 leave at 56%; German customers at 32% (about double France and Spain); inactive members at 27% (vs 14% for active); "
        "customers with exactly two products leave least (7.6%).")]
s += [P("3. Customer segments", H2),
      table([["Segment", "Customers", "Left the bank", "Suggested action"],
             ["High-balance, inactive", "1,979", "32.4%", "Top priority: personal outreach"],
             ["Senior customers", "961", "32.0%", "Relationship manager; retirement products"],
             ["Multi-product, inactive", "1,102", "31.4%", "Review product fit and fees"],
             ["Dormant, near-zero balance", "1,728", "16.4%", "Low-cost digital reminders"],
             ["Active, high-balance", "2,505", "12.6%", "Protect and cross-sell"],
             ["Active, low-balance", "1,725", "8.2%", "Loyal base; help balances grow"]], [4.6*cm, 2.3*cm, 2.5*cm, 7.1*cm])]
s += [P("Recommended actions this quarter", H2),
      P("<b>1. Lending:</b> use the three bands to prioritise review. Fast-track Low-band applicants; send High-band applicants for manual review or tighter terms. Use the model to support, not replace, credit officers.<br/>"
        "<b>2. Retention:</b> focus the retention budget on the three segments with roughly 32% churn (about 4,000 customers) rather than all customers, starting with the High-balance inactive group.<br/>"
        "<b>3. Investigate:</b> why German customers leave twice as often, and why customers with 3-4 products leave so often (small group, 326 customers).<br/>"
        "<b>4. Test before scaling:</b> run the suggested retention actions on a sample and measure whether churn actually falls.")]
s += [P("Things to keep in mind", H2),
      P("The loan data and the customer data describe different people, so we report them separately; we cannot say how likely one individual is to both default and leave. "
        f"About 6 in 10 applicants the loan model flags as likely defaulters do not actually default, so it is a screening aid, not an automatic decision. Segments overlap rather than being sharply separate. "
        "Age, gender and country influence the models, so a fairness review is needed before real use. Suggested actions are recommendations, not proven results.", SM)]
doc.build(s)
print("built")
