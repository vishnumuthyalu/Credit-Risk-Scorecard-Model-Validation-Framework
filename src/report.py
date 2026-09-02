"""
report.py
=========
Renders every result produced by the pipeline (scorecard, validation
suite, loss forecast) into a single self-contained HTML model validation
report — the kind of document a bank's Model Risk Management function
would expect to see alongside a new scorecard before sign-off. Everything
(plots included) is embedded inline as base64 so the file is portable —
no external assets, opens in any browser.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _df_html(df: pd.DataFrame, max_rows: int = 30) -> str:
    return df.head(max_rows).to_html(index=False, classes="tbl", border=0, float_format=lambda x: f"{x:,.4f}")


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_roc(test_df, test_scores, oot_df, oot_scores, target_col):
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for label, y, scores in [("Test", test_df[target_col], test_scores),
                              ("OOT", oot_df[target_col], oot_scores)]:
        fpr, tpr, _ = roc_curve(y, -scores)
        ax.plot(fpr, tpr, label=label, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test vs Out-of-Time")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_score_distribution(train_scores, test_scores, oot_scores):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for label, s in [("Train", train_scores), ("Test", test_scores), ("OOT", oot_scores)]:
        ax.hist(s, bins=30, alpha=0.5, label=label, density=True)
    ax.set_xlabel("Scorecard points")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by Sample")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_calibration(calib_table: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    x = range(len(calib_table))
    ax.plot(x, calib_table["mean_predicted_pd"], "o-", label="Predicted PD")
    ax.plot(x, calib_table["observed_default_rate"], "s-", label="Observed default rate")
    ax.set_xlabel("PD decile (low risk -> high risk)")
    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_psi(psi_table: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    x = np.arange(len(psi_table))
    w = 0.35
    ax.bar(x - w / 2, psi_table["expected_pct"], width=w, label="Expected (train)")
    ax.bar(x + w / 2, psi_table["actual_pct"], width=w, label="Actual")
    ax.set_xticks(x)
    ax.set_xticklabels(psi_table["bucket"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("% of population")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_vintage_curves(vintage_table: pd.DataFrame, max_mob: int = 18):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    cols = [c for c in vintage_table.columns if c <= max_mob]
    for vintage in vintage_table.index[::3]:  # subsample vintages for legibility
        ax.plot(cols, vintage_table.loc[vintage, cols], marker="o", markersize=2, label=vintage)
    ax.set_xlabel("Months on book")
    ax.set_ylabel("Cumulative default rate")
    ax.set_title("Vintage Curves — Cumulative Default Rate by Cohort")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    fig.tight_layout()
    return fig


def plot_points_by_feature(points_table: pd.DataFrame, iv_table: pd.DataFrame, top_n: int = 6):
    top_feats = iv_table.sort_values("iv", ascending=False)["feature"].head(top_n).tolist()
    top_feats = [f for f in top_feats if f in points_table["feature"].unique()]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, feat in zip(axes.flat, top_feats):
        sub = points_table[points_table["feature"] == feat]
        ax.bar(range(len(sub)), sub["points"], color="#3b6ea5")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["bin"], rotation=60, ha="right", fontsize=6)
        ax.set_title(feat, fontsize=10)
        ax.set_ylabel("points")
    for ax in axes.flat[len(top_feats):]:
        ax.axis("off")
    fig.suptitle("Scorecard Points by Feature (top features by IV)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Main report builder
# --------------------------------------------------------------------------- #
def build_report(
    output_path: str,
    df_meta: dict,
    binner,
    scorecard,
    suite_results: dict,
    train_df, test_df, oot_df, target_col,
    loss_results: dict,
) -> str:
    r = suite_results
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    ks_test = r["discrimination"]["test"]["ks"]
    ks_oot = r["discrimination"]["oot"]["ks"]
    gini_test = r["discrimination"]["test"]["gini"]
    psi_test = r["psi"]["test"]["psi"]
    psi_oot = r["psi"]["oot"]["psi"]
    fairness = r["fairness"]

    def status_badge(ok: bool) -> str:
        return '<span class="badge pass">PASS</span>' if ok else '<span class="badge fail">REVIEW</span>'

    overall_checks = [
        ("Discrimination power (test KS >= 20)", ks_test >= 20),
        ("Discrimination power (OOT KS >= 20)", ks_oot >= 20),
        ("Population stability (test PSI < 0.25)", psi_test < 0.25),
        ("Population stability (OOT PSI < 0.25)", psi_oot < 0.25),
        ("Fairness — four-fifths rule", fairness["passes_four_fifths_rule"]),
        ("No unexpected coefficient signs", scorecard.coef_table_["expected_sign_ok"].all()),
    ]

    # --- plots ---
    roc_fig = _fig_to_b64(plot_roc(test_df, r["scores"]["test"], oot_df, r["scores"]["oot"], target_col))
    dist_fig = _fig_to_b64(plot_score_distribution(r["scores"]["train"], r["scores"]["test"], r["scores"]["oot"]))
    calib_test_fig = _fig_to_b64(plot_calibration(r["calibration"]["test"], "Calibration — Test"))
    calib_oot_fig = _fig_to_b64(plot_calibration(r["calibration"]["oot"], "Calibration — OOT"))
    psi_test_fig = _fig_to_b64(plot_psi(r["psi"]["test"]["table"], "PSI — Train vs Test"))
    psi_oot_fig = _fig_to_b64(plot_psi(r["psi"]["oot"]["table"], "PSI — Train vs OOT"))
    points_fig = _fig_to_b64(plot_points_by_feature(scorecard.points_table_, binner.iv_table()))
    vintage_fig = _fig_to_b64(plot_vintage_curves(loss_results["vintage_table"]))

    checks_html = "".join(
        f"<tr><td>{name}</td><td>{status_badge(ok)}</td></tr>" for name, ok in overall_checks
    )
    coef_df = scorecard.coef_table_.reset_index().rename(columns={"index": "feature"})

    html = f"""
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Credit Risk Scorecard — Model Validation Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; padding: 0 0 60px;
          background: #f7f8fa; color: #1c2530; }}
  header {{ background: #10233f; color: white; padding: 32px 48px; }}
  header h1 {{ margin: 0 0 6px; font-size: 24px; }}
  header p {{ margin: 0; color: #b9c6d6; font-size: 13px; }}
  main {{ max-width: 1080px; margin: 0 auto; padding: 0 24px; }}
  section {{ background: white; border-radius: 8px; padding: 24px 28px; margin-top: 22px;
             box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
  h2 {{ font-size: 17px; border-bottom: 2px solid #eef1f4; padding-bottom: 10px; margin-top: 0; }}
  h3 {{ font-size: 14px; color: #3b4b5c; }}
  p.note {{ color: #5b6b7c; font-size: 13px; }}
  table.tbl {{ border-collapse: collapse; width: 100%; font-size: 12.5px; margin: 10px 0; }}
  table.tbl th {{ background: #eef2f6; text-align: left; padding: 6px 10px; }}
  table.tbl td {{ padding: 5px 10px; border-bottom: 1px solid #f0f0f0; }}
  table.tbl tr:hover {{ background: #fafbfc; }}
  .badge {{ padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }}
  .badge.pass {{ background: #e2f5e9; color: #1c7c3f; }}
  .badge.fail {{ background: #fbe6e6; color: #b3261e; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .kpi-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .kpi {{ background: #f4f7fa; border-radius: 8px; padding: 14px 18px; min-width: 150px; }}
  .kpi .val {{ font-size: 22px; font-weight: 700; color: #10233f; }}
  .kpi .lbl {{ font-size: 11.5px; color: #64748b; text-transform: uppercase; letter-spacing: .03em; }}
  img {{ max-width: 100%; }}
</style></head>
<body>
<header>
  <h1>Credit Risk Scorecard — Model Validation Report</h1>
  <p>Generated {now} &nbsp;|&nbsp; Portfolio: {df_meta['n_loans']:,} loans
     &nbsp;|&nbsp; Overall default rate: {df_meta['default_rate']:.1%}
     &nbsp;|&nbsp; Vintages: {df_meta['date_range']}</p>
</header>
<main>

<section>
  <h2>1. Overall Validation Outcome</h2>
  <table class="tbl"><thead><tr><th>Check</th><th>Result</th></tr></thead>
  <tbody>{checks_html}</tbody></table>
  <p class="note">Thresholds follow standard industry rules of thumb (KS &gt;= 20 acceptable discrimination,
  PSI &lt; 0.25 no material population shift, four-fifths rule for adverse impact). A real MRM sign-off
  would document these thresholds' provenance and require independent second-line review.</p>
</section>

<section>
  <h2>2. Feature Selection — Information Value</h2>
  <p class="note">Features with IV &lt; 0.02 were excluded before modeling (standard practice — they carry
  effectively no univariate predictive signal).</p>
  {_df_html(binner.iv_table())}
</section>

<section>
  <h2>3. Logistic Regression — Coefficients &amp; Model Design Checks</h2>
  <div class="grid2">
    <div><h3>Coefficients (target: P(good))</h3>{_df_html(coef_df)}</div>
    <div><h3>Variance Inflation Factor (multicollinearity)</h3>{_df_html(scorecard.vif_table_)}</div>
  </div>
</section>

<section>
  <h2>4. Scorecard — Points by Feature</h2>
  <img src="data:image/png;base64,{points_fig}">
  {_df_html(scorecard.points_table_, max_rows=40)}
</section>

<section>
  <h2>5. Discrimination Power</h2>
  <div class="kpi-row">
    <div class="kpi"><div class="val">{ks_test:.1f}</div><div class="lbl">KS (test)</div></div>
    <div class="kpi"><div class="val">{ks_oot:.1f}</div><div class="lbl">KS (OOT)</div></div>
    <div class="kpi"><div class="val">{gini_test:.3f}</div><div class="lbl">Gini (test)</div></div>
    <div class="kpi"><div class="val">{r['discrimination']['test']['auc']:.3f}</div><div class="lbl">AUC (test)</div></div>
  </div>
  <div class="grid2" style="margin-top:16px;">
    <img src="data:image/png;base64,{roc_fig}">
    <img src="data:image/png;base64,{dist_fig}">
  </div>
</section>

<section>
  <h2>6. Calibration</h2>
  <div class="grid2">
    <div><img src="data:image/png;base64,{calib_test_fig}">{_df_html(r['calibration']['test'])}</div>
    <div><img src="data:image/png;base64,{calib_oot_fig}">{_df_html(r['calibration']['oot'])}</div>
  </div>
</section>

<section>
  <h2>7. Population Stability Index</h2>
  <div class="kpi-row">
    <div class="kpi"><div class="val">{psi_test:.4f}</div><div class="lbl">PSI train->test ({r['psi']['test']['flag']})</div></div>
    <div class="kpi"><div class="val">{psi_oot:.4f}</div><div class="lbl">PSI train->OOT ({r['psi']['oot']['flag']})</div></div>
  </div>
  <div class="grid2" style="margin-top:16px;">
    <img src="data:image/png;base64,{psi_test_fig}">
    <img src="data:image/png;base64,{psi_oot_fig}">
  </div>
</section>

<section>
  <h2>8. Fairness — Disparate Impact (four-fifths rule)</h2>
  <p class="note">{fairness['approval_rate_target']:.1%} approval rate for the target proxy group vs
  {fairness['approval_rate_reference']:.1%} for the reference group at a 70% overall approval rate
  &rarr; ratio = <b>{fairness['disparate_impact_ratio']:.3f}</b>
  ({'passes' if fairness['passes_four_fifths_rule'] else 'FAILS'} the 0.80 threshold).
  This uses a synthetic protected-class proxy for demonstration — see data_prep.py and GUIDE.md.</p>
</section>

<section>
  <h2>9. Loss Forecasting</h2>
  <h3>Vintage curves — cumulative default rate by cohort</h3>
  <img src="data:image/png;base64,{vintage_fig}">
  <h3>Roll-rate matrix (month-over-month state transitions)</h3>
  {_df_html(loss_results['roll_rate'].reset_index())}
  <h3>Expected loss by grade (PD x LGD x EAD)</h3>
  {_df_html(loss_results['expected_loss'])}
</section>

</main>
</body></html>
"""
    with open(output_path, "w") as f:
        f.write(html)
    return output_path