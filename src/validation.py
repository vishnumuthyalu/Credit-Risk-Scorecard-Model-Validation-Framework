"""
validation.py
==============
The independent model-validation battery a bank's Model Risk Management
(MRM) function runs before (and periodically after) a scorecard goes into
production. This is the module that turns "I built a model" into "I built
and validated a model" — directly answering the JD's "apply statistical
and quantitative techniques to validate model design, calibration, and
implementation."

Five checks, each a real MRM standard:
  1. Discrimination power  — KS statistic & Gini coefficient (AUC-based):
     can the model actually separate good loans from bad ones?
  2. Calibration            — do predicted PDs match observed default rates?
  3. Population stability   — PSI: has the scored population drifted from
     the population the model was built on (a leading indicator that a
     model needs to be redeveloped)?
  4. Fairness / disparate impact — the four-fifths (80%) rule borrowed from
     EEOC guidance and commonly applied in fair-lending model review.
  5. Out-of-time performance — everything above, re-run on a vintage the
     model never saw during development (the OOT sample from data_prep).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


# --------------------------------------------------------------------------- #
# 1. Discrimination power
# --------------------------------------------------------------------------- #
def ks_statistic(y_true: np.ndarray, risk_score: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic: the max separation between the cumulative
    distributions of `risk_score` for bads vs goods. `risk_score` must be
    oriented so HIGHER = RISKIER (e.g. predicted PD, or -scorecard_points).
    Industry rule of thumb: KS > 40 is a strong scorecard, 20-40 acceptable,
    < 20 weak.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(risk_score)
    y_sorted = y_true[order]
    cum_bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)) * 100)


def gini_auc(y_true: np.ndarray, risk_score: np.ndarray) -> tuple[float, float]:
    """Returns (gini, auc). `risk_score` oriented HIGHER = RISKIER."""
    auc = roc_auc_score(y_true, risk_score)
    gini = 2 * auc - 1
    return float(gini), float(auc)


def roc_points(y_true: np.ndarray, risk_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fpr, tpr, _ = roc_curve(y_true, risk_score)
    return fpr, tpr


# --------------------------------------------------------------------------- #
# 2. Calibration
# --------------------------------------------------------------------------- #
def calibration_table(y_true: np.ndarray, pd_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Bins by predicted PD decile and compares mean predicted PD to observed
    default rate per bin — the standard reliability/calibration check. A
    well-calibrated model has predicted ~= observed in every bin.
    """
    df = pd.DataFrame({"y": y_true, "pd_pred": pd_pred})
    df["bin"] = pd.qcut(df["pd_pred"], q=n_bins, duplicates="drop")
    out = df.groupby("bin", observed=True).agg(
        n=("y", "size"), mean_predicted_pd=("pd_pred", "mean"), observed_default_rate=("y", "mean"),
    ).reset_index()
    out["abs_gap"] = (out["mean_predicted_pd"] - out["observed_default_rate"]).abs()
    return out


# --------------------------------------------------------------------------- #
# 3. Population Stability Index
# --------------------------------------------------------------------------- #
def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> tuple[float, pd.DataFrame]:
    """
    PSI between a reference distribution (`expected`, typically train/dev
    scores) and a comparison distribution (`actual`, typically test/OOT or
    a later production month). Bucket edges are derived from `expected`'s
    deciles so both distributions are compared on the same scale.

    Interpretation (industry standard):
        < 0.10   no significant population shift
        0.10-0.25  moderate shift — investigate
        > 0.25   significant shift — model likely needs review/redevelopment
    """
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    if len(edges) < 3:
        edges = np.array([-np.inf, np.median(expected), np.inf])

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)
    exp_pct = np.maximum(exp_counts / exp_counts.sum(), 1e-6)
    act_pct = np.maximum(act_counts / act_counts.sum(), 1e-6)

    psi_per_bucket = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    table = pd.DataFrame({
        "bucket": [f"({edges[i]:.0f}, {edges[i+1]:.0f}]" for i in range(len(edges) - 1)],
        "expected_pct": exp_pct, "actual_pct": act_pct, "psi_contribution": psi_per_bucket,
    })
    return float(psi_per_bucket.sum()), table


def psi_flag(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate shift — investigate"
    return "significant shift — review model"


# --------------------------------------------------------------------------- #
# 4. Fairness / disparate impact
# --------------------------------------------------------------------------- #
def disparate_impact_ratio(
    scores: np.ndarray,
    group: np.ndarray,
    target_group: str,
    reference_group: str,
    approval_rate: float = 0.70,
) -> dict:
    """
    Simulates an approval decision by approving the top `approval_rate`
    share of applicants by score, then compares the approval rate for
    `target_group` vs `reference_group` — the "four-fifths rule" adverse-
    impact test (ratio < 0.8 is the traditional adverse-impact trigger,
    borrowed from EEOC employment-discrimination guidance and widely used
    as a first-pass fair-lending screen).

    NOTE: this project's `proxy_group` field is a synthetic stand-in for a
    protected-class proxy (see data_prep.py) — real fair-lending testing
    would use a proper BISG-derived proxy or, ideally, self-reported data
    where legally available, and would sit alongside legal/compliance
    review, not replace it.
    """
    cutoff = np.quantile(scores, 1 - approval_rate)
    approved = scores >= cutoff

    rate_target = approved[group == target_group].mean()
    rate_reference = approved[group == reference_group].mean()
    ratio = rate_target / rate_reference if rate_reference > 0 else np.nan

    return {
        "approval_rate_target": float(rate_target),
        "approval_rate_reference": float(rate_reference),
        "disparate_impact_ratio": float(ratio),
        "passes_four_fifths_rule": bool(ratio >= 0.8),
    }


# --------------------------------------------------------------------------- #
# ValidationSuite: run everything and package results
# --------------------------------------------------------------------------- #
class ValidationSuite:
    def __init__(self, scorecard):
        self.scorecard = scorecard
        self.results: dict = {}

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        oot_df: pd.DataFrame,
        target_col: str = "default",
        protected_col: str = "proxy_group",
        target_group: str = "group_B",
        reference_group: str = "group_A",
    ) -> dict:
        train_scores = self.scorecard.score(train_df)
        test_scores = self.scorecard.score(test_df)
        oot_scores = self.scorecard.score(oot_df)

        test_pd = self.scorecard.predict_pd(test_df)
        oot_pd = self.scorecard.predict_pd(oot_df)

        # Higher score = safer, so risk_score = -score for AUC/KS orientation.
        ks_test = ks_statistic(test_df[target_col].values, -test_scores)
        ks_oot = ks_statistic(oot_df[target_col].values, -oot_scores)
        gini_test, auc_test = gini_auc(test_df[target_col].values, -test_scores)
        gini_oot, auc_oot = gini_auc(oot_df[target_col].values, -oot_scores)

        psi_test, psi_test_table = calculate_psi(train_scores, test_scores)
        psi_oot, psi_oot_table = calculate_psi(train_scores, oot_scores)

        calib_test = calibration_table(test_df[target_col].values, test_pd)
        calib_oot = calibration_table(oot_df[target_col].values, oot_pd)

        fairness_test = disparate_impact_ratio(
            test_scores, test_df[protected_col].values, target_group, reference_group,
        )

        self.results = {
            "scores": {"train": train_scores, "test": test_scores, "oot": oot_scores},
            "discrimination": {
                "test": {"ks": ks_test, "gini": gini_test, "auc": auc_test},
                "oot": {"ks": ks_oot, "gini": gini_oot, "auc": auc_oot},
            },
            "psi": {
                "test": {"psi": psi_test, "flag": psi_flag(psi_test), "table": psi_test_table},
                "oot": {"psi": psi_oot, "flag": psi_flag(psi_oot), "table": psi_oot_table},
            },
            "calibration": {"test": calib_test, "oot": calib_oot},
            "fairness": fairness_test,
        }
        return self.results

    def summary(self) -> str:
        r = self.results
        lines = ["=== VALIDATION SUMMARY ==="]
        lines.append(f"Discrimination (test):  KS={r['discrimination']['test']['ks']:.1f}  "
                      f"Gini={r['discrimination']['test']['gini']:.3f}  "
                      f"AUC={r['discrimination']['test']['auc']:.3f}")
        lines.append(f"Discrimination (OOT):   KS={r['discrimination']['oot']['ks']:.1f}  "
                      f"Gini={r['discrimination']['oot']['gini']:.3f}  "
                      f"AUC={r['discrimination']['oot']['auc']:.3f}")
        lines.append(f"PSI (train->test):  {r['psi']['test']['psi']:.4f}  [{r['psi']['test']['flag']}]")
        lines.append(f"PSI (train->OOT):   {r['psi']['oot']['psi']:.4f}  [{r['psi']['oot']['flag']}]")
        f = r["fairness"]
        lines.append(f"Disparate impact ratio: {f['disparate_impact_ratio']:.3f}  "
                      f"(passes four-fifths rule: {f['passes_four_fifths_rule']})")
        return "\n".join(lines)


if __name__ == "__main__":
    from data_prep import load_data, time_based_split, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
    from woe_binning import WOEBinner
    from scorecard import Scorecard

    df = load_data()
    train, test, oot = time_based_split(df)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    binner = WOEBinner(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    binner.fit(train[features], train[TARGET])
    selected = binner.iv_table().pipe(lambda d: d[d["iv"] >= 0.02])["feature"].tolist()

    sc = Scorecard(binner, selected)
    sc.fit(binner.transform(train[features]), train[TARGET])

    suite = ValidationSuite(sc)
    suite.run(train, test, oot)
    print(suite.summary())