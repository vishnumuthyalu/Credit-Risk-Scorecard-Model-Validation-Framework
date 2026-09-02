"""
scorecard.py
============
Fits a logistic-regression scorecard on WOE-transformed features and
converts it into a traditional points-based scorecard using the standard
PDO (points-to-double-odds) scaling formula used across the credit
industry (FICO, Experian, and internal bank scorecards all use variants of
this same transform).

Design choices, and why:
  * We fit statsmodels' Logit (not just sklearn) because it gives
    coefficient p-values and standard errors for free — a real model-risk
    team's "model design validation" step includes checking that every
    variable is statistically significant and has the expected sign
    (e.g. higher FICO should *reduce* predicted default risk; a variable
    with the "wrong" sign is a red flag investigated before deployment).
  * We regress on P(good) rather than P(bad) so that positive coefficients
    and higher WOE both mean "safer," matching the sign convention used in
    Naeem Siddiqi's "Credit Risk Scorecards" — the standard industry
    reference for this technique.
  * We report Variance Inflation Factors (VIF) per feature, the standard
    multicollinearity check before trusting individual coefficients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from woe_binning import WOEBinner

# `grade` and `int_rate` showed the highest IV of any feature by a wide
# margin once I exectued woe_binning.py on the real data (frequently 0.4-0.5+,
# with VIF ~7-8 against each other).
# 
# That's not because they're "the best
# predictors" in the sense a from-scratch scorecard cares about, they're
# Lending Club's OWN underwriting model's output, assigned to the loan at
# origination, not something the applicant provided. A scorecard built on
# them mostly just re-derives Lending Club's own risk grade rather than
# learning genuine signal from raw applicant/bureau data.
# 
# The __main__ smoke test below (and run_pipeline.py) excludes these from the feature list by
# default, while still showing their IV for context, so the resulting
# scorecard is an independent risk assessment, not a circular one.
UNDERWRITER_ASSIGNED_FEATURES = ["grade", "int_rate"]


class Scorecard:
    def __init__(
        self,
        binner: WOEBinner,
        features: list[str],
        pdo: float = 20.0,
        base_score: float = 600.0,
        base_odds: float = 50.0,
    ):
        """
        pdo         : points to double the odds (industry-standard default 20)
        base_score  : score at which good:bad odds equal `base_odds`
        base_odds   : the good:bad odds anchor point (e.g. 50 means 50 good
                      per 1 bad at `base_score`)
        """
        self.binner = binner
        self.features = features
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

        self.model_ = None
        self.coef_table_ = None
        self.points_table_ = None
        self.vif_table_ = None

    # ------------------------------------------------------------------ #
    def fit(self, X_woe: pd.DataFrame, y_default: pd.Series) -> "Scorecard":
        """
        X_woe       : WOE-transformed features (output of WOEBinner.transform)
        y_default   : 1 = default/bad, 0 = fully paid/good
        """
        y_good = 1 - y_default.astype(int).values
        X = sm.add_constant(X_woe[self.features])

        self.model_ = sm.Logit(y_good, X).fit(disp=False)

        summary = pd.DataFrame({
            "coef": self.model_.params,
            "std_err": self.model_.bse,
            "p_value": self.model_.pvalues,
        })
        summary["significant_5pct"] = summary["p_value"] < 0.05
        summary["expected_sign_ok"] = summary["coef"] >= 0  # WOE-encoded -> should be positive
        summary.loc["const", "expected_sign_ok"] = True
        self.coef_table_ = summary

        # VIF (skip the constant column)
        vif_data = []
        Xv = X_woe[self.features].values
        for i, feat in enumerate(self.features):
            vif_data.append({"feature": feat, "vif": variance_inflation_factor(Xv, i)})
        self.vif_table_ = pd.DataFrame(vif_data).sort_values("vif", ascending=False)

        self._build_points_table()
        return self

    # ------------------------------------------------------------------ #
    def _build_points_table(self):
        alpha = self.model_.params["const"]
        n = len(self.features)
        base_points_per_feature = (self.offset + self.factor * alpha) / n

        rows = []
        for feat in self.features:
            beta = self.model_.params[feat]
            bin_table = self.binner.bin_detail(feat)
            for _, r in bin_table.iterrows():
                pts = base_points_per_feature + self.factor * beta * r["woe"]
                rows.append({
                    "feature": feat, "bin": r["bin"], "woe": r["woe"],
                    "count": r["count"], "bad_rate": r["bad_rate"],
                    "points": round(pts, 1),
                })
        self.points_table_ = pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    def score(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Compute the integer scorecard points for each row of raw feature data."""
        X_woe = self.binner.transform(X_raw[self.features])
        alpha = self.model_.params["const"]
        n = len(self.features)
        base_points_per_feature = (self.offset + self.factor * alpha) / n

        total = np.full(len(X_raw), base_points_per_feature * n)
        for feat in self.features:
            beta = self.model_.params[feat]
            total = total + self.factor * beta * X_woe[feat].values
        return np.round(total).astype(int)

    def predict_pd(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Predicted probability of default (PD) implied by the score."""
        score = self.score(X_raw)
        odds_good = np.exp((score - self.offset) / self.factor)
        p_good = odds_good / (1 + odds_good)
        return 1 - p_good

    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        lines = ["=== Logistic Regression Coefficients (target = P(good)) ==="]
        lines.append(self.coef_table_.round(4).to_string())
        lines.append("\n=== Variance Inflation Factors (multicollinearity check) ===")
        lines.append(self.vif_table_.round(2).to_string(index=False))
        flags = self.coef_table_[~self.coef_table_["expected_sign_ok"]]
        if len(flags):
            lines.append(f"\n WARNING: {len(flags)} feature(s) have an unexpected sign — "
                          f"investigate before using in production: {list(flags.index)}")
        high_vif = self.vif_table_[self.vif_table_["vif"] > 5]
        if len(high_vif):
            lines.append(f"\n WARNING: {len(high_vif)} feature(s) have VIF > 5 "
                          f"(possible multicollinearity): {list(high_vif['feature'])}")
        return "\n".join(lines)


if __name__ == "__main__":
    from data_prep import load_data, time_based_split, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET

    df = load_data()
    train, test, oot = time_based_split(df)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    binner = WOEBinner(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    binner.fit(train[features], train[TARGET])

    # Feature selection by IV — drop weak features, mirroring real practice.
    iv = binner.iv_table()
    selected = iv[iv["iv"] >= 0.02]["feature"].tolist()

    # Exclude underwriter-assigned features (grade, int_rate) from the model
    # itself -- see UNDERWRITER_ASSIGNED_FEATURES above for why.
    excluded = [f for f in selected if f in UNDERWRITER_ASSIGNED_FEATURES]
    selected = [f for f in selected if f not in UNDERWRITER_ASSIGNED_FEATURES]
    if excluded:
        print(f"Excluding underwriter-assigned features from the scorecard "
              f"(high IV, but not applicant-provided): {excluded}")
    print(f"Selected {len(selected)} / {len(features)} features by IV >= 0.02")

    sc = Scorecard(binner, selected)
    train_woe = binner.transform(train[features])
    sc.fit(train_woe, train[TARGET])
    print(sc.summary())

    test_scores = sc.score(test[features])
    print("\nScore distribution (test set):")
    print(pd.Series(test_scores).describe())
    print("\nSample points table:")
    print(sc.points_table_.head(10))