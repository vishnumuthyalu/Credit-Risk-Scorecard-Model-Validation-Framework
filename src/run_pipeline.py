"""
run_pipeline.py
================
End-to-end entry point: data -> WOE binning -> scorecard -> loss forecast
-> validation -> HTML report. Run this after setting up your environment
and downloading the real Lending Club dataset (see GUIDE.md, Section 4).

Usage:
    python src/run_pipeline.py                       # uses data/loan.csv
    python src/run_pipeline.py --data /path/to/loan.csv
"""

from __future__ import annotations

import argparse
import os

from data_prep import load_data, time_based_split, DEFAULT_DATA_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
from woe_binning import WOEBinner
from scorecard import Scorecard, UNDERWRITER_ASSIGNED_FEATURES
from loss_forecast import simulate_monthly_performance, vintage_curve_table, roll_rate_matrix, expected_loss_table
from validation import ValidationSuite
import report as report_mod


def main(csv_path: str, out_path: str, iv_threshold: float, include_underwriter_features: bool = False):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 1. Data
    df = load_data(csv_path=csv_path)
    train, test, oot = time_based_split(df)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    print(f"[pipeline] train={len(train)} test={len(test)} oot={len(oot)}  "
          f"overall default rate={df[TARGET].mean():.2%}")

    # 2. WOE binning + IV-based feature selection (fit on TRAIN only)
    binner = WOEBinner(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    binner.fit(train[features], train[TARGET])
    iv = binner.iv_table()
    selected = iv[iv["iv"] >= iv_threshold]["feature"].tolist()

    # Exclude underwriter-assigned features (grade, int_rate) by default --
    # see UNDERWRITER_ASSIGNED_FEATURES in scorecard.py for why. Pass
    # --include-underwriter-features to opt back in (e.g. to benchmark your
    # scorecard's discrimination power against LC's own grade).
    if not include_underwriter_features:
        excluded = [f for f in selected if f in UNDERWRITER_ASSIGNED_FEATURES]
        selected = [f for f in selected if f not in UNDERWRITER_ASSIGNED_FEATURES]
        if excluded:
            print(f"[pipeline] excluding underwriter-assigned features (high IV, but not "
                  f"applicant-provided) -- pass --include-underwriter-features to keep them: {excluded}")
    print(f"[pipeline] selected {len(selected)}/{len(features)} features "
          f"(IV >= {iv_threshold}): {selected}")

    # 3. Scorecard
    sc = Scorecard(binner, selected)
    train_woe = binner.transform(train[features])
    sc.fit(train_woe, train[TARGET])
    print("\n" + sc.summary() + "\n")

    # 4. Validation (test + out-of-time)
    suite = ValidationSuite(sc)
    suite.run(train, test, oot, target_col=TARGET)
    print(suite.summary())

    # 5. Loss forecasting
    print("[pipeline] simulating monthly performance panel for loss forecasting...")
    panel = simulate_monthly_performance(df)
    vintage_table = vintage_curve_table(panel)
    roll_rate = roll_rate_matrix(panel)
    pd_pred = sc.predict_pd(df[features])
    expected_loss = expected_loss_table(df.assign(_pd=pd_pred), pd_col="_pd", segment_col="grade")
    print("\nExpected loss by grade:\n", expected_loss.round(4))

    loss_results = {
        "vintage_table": vintage_table,
        "roll_rate": roll_rate,
        "expected_loss": expected_loss,
    }

    # 6. Report
    df_meta = {
        "n_loans": len(df),
        "default_rate": df[TARGET].mean(),
        "date_range": f"{df['issue_d'].min().strftime('%Y-%m')} to {df['issue_d'].max().strftime('%Y-%m')}",
    }
    path = report_mod.build_report(
        out_path, df_meta, binner, sc, suite.results, train, test, oot, TARGET, loss_results,
    )
    print(f"\n[pipeline] report written to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH,
                         help="Path to the real Lending Club CSV (see GUIDE.md, Section 4)")
    parser.add_argument("--out", type=str, default="outputs/validation_report.html")
    parser.add_argument("--iv-threshold", type=float, default=0.02)
    parser.add_argument("--include-underwriter-features", action="store_true",
                         help="Keep grade/int_rate in the scorecard (excluded by default -- "
                              "see GUIDE.md, Section 6 Step 3, for why)")
    args = parser.parse_args()
    try:
        main(args.data, args.out, args.iv_threshold, args.include_underwriter_features)
    except FileNotFoundError as e:
        print(f"[pipeline] {e}")
        raise SystemExit(1)