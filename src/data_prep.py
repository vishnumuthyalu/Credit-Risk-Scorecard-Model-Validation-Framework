"""
data_prep.py
============
Loads and prepares Lending Club's public "accepted loans" historical
dataset for credit scorecard development.

This module loads REAL DATA ONLY — no synthetic fallback. Download
instructions are in GUIDE.md, Section 4. Point `load_data()` (or
`load_lending_club()` directly) at the CSV and it will select, clean, and
engineer the standard scorecard fields used by every downstream module
(woe_binning, scorecard, loss_forecast, validation).

Note: tests/test_pipeline.py uses its own small, self-contained synthetic
data generator so the test suite runs fast and doesn't depend on the real
file being present. That generator intentionally lives in the test file,
not here — this module stays focused on the one dataset it's meant to run
against.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Columns every downstream module expects to exist after prep.
NUMERIC_FEATURES = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "revol_util",
    "delinq_2yrs",
    "open_acc",
    "pub_rec",
    "total_acc",
    "fico_score",
    "emp_length_years",
    "credit_history_years",
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "purpose",
    "verification_status",
    "grade",
    "term",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Name of the target column: 1 = defaulted / charged off, 0 = fully paid.
TARGET = "default"

# Synthetic proxy for a protected-class attribute. Real underwriting data
# never contains race/ethnicity/sex directly (fair-lending law forbids using
# them as underwriting inputs), so real-world fair-lending testing infers a
# proxy group, most commonly via BISG (Bayesian Improved Surname Geocoding).
# Lending Club's public file has no such field, so we assign one at random
# purely so the validation module has something to check — replace this
# with a real BISG-derived proxy if you have one available.
PROTECTED_GROUP = "proxy_group"

# Where run_pipeline.py and every module's __main__ smoke test look for the
# CSV by default. See GUIDE.md, Section 4, for how to get the file there.
DEFAULT_DATA_PATH = "data/loan.csv"


# --------------------------------------------------------------------------- #
# Lending Club loader
# --------------------------------------------------------------------------- #
def load_lending_club(csv_path: str) -> pd.DataFrame:
    """
    Load and clean the real Lending Club 'accepted loans' CSV.

    Expects the standard Lending Club column names (loan_amnt, int_rate,
    annual_inc, dti, revol_util, delinq_2yrs, open_acc, pub_rec, total_acc,
    fico_range_low, fico_range_high, emp_length, earliest_cr_line,
    home_ownership, purpose, verification_status, grade, term, issue_d,
    loan_status). Download instructions are in GUIDE.md, Section 4.
    """
    usecols = [
        "loan_amnt", "int_rate", "annual_inc", "dti", "revol_util",
        "delinq_2yrs", "open_acc", "pub_rec", "total_acc",
        "fico_range_low", "fico_range_high", "emp_length",
        "earliest_cr_line", "home_ownership", "purpose",
        "verification_status", "grade", "term", "issue_d", "loan_status",
    ]
    df = pd.read_csv(csv_path, usecols=lambda c: c in usecols, low_memory=False)

    # --- Target: keep only loans with a resolved outcome (industry standard
    # practice — 'Current' loans haven't matured and can't yet be labeled) ---
    bad_status = {
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
    }
    good_status = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
    df = df[df["loan_status"].isin(bad_status | good_status)].copy()
    df[TARGET] = df["loan_status"].isin(bad_status).astype(int)

    # --- Feature engineering ---
    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2.0
    df["int_rate"] = (
        df["int_rate"].astype(str).str.replace("%", "", regex=False).astype(float)
    )
    df["revol_util"] = (
        df["revol_util"].astype(str).str.replace("%", "", regex=False).astype(float)
    )
    df["term"] = df["term"].astype(str).str.strip()

    emp_map = {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
        "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
        "10+ years": 10,
    }
    df["emp_length_years"] = df["emp_length"].map(emp_map)

    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    earliest = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_history_years"] = (
        (df["issue_d"] - earliest).dt.days / 365.25
    ).clip(lower=0)

    # Lending Club's public file has no protected-class proxy. We assign one
    # at random purely so the validation module has something to check —
    # replace this with a real BISG-derived proxy if you have one available.
    rng = np.random.default_rng(42)
    df[PROTECTED_GROUP] = rng.choice(["group_A", "group_B"], size=len(df))

    df = df.dropna(subset=NUMERIC_FEATURES + [TARGET, "issue_d"])
    return df[ALL_FEATURES + [TARGET, "issue_d", PROTECTED_GROUP]].reset_index(drop=True)


def load_data(csv_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load and clean the real Lending Club CSV at `csv_path`.

    Raises FileNotFoundError (with a pointer to GUIDE.md, Section 4) if the
    file isn't there yet. This project intentionally does not fall back to
    synthetic data for the pipeline itself — every number the report
    produces should be traceable to the real historical portfolio. (The
    test suite is the one place synthetic data is still used, and it
    generates its own fixture locally rather than calling this function.)
    """
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Lending Club CSV not found at '{csv_path}'. Download it and place it "
            f"there (see GUIDE.md, Section 4), or pass a different path, e.g. "
            f"python src/run_pipeline.py --data /path/to/your/loan.csv"
        )
    print(f"[data_prep] Loading Lending Club data from {csv_path}")
    return load_lending_club(csv_path)


def time_based_split(
    df: pd.DataFrame,
    date_col: str = "issue_d",
    train_frac: float = 0.6,
    test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by vintage (origination date), not randomly. This mirrors how a
    real credit risk team validates a scorecard: train on the oldest
    cohorts, test on the next slice, and hold out the most recent cohort as
    an "out-of-time" (OOT) sample — the closest proxy available without
    live data for how the model will behave on business it hasn't seen yet.
    Returns (train, test, oot).
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    test_end = int(n * (train_frac + test_frac))
    train = df.iloc[:train_end].reset_index(drop=True)
    test = df.iloc[train_end:test_end].reset_index(drop=True)
    oot = df.iloc[test_end:].reset_index(drop=True)
    return train, test, oot


if __name__ == "__main__":
    data = load_data()
    print(data.shape)
    print(data[TARGET].mean(), "default rate")
    tr, te, oo = time_based_split(data)
    print(f"train={len(tr)} test={len(te)} oot={len(oo)}")