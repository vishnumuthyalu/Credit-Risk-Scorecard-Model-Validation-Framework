"""
loss_forecast.py
=================
Vintage analysis, roll-rate transition matrices, and expected-loss
projection — the standard toolkit a bank credit-risk / loss-forecasting
team uses on top of a scorecard to answer "how much will this portfolio
lose, and when."

Note on data: the public Lending Club "accepted loans" file only records
each loan's *final* outcome (Fully Paid / Charged Off), not a month-by-month
payment history. Real vintage and roll-rate analysis needs that monthly
panel. `simulate_monthly_performance()` builds a realistic synthetic
monthly panel *consistent with each loan's known final outcome* (loans that
defaulted pass through 30 -> 60 -> 90 DPD -> Charged Off before their known
outcome; loans that were fully paid stay Current) so the vintage-curve and
roll-rate machinery below can be demonstrated exactly as it's used in
practice. If you have access to a real monthly loan-level payment panel,
swap `simulate_monthly_performance` for a loader over that data — every
downstream function only needs the same long-format schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STATES = ["current", "30dpd", "60dpd", "90dpd", "charged_off", "paid_off"]
DELINQUENCY_PATH = ["current", "30dpd", "60dpd", "90dpd", "charged_off"]


def simulate_monthly_performance(
    df: pd.DataFrame,
    loan_amnt_col: str = "loan_amnt",
    term_col: str = "term",
    issue_col: str = "issue_d",
    default_col: str = "default",
    seed: int = 11,
) -> pd.DataFrame:
    """
    Build a long-format monthly performance panel consistent with each
    loan's known final outcome. Returns columns:
        loan_id, issue_d, month_on_book, calendar_month, status, loan_amnt
    """
    rng = np.random.default_rng(seed)
    term_months = df[term_col].str.extract(r"(\d+)").astype(int)[0].values

    rows = []
    for i, (idx, row) in enumerate(df.iterrows()):
        term = term_months[i]
        loan_id = idx
        issue = pd.Timestamp(row[issue_col])
        is_default = int(row[default_col]) == 1

        if is_default:
            # Charge-off timing: right-skewed, peaking early-mid life —
            # consistent with typical unsecured consumer-loan loss curves.
            month_default = int(np.clip(rng.gamma(shape=3.2, scale=3.5), 3, term))
            # Roll through delinquency buckets in the 3 months before charge-off.
            pre_states_start = max(month_default - 3, 0)
            for m in range(0, pre_states_start):
                rows.append((loan_id, issue, m, issue + pd.DateOffset(months=m),
                             "current", row[loan_amnt_col]))
            path = DELINQUENCY_PATH[1:]  # 30dpd,60dpd,90dpd,charged_off
            for j, m in enumerate(range(pre_states_start, month_default + 1)):
                state = path[min(j, len(path) - 1)]
                rows.append((loan_id, issue, m, issue + pd.DateOffset(months=m),
                             state, row[loan_amnt_col]))
        else:
            # Fully-paid loans: current every month, paid_off at term end.
            for m in range(0, term):
                state = "paid_off" if m == term - 1 else "current"
                rows.append((loan_id, issue, m, issue + pd.DateOffset(months=m),
                             state, row[loan_amnt_col]))

    panel = pd.DataFrame(rows, columns=[
        "loan_id", "issue_d", "month_on_book", "calendar_month", "status", "loan_amnt",
    ])
    return panel


# --------------------------------------------------------------------------- #
# Vintage analysis
# --------------------------------------------------------------------------- #
def vintage_curve_table(panel: pd.DataFrame, observation_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Cumulative default rate by vintage (issue month) x months-on-book (MOB).
    Cells where the vintage hasn't yet seasoned to that MOB as of
    `observation_date` are left NaN (right-censoring) rather than
    silently treated as zero-loss — this is the detail that trips up naive
    vintage analyses and the exact reason a real team plots this as a
    triangle, not a full rectangle.
    """
    if observation_date is None:
        observation_date = panel["calendar_month"].max()

    panel = panel.copy()
    panel["is_defaulted_ever"] = panel["status"] == "charged_off"

    vintages = sorted(panel["issue_d"].unique())
    max_mob = panel["month_on_book"].max()

    out = {}
    for v in vintages:
        v = pd.Timestamp(v)
        sub = panel[panel["issue_d"] == v]
        n_loans = sub["loan_id"].nunique()
        months_available = (observation_date.year - v.year) * 12 + (observation_date.month - v.month)

        curve = []
        cum_defaults = set()
        for mob in range(0, max_mob + 1):
            if mob > months_available:
                curve.append(np.nan)
                continue
            newly = sub[(sub["month_on_book"] == mob) & (sub["status"] == "charged_off")]["loan_id"]
            cum_defaults.update(newly.tolist())
            curve.append(len(cum_defaults) / n_loans if n_loans else np.nan)
        out[v.strftime("%Y-%m")] = curve

    return pd.DataFrame(out, index=range(0, max_mob + 1)).T  # vintages x MOB


# --------------------------------------------------------------------------- #
# Roll-rate matrix
# --------------------------------------------------------------------------- #
def roll_rate_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Month-over-month state transition probability matrix, pooled across all
    loans and months. Rows = state this month, columns = state next month.
    The 30dpd -> 60dpd/90dpd "roll forward" rates are the numbers a
    collections/loss-forecasting team watches most closely as an early
    warning signal.
    """
    panel = panel.sort_values(["loan_id", "month_on_book"])
    panel["next_status"] = panel.groupby("loan_id")["status"].shift(-1)
    trans = panel.dropna(subset=["next_status"])

    counts = pd.crosstab(trans["status"], trans["next_status"])
    counts = counts.reindex(index=STATES, columns=STATES, fill_value=0)
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return probs.round(3)


# --------------------------------------------------------------------------- #
# Expected loss projection
# --------------------------------------------------------------------------- #
def expected_loss_table(
    df: pd.DataFrame,
    pd_col: str,
    ead_col: str = "loan_amnt",
    lgd: float = 0.65,
    segment_col: str = "grade",
) -> pd.DataFrame:
    """
    Standard EL = PD x LGD x EAD decomposition, aggregated by segment
    (typically grade or score band). `lgd` defaults to 0.65 — a commonly
    cited industry loss-given-default assumption for unsecured consumer
    credit; in practice this would come from the bank's own recovery data.
    """
    tmp = df.copy()
    tmp["expected_loss"] = tmp[pd_col] * lgd * tmp[ead_col]

    agg = tmp.groupby(segment_col).agg(
        n_loans=("expected_loss", "size"),
        total_exposure=(ead_col, "sum"),
        avg_pd=(pd_col, "mean"),
        total_expected_loss=("expected_loss", "sum"),
    ).reset_index()
    agg["expected_loss_rate"] = agg["total_expected_loss"] / agg["total_exposure"]
    agg = agg.sort_values("avg_pd").reset_index(drop=True)

    total_row = pd.DataFrame([{
        segment_col: "TOTAL PORTFOLIO",
        "n_loans": agg["n_loans"].sum(),
        "total_exposure": agg["total_exposure"].sum(),
        "avg_pd": np.average(agg["avg_pd"], weights=agg["n_loans"]),
        "total_expected_loss": agg["total_expected_loss"].sum(),
        "expected_loss_rate": agg["total_expected_loss"].sum() / agg["total_exposure"].sum(),
    }])
    return pd.concat([agg, total_row], ignore_index=True)


if __name__ == "__main__":
    from data_prep import load_data

    # Sample down for a quick smoke test — simulate_monthly_performance
    # loops per-loan, so the full multi-million-row file is slow here.
    df = load_data()
    df = df.sample(n=min(6000, len(df)), random_state=0).reset_index(drop=True)
    panel = simulate_monthly_performance(df)
    print("Panel rows:", len(panel))

    vc = vintage_curve_table(panel)
    print("\nVintage cumulative default-rate curve (first 5 vintages, MOB 0-12):")
    print(vc.iloc[:5, :13].round(3))

    rr = roll_rate_matrix(panel)
    print("\nRoll-rate matrix:")
    print(rr)

    el = expected_loss_table(df.assign(naive_pd=df["default"]), pd_col="naive_pd")
    print("\nExpected loss by grade (using realized default as a PD proxy for this smoke test):")
    print(el.round(4))